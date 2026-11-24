"""Deterministic open-loop predictions and physical baselines on complete episodes."""

from collections import Counter

import numpy as np
import torch

from .interface import episode_features, physical_fields, to_internal_action
from .replay import Replay


HORIZONS = (1, 5, 15, 50)
_VECTOR_ERRORS = {"position": "position_error_m", "velocity": "velocity_error_m_s",
                  "angular_velocity": "angular_velocity_error_rad_s"}
_ERROR_KEYS = (*_VECTOR_ERRORS.values(), "tilt_mae_deg", "fuel_mae_kg")


def _errors(estimated, actual):
    result = {key: np.linalg.norm(estimated[field].astype(np.float64) - actual[field], axis=-1)
              for field, key in _VECTOR_ERRORS.items()}
    valid = estimated["tilt_valid"] & actual["tilt_valid"]
    result["tilt_valid_count"] = valid
    result["tilt_mae_deg"] = np.where(valid, np.abs(estimated["tilt_deg"] - actual["tilt_deg"]), 0)
    result["fuel_mae_kg"] = np.abs(estimated["fuel"].astype(np.float64) - actual["fuel"])
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("world model yielded nonfinite physical errors")
    return result


def _mean(total, count):
    return float(total / count) if count else None


def _distribute(chunks):
    if not chunks:
        return {"count": 0, "mean": None, "min": None, "p05": None, "p25": None,
                "median": None, "p75": None, "p95": None, "max": None}
    values = np.concatenate(chunks)
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()
    return {"count": len(values), "mean": float(np.mean(values, dtype=np.float64)),
            **dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), quantiles))}


class _Row:
    """Accumulate the models and baselines on exactly the same endpoints."""
    def __init__(self, has_mean):
        self.physical = {k: Counter() for k in ("model", "persistence")}
        if has_mean:
            self.physical["train_mean"] = Counter()
        self.count = 0
        self.heads = Counter()
        self.continue_prob = {"terminal": [], "nonterminal": []}

    def add(self, errors, selected, prediction, reward, cont, done):
        count = int(selected.sum())
        if not count:
            return
        self.count += count
        for method, by_key in errors.items():
            for key, error in by_key.items():
                self.physical[method][key] += float(error[selected].sum(dtype=np.float64))
        for name, flag in (("terminal", done), ("nonterminal", ~done)):
            subset = selected & flag
            if subset.any():
                self.continue_prob[name].append(cont[subset])
                self.heads[name] += int(subset.sum())
                self.heads[name, "error"] += np.abs(prediction[subset] - reward[subset]).sum(dtype=np.float64)
                self.heads[name, "zero"] += np.abs(reward[subset]).sum(dtype=np.float64)
        end = cont < 0.5
        self.heads["tp"] += int((selected & done & end).sum())
        self.heads["fp"] += int((selected & ~done & end).sum())

    def summary(self):
        def physical(method):
            totals = self.physical[method]
            valid = int(totals["tilt_valid_count"])
            row = {"count": self.count, "tilt_valid_count": valid,
                   "tilt_invalid_count": self.count - valid}
            for key in _ERROR_KEYS:
                row[key] = _mean(totals[key], valid if key == "tilt_mae_deg" else self.count)
            return row

        heads = self.heads
        groups = {k: {"count": heads[k], "mae": _mean(heads[k, "error"], heads[k]),
                      "zero_reward_mae": _mean(heads[k, "zero"], heads[k])}
                  for k in self.continue_prob}
        tp, fp = heads["tp"], heads["fp"]
        return {
            **physical("model"),
            "baselines": {"persistence": physical("persistence"),
                          "train_mean": physical("train_mean") if "train_mean" in self.physical else None},
            "reward_mae": _mean(heads["terminal", "error"] + heads["nonterminal", "error"], self.count),
            "zero_reward_mae": _mean(heads["terminal", "zero"] + heads["nonterminal", "zero"], self.count),
            "reward_by_endpoint": groups,
            "terminal": {"tp": tp, "fp": fp, "tn": heads["nonterminal"] - fp,
                         "fn": heads["terminal"] - tp, "recall": _mean(tp, heads["terminal"]),
                         "false_positive_rate": _mean(fp, heads["nonterminal"])},
            "continue_probability": {k: _distribute(chunks) for k, chunks in self.continue_prob.items()},
        }


def _record(rows, predictions, current, target, true_reward, done, altitude, spec, mean_fields):
    features, reward, cont = (x.detach().float().cpu().numpy() for x in predictions)
    if reward.shape != true_reward.shape or cont.shape != done.shape:
        raise ValueError("heads must return one scalar per endpoint")
    if not np.isfinite(reward).all() or not np.all((cont >= 0) & (cont <= 1)):
        raise ValueError("world model returned invalid reward or continuation")
    actual = physical_fields(target[..., :16], spec)
    errors = {"model": _errors(physical_fields(features, spec), actual),
              "persistence": _errors(physical_fields(current[..., :16], spec), actual)}
    if mean_fields is not None:
        errors["train_mean"] = _errors(mean_fields, actual)
    for row, selected in zip(rows, (np.ones(done.shape, bool), altitude < 10), strict=True):
        row.add(errors, selected, reward.astype(np.float64), true_reward, cont, done)
    return reward


def _stack(states):
    return tuple(torch.cat(parts) for parts in zip(*states, strict=True))


def _score_group(model, episodes, records, spec, mean_fields, posterior_rows, horizon_rows, tensor):
    # Keep identifiers with their episodes while sorting for variable-length posterior batches.
    group = sorted(zip(episodes, records, strict=True), key=lambda pair: len(pair[0]), reverse=True)
    episodes, records = zip(*group)
    lengths = np.asarray([len(ep) for ep in episodes])
    action_offsets = np.cumsum(np.r_[0, lengths[:-1]])
    obs_offsets = action_offsets + np.arange(len(episodes))
    features = np.concatenate([episode_features(ep.obs, spec) for ep in episodes])
    actions = np.concatenate([to_internal_action(ep.action) for ep in episodes])
    rewards = np.concatenate([ep.reward for ep in episodes])
    done = np.concatenate([ep.terminated | ep.truncated for ep in episodes])
    altitude = np.concatenate([ep.obs[:, 2] for ep in episodes])

    posterior = model.observe_initial(tensor(features[obs_offsets]), sample=False)
    tile = 256 // len(episodes)
    for first in range(0, lengths[0], tile):
        states, next_states, episode_ids, times = [], [], [], []
        for t in range(first, min(first + tile, lengths[0])):
            active = int((lengths > t).sum())
            prior_origin = tuple(x[:active] for x in posterior)
            states.append(prior_origin)
            episode_ids.append(np.arange(active))
            times.append(np.full(active, t))
            posterior = model.observe_step(prior_origin, tensor(actions[action_offsets[:active] + t]),
                                           tensor(features[obs_offsets[:active] + t + 1]), sample=False)
            next_states.append(posterior)

        ids, t0 = np.concatenate(episode_ids), np.concatenate(times)
        a0, x0 = action_offsets[ids] + t0, obs_offsets[ids] + t0
        posterior_reward = _record(
            posterior_rows, model.predict(_stack(next_states)), features[x0], features[x0 + 1],
            rewards[a0], done[a0], altitude[x0 + 1], spec, mean_fields)
        for i in np.flatnonzero(done[a0]):
            records[ids[i]]["posterior_reward"] = float(posterior_reward[i])

        remaining = lengths[ids] - t0
        order = np.argsort(-remaining)
        remaining, ids, a0, x0 = (arr[order] for arr in (remaining, ids, a0, x0))
        indices = torch.as_tensor(order, device=posterior[0].device)
        imagined = tuple(x.index_select(0, indices) for x in _stack(states))
        # Only this small contiguous slice contributes to the last-15-step sum.
        suffix = np.flatnonzero(remaining == 15)
        suffix_slice = slice(suffix[0], suffix[-1] + 1) if len(suffix) else slice(0, 0)
        last15 = np.zeros(len(suffix), np.float64)
        for h in range(1, min(max(horizon_rows), remaining[0]) + 1):
            n = int((remaining >= h).sum())
            steps = a0[:n] + h - 1
            imagined = model.imagine_step(tuple(x[:n] for x in imagined), tensor(actions[steps]), sample=False)
            if h in horizon_rows:
                endpoint = x0[:n] + h
                pred_reward = _record(horizon_rows[h], model.predict(imagined), features[x0[:n]],
                                      features[endpoint], rewards[steps], done[steps], altitude[endpoint],
                                      spec, mean_fields)
                for i in np.flatnonzero(done[steps]):
                    records[ids[i]]["prior_rewards"][str(h)] = float(pred_reward[i])
                suffix_reward = pred_reward[suffix_slice]
            elif len(suffix) and 15 in horizon_rows and h <= 15:
                _, reward, _ = model.predict(tuple(x[suffix_slice] for x in imagined))
                suffix_reward = reward.float().cpu().numpy()
            if len(suffix) and 15 in horizon_rows and h <= 15:
                last15 += suffix_reward
                if h == 15:
                    for i, predicted in zip(ids[suffix], last15, strict=True):
                        record = records[i]
                        actual = record["g5_true_sum"]
                        record["g5_predicted_sum"] = float(predicted)
                        record["g5_hit"] = bool(np.sign(predicted) == np.sign(actual)
                                                and abs(predicted) >= 0.5 * abs(actual))


def _gates(posterior, horizons, records):
    def threshold(row, limits):
        values = {key: row.get(key) for key in limits}
        count = row.get("count", 0)
        valid = row.get("tilt_valid_count", 0)
        available = count and valid == count and all(x is not None for x in values.values())
        return {"pass": all(values[k] <= v for k, v in limits.items()) if available else None,
                "count": count, "tilt_valid_count": valid, "values": values, "limits": limits}

    g1 = threshold(posterior, {"position_error_m": .5, "velocity_error_m_s": .2, "tilt_mae_deg": 1.})
    comparisons = {}
    for h in ("5", "15"):
        row = horizons.get(h)
        if row is None or row["count"] == 0:
            continue
        comparisons[h] = {key: {"model": row[key], "persistence": row["baselines"]["persistence"][key]}
                          for key in ("position_error_m", "velocity_error_m_s")}
        comparisons[h]["count"] = row["count"]
    g2 = {"values": comparisons, "pass": None}
    if len(comparisons) == 2:
        g2["pass"] = all(row[k]["model"] <= row[k]["persistence"] for row in comparisons.values()
                         for k in ("position_error_m", "velocity_error_m_s"))
    g3 = threshold(horizons.get("15", {}).get("near_ground", {}),
                   {"velocity_error_m_s": .5, "tilt_mae_deg": 2.})

    groups = {}
    for status, direction in (("safe_landing", 1), ("crash", -1)):
        values = [r["posterior_reward"] for r in records if r["status"] == status]
        hits = sum(direction * value >= 50 for value in values)
        groups[status] = {"hits": hits, "total": len(values), "rate": _mean(hits, len(values))}
    known = sum(r["status"] is not None for r in records)
    g4 = {"groups": groups, "known_status_count": known, "episode_count": len(records),
          "required_rate": .8, "pass": None}
    if known == len(records) and all(group["total"] for group in groups.values()):
        g4["pass"] = all(group["rate"] >= .8 for group in groups.values())

    eligible = sum(r["g5_hit"] is not None for r in records)
    hits = sum(r["g5_hit"] is True for r in records)
    g5 = {"hits": hits, "total": len(records), "evaluated_count": eligible,
          "rate": _mean(hits, len(records)) if eligible == len(records) else None,
          "required_rate": .7, "pass": None}
    if g5["rate"] is not None:
        g5["pass"] = g5["rate"] >= .7
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}


@torch.inference_mode()
def open_loop_report(model, replay: Replay, spec: dict, *, train_mean=None, provenance=None, horizons=HORIZONS):
    """Evaluate every eligible origin at the requested horizons; no filtering across resets."""
    if not replay.episodes:
        raise ValueError("open-loop report requires episodes")
    horizons = tuple(horizons)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain distinct positive integers")
    if provenance is not None and len(provenance) != len(replay.episodes):
        raise ValueError("provenance and replay episode counts differ")
    if train_mean is not None and np.shape(train_mean) != (16,):
        raise ValueError("training mean must be the mean 16-feature vector of the training replay")
    mean_fields = physical_fields(train_mean, spec) if train_mean is not None else None
    records = []
    for i, ep in enumerate(replay.episodes):
        meta = provenance[i] if provenance is not None else {}
        records.append({"episode_index": i, "start_id": meta.get("start_id"), "status": meta.get("status"),
                        "steps": len(ep), "true_reward": float(ep.reward[-1]), "posterior_reward": None,
                        "prior_rewards": {}, "g5_predicted_sum": None, "g5_hit": None,
                        "g5_true_sum": float(ep.reward[-15:].sum(dtype=np.float64)) if len(ep) >= 15 else None})
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), torch.empty(0))
    def tensor(values):
        return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)

    def row_pair():
        return (_Row(mean_fields is not None), _Row(mean_fields is not None))

    posterior_rows = row_pair()
    rows = {h: row_pair() for h in horizons}
    training = model.training
    model.eval()
    try:
        for i in range(0, len(records), 8):
            _score_group(model, replay.episodes[i:i + 8], records[i:i + 8], spec,
                         mean_fields, posterior_rows, rows, tensor)
    finally:
        model.train(training)

    def summary(pair):
        return dict(pair[0].summary(), near_ground=pair[1].summary())

    posterior = summary(posterior_rows)
    tables = {str(h): summary(pair) for h, pair in rows.items()}
    gates = _gates(posterior, tables, records)
    return {
        "episodes": len(records), "horizons": tables, "posterior": posterior,
        "terminal_episodes": records, "gates": gates,
        "acceptance": {"minimum_episodes": 50, "episode_count": len(records),
                       "pass": len(records) >= 50 and all(g["pass"] is True for g in gates.values())},
        "metric_definitions": {
            "origin": "Filter the true history through t; roll with recorded actions t..t+h-1 regardless of predicted termination.",
            "posterior": "Decode after observing t+1. Compare to that frame, with one sample per transition.",
            "baselines": "Persistence uses the true origin observation; train_mean uses a fixed feature vector from the training replay.",
            "errors": "Vectors: mean Euclidean distance. Scalars: mean absolute error. Tilt excludes degenerate rot6d; tilt gates need all endpoints valid.",
            "near_ground": "Select on true endpoint altitude strictly below 10 m.",
            "terminal": "Endpoint terminated OR truncated; predicted endpoint when continuation < 0.5.",
            "G4": "Posterior endpoint reward >= 50 for safe_landing or <= -50 for crash; requires recorded status for all episodes and both groups.",
            "G5": "At L-15, roll with 15 recorded actions and sum the 15 prior rewards. Check sign and at least half the absolute true sum. All episodes required.",
            "missing": "null means unavailable (e.g. no denominator, absent status, training mean or horizon).",
        },
    }
