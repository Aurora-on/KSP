"""S1 diagnostics on complete episodes with recorded actions and fixed gates."""
from collections import Counter

import numpy as np
import torch

from .interface import episode_features, physical_fields, to_internal_action
from .task_reward import rule_outcome, task_reward
from .wm_report import _distribute, _errors, _mean, open_loop_report


HORIZONS = (1, 5, 15, 32)
METHODS = ("deterministic", "persistence", "train_mean")
METRICS = ("position_error_m", "velocity_error_m_s", "tilt_mae_deg")
OUTCOMES = ("continue", "safe_landing", "crash", "timeout", "out_of_bounds", "fuel_exhausted")


class PhysicalTotals:
    """Each vector error is one Euclidean distance per endpoint."""
    def __init__(self):
        self.totals = Counter()
        self.count = 0

    def add(self, errors, selected):
        self.count += int(np.count_nonzero(selected))
        for key in (*METRICS, "tilt_valid_count"):
            self.totals[key] += float(errors[key][selected].sum(dtype=np.float64))

    def summary(self):
        valid = int(self.totals["tilt_valid_count"])
        return {"count": self.count, "tilt_valid_count": valid,
                **{key: _mean(self.totals[key], valid if key == "tilt_mae_deg" else self.count)
                   for key in METRICS}}


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _first_event(probabilities, rules, true_status):
    """One predicted endpoint per real origin; never use the real successor state."""
    predicted = np.where(rules > 0, rules + 2, probabilities.argmax(axis=-1))
    found = np.flatnonzero(predicted)
    step = int(found[0] + 1) if len(found) else None
    status = OUTCOMES[predicted[found[0]]] if len(found) else None
    return {"first_step": step, "first_status": status,
            "hit": status == true_status and abs(step - len(rules)) <= 1}


def _score_episode(model, ep, meta, index, spec, mean_fields, tables, impact, samples, reference):
    length = len(ep)
    features = episode_features(ep.obs, spec)
    states = torch.as_tensor(features, dtype=reference.dtype, device=reference.device)
    actions = torch.as_tensor(to_internal_action(ep.action), dtype=reference.dtype, device=reference.device)
    status = meta["status"]
    if status not in OUTCOMES[1:]:
        raise ValueError(f"unrecognized episode status {status!r}")
    # All origins of a single episode, shortened at each horizon to avoid crossing reset.
    predicted = states[:-1]
    survival = torch.ones_like(predicted[:, 0])
    suffix = length - 15
    predicted_return = torch.zeros_like(survival[0])
    record = {"episode_index": index, "start_id": meta.get("start_id"), "steps": length, "status": status,
              "w5": None}
    for h in range(1, min(length, max(HORIZONS)) + 1):
        n = length - h + 1
        origin = predicted[:n]
        action = actions[h - 1:]
        predicted, probs, head = model.step(origin, action)
        rules = rule_outcome(predicted, spec)
        if h == 1:
            record["w4"] = _first_event(_numpy(probs), _numpy(rules), status)
            risk = torch.where(rules != 0, 1., 1. - probs[:, 0])
            if n > 1:
                samples["nonterminal_probability"].append(_numpy(risk[:-1]))
            head_error = np.abs(_numpy(head).reshape(n).astype(np.float64) - ep.reward)
            samples["reward_error"].append(head_error)
            samples["terminal_reward_error"].append(head_error[-1:])
        if suffix >= 0 and h <= 15:
            sl = slice(suffix, suffix + 1)
            reward, _ = task_reward(origin[sl], action[sl], predicted[sl], probs[sl], spec)
            predicted_return += survival[suffix] * reward.reshape(())
            if h == 15:
                actual = float(ep.reward[suffix:].sum(dtype=np.float64))
                estimate = float(predicted_return.item())
                error = abs(actual - estimate)
                record["w5"] = {"true_return": actual, "predicted_return": estimate,
                                "absolute_error": error, "hit": error <= 25.}
        survival = survival[:n] * torch.where(rules == 0, probs[:, 0], 0.)
        if h == 32 and n > 1:
            # Exclude the last origin, whose real window ends in termination.
            near = ep.obs[:n - 1, 2] < 10.
            survived = _numpy(survival[:-1])
            for label, mask in (("near_ground", near), ("airborne_10m_or_higher", ~near)):
                if mask.any():
                    samples[label].append(survived[mask])
        if h not in HORIZONS:
            continue
        truth = physical_fields(features[h:, :16], spec)
        fields = (physical_fields(_numpy(predicted[:, :16]), spec),
                  physical_fields(features[:n, :16], spec), mean_fields)
        near = ep.obs[h:, 2] < 10.
        all_rows = np.ones(n, bool)
        for method, estimate in zip(METHODS, fields, strict=True):
            errors = _errors(estimate, truth)
            tables[h]["all"][method].add(errors, all_rows)
            tables[h]["near_ground"][method].add(errors, near)
            if h == 1 and method == "deterministic" and status in impact:
                epsilon = spec["ground_epsilon_m"]
                contact = (ep.obs[:-1, 2] >= epsilon) & (ep.obs[1:, 2] < epsilon)
                impact[status].add(errors, contact)
    return record


def _hit_rate(records, key):
    hits = sum(row[key] is not None and row[key]["hit"] for row in records)
    return {"hits": hits, "total": len(records), "rate": _mean(hits, len(records))}


def _physical_gates(horizons):
    w1 = {}
    for h in HORIZONS:
        table = horizons[str(h)]["all"]
        model = table["deterministic"]
        ceilings = {k: table["persistence"][k] * .5 if table["persistence"][k] is not None else None
                    for k in METRICS}
        if h == 15 and model["count"]:
            for key, limit in zip(METRICS, (.5, .2, 1.), strict=True):
                ceilings[key] = min(ceilings[key], limit)
        w1[str(h)] = {"limits": ceilings, "pass": _under_limits(model, ceilings)}
    near = horizons["15"]["near_ground"]["deterministic"]
    limits = {"velocity_error_m_s": .2, "tilt_mae_deg": 1.}
    return ({"horizons": w1, "pass": all(x["pass"] for x in w1.values())},
            {"values": near, "limits": limits, "pass": _under_limits(near, limits)})


def _under_limits(row, limits):
    return bool(row["count"] and row["tilt_valid_count"] == row["count"]
                and all(row[k] is not None and limit is not None and row[k] <= limit
                        for k, limit in limits.items()))


@torch.inference_mode()
def deterministic_report(model, replay, spec, *, provenance, train_mean, rssm=None, rssm_spec=None):
    """Score a deterministic model. Supply the selected RSSM for the four-method comparison."""
    episodes = replay.episodes
    if not episodes or len(episodes) != len(provenance):
        raise ValueError("S1 reporting requires complete episodes with outcome provenance")
    if np.shape(train_mean) != (16,):
        raise ValueError("train_mean must contain the 16 physical means from training frames")
    if (rssm is None) != (rssm_spec is None):
        raise ValueError("RSSM and its checkpoint interface must be supplied together")
    mean_fields = physical_fields(train_mean, spec)
    totals = {h: {group: {name: PhysicalTotals() for name in METHODS}
                  for group in ("all", "near_ground")} for h in HORIZONS}
    impact = {status: PhysicalTotals() for status in ("safe_landing", "crash")}
    samples = {key: [] for key in ("nonterminal_probability", "reward_error", "terminal_reward_error",
                                    "near_ground", "airborne_10m_or_higher")}
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), torch.empty(0))
    was_training = model.training
    model.eval()
    try:
        records = [_score_episode(model, ep, meta, i, spec, mean_fields, totals, impact, samples, reference)
                   for i, (ep, meta) in enumerate(zip(episodes, provenance, strict=True))]
    finally:
        model.train(was_training)
    horizons = {str(h): {group: {name: value.summary() for name, value in methods.items()}
                         for group, methods in groups.items()} for h, groups in totals.items()}
    if rssm is not None:
        comparison = open_loop_report(rssm, replay, rssm_spec, provenance=provenance, horizons=HORIZONS)
        for h, row in comparison["horizons"].items():
            for group, source in (("all", row), ("near_ground", row["near_ground"])):
                target = horizons[h][group]
                if target["deterministic"]["count"] != source["count"]:
                    raise AssertionError("RSSM and deterministic models must score identical endpoints")
                target["rssm"] = {key: source[key] for key in ("count", "tilt_valid_count", *METRICS)}
    w1, w2 = _physical_gates(horizons)
    groups = {s: _hit_rate([r for r in records if r["status"] == s], "w4") for s in sorted(
        set(impact).union(r["status"] for r in records))}
    overall = _hit_rate(records, "w4")
    nonterminal = _distribute(samples["nonterminal_probability"])
    w4 = {"overall": overall, "by_status": groups, "nonterminal_end_probability": nonterminal,
          "pass": (overall["rate"] >= .95 and all(groups[s]["total"] and groups[s]["rate"] >= .90
                   for s in impact) and nonterminal["mean"] is not None and nonterminal["mean"] <= .01)}
    w5 = _hit_rate(records, "w5")
    w5["eligible"] = sum(r["w5"] is not None for r in records)
    w5["pass"] = w5["eligible"] == len(records) and w5["rate"] >= .85
    gates = {"W1": w1, "W2": w2, "W3": {"by_final_status": {s: v.summary() for s, v in impact.items()}},
             "W4": w4, "W5": w5, "W6": {"all_steps": _distribute(samples["reward_error"]),
                                         "terminal_steps": _distribute(samples["terminal_reward_error"])}}
    return {"episodes": len(records), "horizons": horizons, "gates": gates,
            "acceptance": {"minimum_episodes": 50, "episode_count": len(records),
                           "pass": len(records) >= 50 and all(gates[k]["pass"] for k in ("W1","W2","W4","W5"))},
            "terminal_episodes": records, "survival_32_nonterminal_windows":
                {key: _distribute(samples[key]) for key in ("near_ground", "airborne_10m_or_higher")},
            "metric_definitions": {
                "comparison": "All eligible origins. Deterministic and persistence use the real origin. RSSM filters the full real history through that origin before rolling out recorded actions. Train mean is fixed. Score physical errors even after a predicted terminal event.",
                "near_ground": "W2 uses the true endpoint altitude < 10 m.",
                "W3": "One-step error on each real crossing from height >= epsilon to < epsilon; group by final episode status.",
                "W4": "From each real origin, predict one successor without observing it. Rule abort overrides corrected argmax. First event only, one-based transition index; a missing event is a miss. Mean nonterminal risk includes every real nonterminal step, also after early model alarms.",
                "W5": "From L-15, sum 15 nominal-dt model rewards using the incoming survival product, gamma=1, with recorded actions. Compare to the undiscounted sum of real rewards.",
                "W6": "Absolute error of the diagnostic reward head on every real origin with a predicted successor; mean is MAE.",
                "survival_32_nonterminal_windows": "Product of predicted continuation probabilities with rule aborts, on real windows with all 32 transitions nonterminal. Group by true origin altitude.",
            }}
