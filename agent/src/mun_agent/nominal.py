"""Fixed nominal 6DOF simulator starts, full-action collection and evaluation."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mun_lander import SimConfig
from mun_lander.evaluation import SPLIT_IDS, evaluate, generate_initial_states, validate_manifest

from .policy_rollouts import collect_from_manifest, compare_returns, policy_trajectories, write_manifest_once


def _check_manifest(manifest, split):
    validate_manifest(manifest)
    if split not in ("validation", "test") or manifest["split"] != split or "diagnostic" in manifest:
        raise ValueError("expected a nominal validation or test manifest with the requested split")
    config = SimConfig()
    # The simulator serializes dataclass tuples as lists.
    if (manifest["initial_config"] != json.loads(json.dumps(asdict(config.initial)))
            or manifest["fuel_capacity"] != config.vehicle.fuel_capacity):
        raise ValueError("manifest initial distribution differs from nominal SimConfig")
    for index, start in enumerate(manifest["starts"]):
        seed = manifest["seed"] * 10_000_000 + SPLIT_IDS[split] * 1_000_000 + index
        if start["seed"] != seed or start["id"] != f"{split}-{manifest['seed']}-{index:06d}":
            raise ValueError("unexpected reset seed or start ID in nominal manifest")


def load_manifest(path, split):
    manifest = json.loads(Path(path).read_text())
    _check_manifest(manifest, split)
    return manifest


def prepare_manifest(path, split, *, seed=2026, episodes=100):
    """Publish once. The saved manifest's checksum travels with it across platforms."""
    if split not in ("test", "validation"):
        raise ValueError("an evaluation manifest needs the validation or test split")
    write_manifest_once(path, lambda: generate_initial_states(SimConfig(), episodes, seed, split))
    manifest = load_manifest(path, split)
    if manifest["seed"] != seed or len(manifest["starts"]) != episodes:
        raise ValueError("existing nominal manifest has a different seed or size")
    return manifest


def collect_policy(policy, *, seed, episodes, noise_std=.1):
    manifest = generate_initial_states(SimConfig(), episodes, seed, "train")
    return collect_from_manifest(policy, manifest, noise_std=noise_std)


def _touchdown_summary(episodes):
    # First impact exists also for crashes. Report the denominator explicitly.
    contacts = [ep["touchdown"] for ep in episodes if ep["touchdown"] is not None]
    result = {"touchdown_count": len(contacts)}
    for output, key in (("speed_m_s", "speed"), ("tilt_deg", "tilt_deg"),
                        ("angular_speed_rad_s", "angular_speed")):
        values = [c[key] for c in contacts]
        result["mean_touchdown_" + output] = float(np.mean(values)) if values else None
        result["max_touchdown_" + output] = max(values) if values else None
    errors = [c["horizontal_error"] for c in contacts]
    result["mean_landing_error_m"] = float(np.mean(errors)) if errors else None
    return result


def evaluate_nominal(policy, manifest):
    _check_manifest(manifest, manifest["split"])
    result = evaluate(SimConfig(), manifest, policy)
    episodes = result["episodes"]
    result.update(_touchdown_summary(episodes))
    result.update(mean_return=float(np.mean([ep["return"] for ep in episodes])),
                  mean_flight_time_s=float(np.mean([ep["time"] for ep in episodes])),
                  mean_total_fuel_used_kg=float(np.mean([ep["total_fuel_used"] for ep in episodes])))
    return result


def _physical_errors(model, replay, rows):
    """All eligible true origins at each horizon, scored using logged actions."""
    import torch
    from .deterministic_fit import EpisodeWindows
    from .interface import physical_fields
    from .wm_report import _errors

    if len(rows) != len(replay.episodes) or not rows:
        raise ValueError("physical risk requires a complete episode and outcome for every origin")
    device = next(model.parameters()).device
    pool = EpisodeWindows(replay, {"episode_provenance": rows}, model.spec, device)
    horizons = (1, 5, 15, 32)
    reports = {}
    with torch.inference_mode():
        for h in horizons:
            starts = np.concatenate([offset + np.arange(max(0, n - h + 1))
                                     for offset, n in zip(pool.offsets, pool.lengths, strict=True)])
            totals = {"all": {}, "near_ground": {}}
            counts = {group: 0 for group in totals}
            valid_tilts = counts.copy()
            for indices in torch.as_tensor(starts, dtype=torch.long, device=device).split(2048):
                if not len(indices):
                    continue
                state = pool.current[indices]
                for t in range(h):
                    state = model.advance(state, pool.actions[indices + t])
                target = pool.following[indices + h - 1]
                predicted = physical_fields(state[:, :16].cpu().numpy(), model.spec)
                truth = physical_fields(target[:, :16].cpu().numpy(), model.spec)
                errors = _errors(predicted, truth)
                valid = errors.pop("tilt_valid_count")
                errors["horizontal_position_error_m"] = np.linalg.norm(
                    predicted["position"][:, :2] - truth["position"][:, :2], axis=1)
                for group, mask in (("all", np.ones(len(indices), bool)),
                                    ("near_ground", truth["position"][:, 2] < 10.)):
                    counts[group] += int(mask.sum())
                    valid_tilts[group] += int(valid[mask].sum())
                    for name, values in errors.items():
                        totals[group][name] = totals[group].get(name, 0.) + float(values[mask].sum(dtype=np.float64))
            groups = {}
            for group, sums in totals.items():
                means = {}
                for key, value in sums.items():
                    denom = valid_tilts[group] if key == "tilt_mae_deg" else counts[group]
                    means[key] = value / denom if denom else None
                groups[group] = dict(count=counts[group], tilt_valid_count=valid_tilts[group], **means)
            reports[str(h)] = groups
    return reports


def _visited_replay(trajectories, manifest):
    from .replay import Episode, Replay
    episodes, rows = [], []
    for start, (outcome, data) in zip(manifest["starts"], trajectories, strict=True):
        episodes.append(Episode(data["observations"], data["actions"], data["rewards"],
                                data["terminated"], data["truncated"], dt=data["dt"]))
        rows.append(dict(start_id=start["id"], reset_seed=start["seed"], **outcome))
    return Replay(episodes), rows


def _visited_summary(model, trajectories, manifest):
    from collections import Counter
    from .wm_report import _distribute
    visited, rows = _visited_replay(trajectories, manifest)
    status_counts = dict(Counter(row["status"] for row in rows))
    touch = {}
    for group in ("all", *sorted(status_counts)):
        touches = [row["touchdown"] for row in rows if row["touchdown"] and (group == "all" or row["status"] == group)]
        touch[group] = {}
        for name, key in (("speed_m_s", "speed"), ("tilt_deg", "tilt_deg"),
                          ("angular_speed_rad_s", "angular_speed"), ("horizontal_error_m", "horizontal_error")):
            touch[group][name] = _distribute([np.array([c[key] for c in touches])] if touches else [])
    return {"episodes": len(rows), "transitions": sum(map(len, visited.episodes)), "status_counts": status_counts,
            "touchdown_distributions": touch, "open_loop": _physical_errors(model, visited, rows),
            "first_touchdowns": [{"start_id": row["start_id"], "status": row["status"], "touchdown": row["touchdown"]}
                                 for row in rows]}


def policy_risk(model, policy, manifest):
    """Counterpart for a single checkpoint before or after a validation decline."""
    _check_manifest(manifest, "validation")
    summary = _visited_summary(model, policy_trajectories(policy, manifest), manifest)
    return dict(summary, manifest_sha256=manifest["sha256"])


def visited_pairs(model, actor, policy, manifest, *, horizon=32):
    """Use the same real episodes for physical errors and independently controlled return pairs."""
    _check_manifest(manifest, "validation")
    trajectories = list(policy_trajectories(policy, manifest))
    return {
        "manifest_sha256": manifest["sha256"],
        "definition": "Independent closed-loop policy returns on the same real origins. Physical errors roll forward the actual logged actions from every eligible origin, grouped by source and true endpoint altitude below 10 m. Touchdown distributions use the first contact of each episode that contacts the surface.",
        "paired_returns": compare_returns(model, actor, manifest, trajectories, horizon=horizon),
        "policy_visited": _visited_summary(model, trajectories, manifest),
    }


def risk_report(model, actor, policy, manifest, prefill, prefill_rows, *, horizon=32):
    """Measure the validation-selected policy and separately score its model on permanent prefill."""
    result = visited_pairs(model, actor, policy, manifest, horizon=horizon)
    result["permanent_prefill"] = {"episodes": len(prefill.episodes), "transitions": sum(map(len, prefill.episodes)),
                                  "open_loop": _physical_errors(model, prefill, prefill_rows)}
    return result
