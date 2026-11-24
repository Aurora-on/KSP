"""Action-conditioned open-loop prediction from deterministic replay posteriors."""

from collections import Counter

import numpy as np
import torch

from .interface import physical_fields, preprocess, to_internal_action
from .replay import Replay


HORIZONS = (1, 5, 15, 50)
_VECTOR_ERRORS = {
    "position": "position_error_m",
    "velocity": "velocity_error_m_s",
    "angular_velocity": "angular_velocity_error_rad_s",
}


def _accumulate(total, prediction, target_features, target_reward, ended, spec):
    features, reward, cont = (x.detach().float().cpu().numpy() for x in prediction)
    if reward.shape != target_reward.shape or cont.shape != ended.shape:
        raise ValueError("reward and continuation heads must return one scalar per endpoint")
    if not np.isfinite(reward).all() or not np.all((cont >= 0) & (cont <= 1)):
        raise ValueError("world model returned invalid reward or continuation")

    estimated = physical_fields(features, spec)
    actual = physical_fields(target_features, spec)
    total["count"] += len(ended)
    for field, key in _VECTOR_ERRORS.items():
        error = np.linalg.norm(estimated[field].astype(np.float64) - actual[field], axis=-1)
        if not np.isfinite(error).all():
            raise ValueError(f"nonfinite {field} prediction in physical units")
        total[key] += float(error.sum(dtype=np.float64))
    tilt_valid = estimated["tilt_valid"] & actual["tilt_valid"]
    total["tilt_valid_count"] += int(tilt_valid.sum())
    tilt_error = np.abs(estimated["tilt_deg"] - actual["tilt_deg"])
    total["tilt_mae_deg"] += float(tilt_error[tilt_valid].sum(dtype=np.float64))
    fuel_error = np.abs(estimated["fuel"].astype(np.float64) - actual["fuel"])
    if not np.isfinite(fuel_error).all():
        raise ValueError("nonfinite fuel prediction in kilograms")
    total["fuel_mae_kg"] += float(fuel_error.sum())
    total["reward_mae"] += float(np.abs(reward.astype(np.float64) - target_reward).sum())

    predicted_end = cont < 0.5
    for key, match in (("tp", predicted_end & ended), ("fp", predicted_end & ~ended),
                       ("tn", ~predicted_end & ~ended), ("fn", ~predicted_end & ended)):
        total[key] += int(match.sum())


def _summary(total):
    count = total["count"]
    valid = total["tilt_valid_count"]
    means = {key: total[key] / count if count else None
             for key in (*_VECTOR_ERRORS.values(), "fuel_mae_kg", "reward_mae")}
    tp, fp, tn, fn = (total[key] for key in ("tp", "fp", "tn", "fn"))
    return {
        "count": count, **means, "tilt_valid_count": valid, "tilt_invalid_count": count - valid,
        "tilt_mae_deg": total["tilt_mae_deg"] / valid if valid else None,
        "terminal": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "false_positive_rate": fp / (fp + tn) if fp + tn else None},
    }


def _score_group(model, episodes, spec, totals, tensor):
    # Sort so all still-active episodes form a prefix of the posterior batch.
    episodes = sorted(episodes, key=len, reverse=True)
    lengths = np.asarray([len(ep) for ep in episodes])
    action_offsets = np.cumsum(np.r_[0, lengths[:-1]])
    obs_offsets = action_offsets + np.arange(len(episodes))  # one extra frame per episode
    features = np.concatenate([preprocess(ep.obs, spec) for ep in episodes])
    actions = np.concatenate([to_internal_action(ep.action) for ep in episodes])
    rewards = np.concatenate([ep.reward for ep in episodes])
    ended = np.concatenate([ep.terminated | ep.truncated for ep in episodes])

    posterior = model.observe_initial(tensor(features[obs_offsets]), sample=False)
    # At most 256 origins occupy latent-state memory, regardless of replay size.
    tile = 256 // len(episodes)
    for first in range(0, lengths[0], tile):
        states, action_starts, obs_starts, remaining_steps = [], [], [], []
        for t in range(first, min(first + tile, lengths[0])):
            active = int((lengths > t).sum())
            states.append(posterior)
            action_starts.append(action_offsets[:active] + t)
            obs_starts.append(obs_offsets[:active] + t)
            remaining_steps.append(lengths[:active] - t)
            following = int((lengths > t + 1).sum())
            if following:
                posterior = model.observe_step(
                    tuple(x[:following] for x in posterior),
                    tensor(actions[action_offsets[:following] + t]),
                    tensor(features[obs_offsets[:following] + t + 1]), sample=False)

        # Longest remaining windows first. At horizon h, only the prefix
        # with at least h real transitions enters the prior and the denominator.
        remaining = np.concatenate(remaining_steps)
        order = np.argsort(-remaining)
        remaining = remaining[order]
        a0 = np.concatenate(action_starts)[order]
        x0 = np.concatenate(obs_starts)[order]
        indices = torch.as_tensor(order, device=states[0][0].device)
        imagined = tuple(torch.cat(parts).index_select(0, indices) for parts in zip(*states))
        for h in range(1, min(HORIZONS[-1], remaining[0]) + 1):
            n = int((remaining >= h).sum())
            steps = a0[:n] + h - 1
            imagined = model.imagine_step(tuple(x[:n] for x in imagined),
                                          tensor(actions[steps]), sample=False)
            if h in totals:
                _accumulate(totals[h], model.predict(imagined), features[x0[:n] + h],
                            rewards[steps], ended[steps], spec)


@torch.inference_mode()
def open_loop_report(model, replay: Replay, spec: dict) -> dict:
    """Evaluate every eligible origin in supplied held-out episodes at each horizon."""
    if not replay.episodes:
        raise ValueError("open-loop report requires held-out episodes")
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), torch.empty(0))

    def tensor(values):
        return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)

    totals = {h: Counter() for h in HORIZONS}
    training = model.training
    model.eval()
    try:
        for i in range(0, len(replay.episodes), 8):
            _score_group(model, replay.episodes[i:i + 8], spec, totals, tensor)
    finally:
        model.train(training)

    return {
        "episodes": len(replay.episodes),
        "horizons": {str(h): _summary(totals[h]) for h in HORIZONS},
        "metric_definitions": {
            "origin": "Posterior through observation t. Predict observation t+h using recorded actions t through t+h-1.",
            "errors": "Mean Euclidean error for vector fields. Mean absolute error for fuel, tilt and endpoint reward.",
            "tilt": "Only nondegenerate decoded rotations contribute; tilt_valid_count is the denominator.",
            "terminal": "Endpoint terminated OR truncated against predicted continuation < 0.5. Rollouts use all recorded actions regardless of predicted termination.",
            "missing": "null means zero eligible denominator.",
        },
    }
