"""Differentiable simulator reward and causal history updates for imagination."""

import numpy as np
import torch
from mun_lander.physics import G0

from .interface import episode_features, to_internal_action


def advance_history(state, next_physical, spec):
    """Supply rule-updated, gradient-free counters alongside predicted dynamics."""
    ground_cap = spec["ground_history_cap"]
    step_cap = spec["episode_step_cap"]
    with torch.no_grad():
        grounded = next_physical[..., 2] * spec["scales"]["position"] < spec["ground_epsilon_m"]
        ground_steps = torch.round(state[..., 16] * ground_cap) + 1
        ground = grounded * ground_steps.clamp(max=ground_cap) / ground_cap
        steps = torch.round(state[..., 17] * step_cap) + 1
        clock = steps.clamp(max=step_cap) / step_cap
    return torch.cat((next_physical, ground.unsqueeze(-1), clock.unsqueeze(-1)), dim=-1)


def rule_outcome(next_state, spec):
    """Return 0 for no override, 1 for timeout, 2 for bounds, 3 for airborne exhaustion."""
    config = spec["source_config"]
    position = next_state[..., :3] * spec["scales"]["position"]
    fuel = next_state[..., 15] * spec["scales"]["fuel"]
    outside = (torch.linalg.vector_norm(position[..., :2], dim=-1) > config["max_horizontal_distance"]
               ) | (position[..., 2] > config["max_altitude"])
    empty_in_flight = (fuel <= 1e-10) & (position[..., 2] >= spec["ground_epsilon_m"])
    code = (next_state[..., 17] >= 1).long()
    code = torch.where(empty_in_flight, 3, code)
    # The simulator tests bounds before fuel exhaustion and timeout.
    return torch.where(outside, 2, code)


def process_reward(state, internal_action, next_state, spec, *, dt=None):
    """Evaluate the six process terms in SI units at the end of each transition."""
    config = spec["source_config"]
    vehicle, cost, scales = config["vehicle"], config["reward"], spec["scales"]
    dt = spec["control_dt"] if dt is None else dt
    distance = torch.linalg.vector_norm(next_state[..., :2], dim=-1) * scales["position"]
    speed = torch.linalg.vector_norm(next_state[..., 3:6], dim=-1) * scales["velocity"]
    # The normalized cross product is the thrust axis. atan2 also works before
    # normalization, with defined gradient at upright attitude.
    up = torch.linalg.cross(next_state[..., 6:9], next_state[..., 9:12], dim=-1)
    horizontal = torch.linalg.vector_norm(up[..., :2], dim=-1)
    up_z = torch.where((up == 0).all(dim=-1), 1.0, up[..., 2])
    tilt = torch.atan2(horizontal, up_z)
    main_used = (state[..., 15] - next_state[..., 15]).clamp_min(0.0) * scales["fuel"]
    torque = internal_action[..., 1:] * internal_action.new_tensor(vehicle["rcs_torque"])
    rcs_rate = torque.abs().sum(dim=-1) / (vehicle["rcs_lever_arm"] * vehicle["rcs_isp"] * G0)
    rate = (cost["time_cost"] + cost["horizontal_cost"] * distance + cost["speed_cost"] * speed
            + cost["tilt_cost"] * tilt + cost["rcs_fuel_cost"] * rcs_rate)
    return -rate * dt - cost["main_fuel_cost"] * main_used


def task_reward(state, internal_action, next_state, probs, spec, *, dt=None):
    """Return immediate expected reward and conditional survival for one step."""
    cost = spec["source_config"]["reward"]
    process = process_reward(state, internal_action, next_state, spec, dt=dt)
    abort = rule_outcome(next_state, spec) != 0
    terminal = probs[..., 1] * cost["safe_bonus"] - probs[..., 2] * cost["crash_cost"]
    reward = process + torch.where(abort, -cost["abort_cost"], terminal)
    return reward, torch.where(abort, 0.0, probs[..., 0])


def _distribution(values):
    errors = np.abs(np.concatenate(values))
    if not errors.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    if not errors.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": int(errors.size), "mean": float(errors.mean()), "p50": float(np.median(errors)),
            "p95": float(np.quantile(errors, 0.95)), "p99": float(np.quantile(errors, 0.99)),
            "max": float(errors.max())}


@torch.no_grad()
def reward_parity(replay, metadata, spec):
    """Measure process errors on real transitions, including shortened terminal steps.

    Terminal probabilities here are one-hot true labels. Subtracting the true
    terminal award from the environment reward isolates its process component.
    """
    if not replay.has_dt:
        raise ValueError("reward parity requires recorded per-transition dt")
    cost = spec["source_config"]["reward"]
    events = {"safe_landing": (1, cost["safe_bonus"]), "crash": (2, -cost["crash_cost"])}
    aborts = {"timeout", "out_of_bounds", "fuel_exhausted"}
    nonterminal, terminal, terminal_total = [], [], []
    for ep, provenance in zip(replay.episodes, metadata["episode_provenance"], strict=True):
        status = provenance["status"]
        if status in aborts:
            label, award = 0, -cost["abort_cost"]
        else:
            label, award = events[status]
        states = torch.from_numpy(episode_features(ep.obs, spec)).double()
        actions = torch.from_numpy(to_internal_action(ep.action)).double()
        dt = torch.tensor(ep.dt, dtype=torch.float64)
        predicted = process_reward(states[:-1], actions, states[1:], spec, dt=dt).numpy()
        nonterminal.append(predicted[:-1] - ep.reward[:-1])
        terminal.append(predicted[-1:] - (ep.reward[-1:].astype(np.float64) - award))
        probs = torch.zeros(3, dtype=states.dtype)
        probs[label] = 1.0
        total, _ = task_reward(states[-2], actions[-1], states[-1], probs, spec, dt=dt[-1])
        terminal_total.append(np.array([total.item() - ep.reward[-1].item()]))
    return {"nonterminal_process_abs_error": _distribution(nonterminal),
            "terminal_process_abs_error": _distribution(terminal),
            "terminal_total_abs_error": _distribution(terminal_total)}
