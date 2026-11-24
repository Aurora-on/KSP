"""Collect complete nominal simulator episodes for training replay."""

import math
from collections import Counter
from numbers import Real

import numpy as np
from mun_lander import LanderEnv, SimConfig
from mun_lander.controller import ScriptedController
from mun_lander.evaluation import generate_initial_states, validate_manifest

from .replay import Episode, Replay


def _hard_parameters(seed):
    rng = np.random.default_rng([seed, 3])
    # A uniform draw in sqrt(height) spreads speeds across a ballistic descent.
    height = float(rng.uniform(0.52, 3.55) ** 2)
    cap = float(rng.uniform(0.0, 0.09))
    azimuth = float(rng.uniform(-np.pi, np.pi))
    amplitude = float(rng.uniform(0.0, 1.0))
    return {
        "rng_seed": [seed, 3], "cut_height_m": height, "throttle_cap": cap,
        "rcs_x": amplitude * math.cos(azimuth), "rcs_y": amplitude * math.sin(azimuth),
        "pulse_lead_s": float(rng.uniform(0.90, 1.45)),
    }


def _apply_hard_landing(action, obs, vehicle, perturbation):
    height = float(obs[2])
    if 0 < height <= perturbation["cut_height_m"]:
        cap = perturbation["throttle_cap"]
        action[0] = min(action[0], cap)
        # Approximate remaining descent under capped thrust. A short roll/pitch
        # pulse varies the impact attitude independently of the commanded cutoff.
        lead = perturbation["pulse_lead_s"]
        accel = vehicle.gravity - cap * vehicle.max_thrust / (vehicle.dry_mass + vehicle.fuel_capacity)
        if height <= max(-float(obs[5]), 0.0) * lead + 0.5 * accel * lead**2:
            action[1:3] = perturbation["rcs_x"], perturbation["rcs_y"]


def _collect_episode(env, start, controller, noise_std, hard_parameters, action_transform=None):
    # A separate RNG stream keeps action noise independent of reset-state draws.
    rng = np.random.default_rng([start["seed"], 1])
    obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
    observations = [obs.copy()]
    durations = []
    actions, rewards, terminations, truncations = [], [], [], []
    low, high = env.action_space.low, env.action_space.high
    controller.reset()
    while True:
        if noise_std is None:
            action = rng.uniform(low, high)
        else:
            action = controller(obs)
            if hard_parameters is not None:
                _apply_hard_landing(action, obs, controller.config.vehicle, hard_parameters)
            elif noise_std > 0:
                action = np.clip(action + rng.normal(0, noise_std, size=action.shape), low, high)

        if action_transform is not None:
            action = action_transform(action)
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        observations.append(obs.copy())
        actions.append(info["applied_action"].copy())
        rewards.append(reward)
        durations.append(info["dt"])
        terminations.append(terminated)
        truncations.append(truncated)
        if terminated or truncated:
            break

    episode = Episode(
        obs=np.asarray(observations, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminations, dtype=bool),
        truncated=np.asarray(truncations, dtype=bool),
        dt=np.asarray(durations, dtype=np.float32),
    )
    # Diagnostics stay in the collection record, outside the training replay.
    touchdown = info.get("touchdown")
    outcome = {
        "status": info["status"],
        "touchdown_speed": touchdown["speed"] if touchdown is not None else None,
        "touchdown_tilt_deg": touchdown["tilt_deg"] if touchdown is not None else None,
    }
    return episode, outcome


def collect(*, seed: int, episodes: int, noise_levels: tuple[float, ...],
            random_fraction: float, hard_fraction: float = 0.0, manifest=None,
            action_transform=None) -> tuple[Replay, dict]:
    """Mix uniform random actions with scripted actions at cycling noise levels.

    The hard fraction uses capped near-ground throttle and roll/pitch pulses.
    Fractional counts round to nearest (ties up), allocating random episodes
    first. Each remaining scripted episode uses a cycling native noise level.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(episodes) is not int or not 1 <= episodes < 1_000_000:
        raise ValueError("episodes must be an integer in [1, 1000000)")
    for name, fraction in (("random_fraction", random_fraction), ("hard_fraction", hard_fraction)):
        if isinstance(fraction, bool) or not isinstance(fraction, Real) or not 0 <= fraction <= 1:
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    if hard_fraction + random_fraction > 1:
        raise ValueError("hard_fraction and random_fraction must sum to at most 1")
    if not noise_levels or any(isinstance(std, bool) or not isinstance(std, Real)
                               or not math.isfinite(std) or std < 0 for std in noise_levels):
        raise ValueError("noise_levels must contain finite nonnegative standard deviations")

    config = SimConfig()
    if manifest is None:
        manifest = generate_initial_states(config, episodes, seed, split="train")
    validate_manifest(manifest)
    if manifest["split"] != "train" or manifest["seed"] != seed or len(manifest["starts"]) != episodes:
        raise ValueError("collection manifest does not match train split, seed and size")
    random_count = math.floor(episodes * random_fraction + 0.5)
    hard_count = min(episodes - random_count, math.floor(episodes * hard_fraction + 0.5))
    schedule = [("scripted", float(noise_levels[i % len(noise_levels)]))
                for i in range(episodes - random_count - hard_count)]
    schedule += [("random", None)] * random_count + [("hard_landing", 0.0)] * hard_count
    np.random.default_rng([seed, 2]).shuffle(schedule)

    env = LanderEnv(config)
    controller = ScriptedController(config)
    collected, provenance = [], []
    try:
        for start, (kind, noise_std) in zip(manifest["starts"], schedule, strict=True):
            hard = _hard_parameters(start["seed"]) if kind == "hard_landing" else None
            episode, outcome = _collect_episode(env, start, controller, noise_std, hard, action_transform)
            collected.append(episode)
            provenance.append({
                "start_id": start["id"], "reset_seed": start["seed"],
                "controller": kind, "noise_std": noise_std, "hard_parameters": hard,
                "action_rng_seed": [start["seed"], 1], "steps": len(episode.action), **outcome,
            })
    finally:
        env.close()
    return Replay(collected), {
        "split": manifest["split"], "seed": seed, "episodes": episodes,
        "noise_levels": list(map(float, noise_levels)), "random_fraction": float(random_fraction),
        "hard_fraction": float(hard_fraction), "status_counts": dict(Counter(r["status"] for r in provenance)),
        "manifest": manifest, "train_manifest_sha256": manifest["sha256"],
        "reset_seeds": [start["seed"] for start in manifest["starts"]],
        "env_steps": sum(record["steps"] for record in provenance),
        "episode_provenance": provenance,
    }
