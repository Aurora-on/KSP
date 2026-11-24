"""Collect complete nominal simulator episodes for training replay."""

import math
from numbers import Real

import numpy as np
from mun_lander import LanderEnv, SimConfig
from mun_lander.controller import ScriptedController
from mun_lander.evaluation import generate_initial_states, validate_manifest

from .replay import Episode, Replay


def _collect_episode(env, start, controller, noise_std):
    # A separate RNG stream keeps action noise independent of reset-state draws.
    rng = np.random.default_rng([start["seed"], 1])
    obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
    observations = [obs.copy()]
    actions, rewards, terminations, truncations = [], [], [], []
    low, high = env.action_space.low, env.action_space.high
    controller.reset()
    while True:
        if noise_std is None:
            action = rng.uniform(low, high)
        else:
            action = controller(obs)
            if noise_std > 0:
                action = np.clip(action + rng.normal(0, noise_std, size=action.shape), low, high)

        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        observations.append(obs.copy())
        actions.append(info["applied_action"].copy())
        rewards.append(reward)
        terminations.append(terminated)
        truncations.append(truncated)
        if terminated or truncated:
            break

    return Episode(obs=np.asarray(observations, dtype=np.float32),
                   action=np.asarray(actions, dtype=np.float32),
                   reward=np.asarray(rewards, dtype=np.float32),
                   terminated=np.asarray(terminations, dtype=bool),
                   truncated=np.asarray(truncations, dtype=bool))


def collect(*, seed: int, episodes: int, noise_levels: tuple[float, ...],
            random_fraction: float) -> tuple[Replay, dict]:
    """Mix uniform random actions with scripted actions at cycling noise levels.

    Noise standard deviations are in native action units. The episode count
    for the random policy is rounded to the nearest integer (ties round up).
    The shuffled assignment and all actions are reproducible from the seed.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(episodes) is not int or not 1 <= episodes < 1_000_000:
        raise ValueError("episodes must be an integer in [1, 1000000)")
    if (isinstance(random_fraction, bool) or not isinstance(random_fraction, Real)
            or not 0 <= random_fraction <= 1):
        raise ValueError("random_fraction must be a finite fraction in [0, 1]")
    if not noise_levels or any(isinstance(std, bool) or not isinstance(std, Real)
                               or not math.isfinite(std) or std < 0 for std in noise_levels):
        raise ValueError("noise_levels must contain finite nonnegative standard deviations")

    config = SimConfig()
    manifest = generate_initial_states(config, episodes, seed, split="train")
    validate_manifest(manifest)
    random_count = math.floor(episodes * random_fraction + 0.5)
    levels = [float(noise_levels[i % len(noise_levels)]) for i in range(episodes - random_count)]
    schedule = levels + [None] * random_count
    np.random.default_rng([seed, 2]).shuffle(schedule)

    env = LanderEnv(config)
    controller = ScriptedController(config)
    collected, provenance = [], []
    try:
        for start, noise_std in zip(manifest["starts"], schedule, strict=True):
            episode = _collect_episode(env, start, controller, noise_std)
            collected.append(episode)
            provenance.append({
                "start_id": start["id"], "reset_seed": start["seed"],
                "controller": "random" if noise_std is None else "scripted",
                "noise_std": noise_std, "steps": len(episode.action),
            })
    finally:
        env.close()
    return Replay(collected), {
        "split": manifest["split"], "seed": seed, "episodes": episodes,
        "noise_levels": list(map(float, noise_levels)), "random_fraction": float(random_fraction),
        "manifest": manifest, "train_manifest_sha256": manifest["sha256"],
        "reset_seeds": [start["seed"] for start in manifest["starts"]],
        "env_steps": sum(record["steps"] for record in provenance),
        "episode_provenance": provenance,
    }
