"""Reproducible held-out starts, engineering baselines, and parameter sweeps."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import platform
import time
from typing import Callable

import numpy as np

from .config import SimConfig
from .controller import ScriptedController
from .env import LanderEnv
from .state import sample_initial_state

SPLIT_IDS = {"train": 0, "adaptation": 1, "validation": 2, "test": 3, "generalization": 4}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def generate_initial_states(config: SimConfig, count: int = 30, seed: int = 2026,
                            split: str = "test") -> dict:
    """Generate immutable common starts using non-overlapping integer seed sets.

    Within a root seed, each (split,index) maps to a unique integer reset seed.
    Keep the resulting manifest with experiment artifacts; do not regenerate
    starts using each method's training seed.
    """
    if split not in SPLIT_IDS:
        raise ValueError(f"split must be one of {tuple(SPLIT_IDS)}")
    if not isinstance(count, int) or not 1 <= count < 1_000_000:
        raise ValueError("count must be an integer in [1, 1000000)")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    starts = []
    for index in range(count):
        episode_seed = seed * 10_000_000 + SPLIT_IDS[split] * 1_000_000 + index
        state = sample_initial_state(np.random.default_rng(episode_seed), config).to_dict()
        starts.append({"id": f"{split}-{seed}-{index:06d}", "seed": episode_seed,
                       "initial_state": state})
    result = {"schema_version": 1, "split": split, "seed": seed,
              "initial_config": asdict(config.initial), "fuel_capacity": config.vehicle.fuel_capacity,
              "starts": starts}
    result["sha256"] = hashlib.sha256(json.dumps(_jsonable(result), sort_keys=True).encode()).hexdigest()
    return _jsonable(result)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or not manifest.get("starts"):
        raise ValueError("Expected nonempty initial-state manifest schema_version=1")
    ids = [entry["id"] for entry in manifest["starts"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate start IDs")
    if "sha256" in manifest:
        payload = {k: v for k, v in manifest.items() if k != "sha256"}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if actual != manifest["sha256"]:
            raise ValueError("Initial-state manifest checksum mismatch")


def rollout(config: SimConfig, initial_state: dict | None = None, seed: int = 7,
            controller: str | Callable = "scripted", controller_config: SimConfig | None = None,
            record: bool = False) -> tuple[dict, dict | None]:
    """Run one episode. A supplied policy sees only the observation array."""
    env = LanderEnv(config)
    options = {"initial_state": initial_state} if initial_state is not None else None
    obs, info = env.reset(seed=seed, options=options)
    if isinstance(controller, str):
        if controller == "scripted":
            policy = ScriptedController(controller_config or config)
        elif controller == "zero":
            policy = lambda _: np.zeros(4, dtype=np.float32)
        elif controller == "random":
            rng = np.random.default_rng(seed)
            policy = lambda _: np.array([rng.uniform(0, 1), *rng.uniform(-1, 1, 3)], dtype=np.float32)
        else:
            raise ValueError("Unknown controller: expected scripted, zero or random")
    else:
        policy = controller
    if hasattr(policy, "reset"):
        policy.reset()
    observations = [obs.copy()] if record else None
    times, rewards, actions = [0.0], [], []
    terminated_flags, truncated_flags, durations = [], [], []
    main_fuel, rcs_fuel = [0.0], [0.0]
    total_return = 0.0
    transitions = 0
    start = time.perf_counter()
    while True:
        action = policy(obs.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        transitions += 1
        total_return += reward
        if record:
            observations.append(obs.copy())
            actions.append(np.asarray(info.get("applied_action", action)).copy())
            times.append(info["time"])
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            durations.append(info["dt"])
            main_fuel.append(info["main_fuel_used"])
            rcs_fuel.append(info["rcs_fuel_used"])
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - start
    result = {key: info[key] for key in ["status", "is_success", "safe_in_target", "time",
              "horizontal_error", "main_fuel_used", "rcs_fuel_used", "total_fuel_used"]}
    result.update({"touchdown": info.get("touchdown"), "last_touchdown": info.get("last_touchdown"),
                   "touchdown_count": info.get("touchdown_count", 0), "stable_time": info["stable_time"]})
    result.update({"return": float(total_return), "transitions": transitions,
                   "terminated": bool(terminated), "truncated": bool(truncated),
                   "wall_time_seconds": elapsed,
                   "controller": controller if isinstance(controller, str) else "callable"})
    env.close()
    trajectory = None
    if record:
        trajectory = {"observations": np.asarray(observations), "actions": np.asarray(actions),
                      "times": np.asarray(times), "rewards": np.asarray(rewards),
                      "terminated": np.asarray(terminated_flags, dtype=bool),
                      "truncated": np.asarray(truncated_flags, dtype=bool),
                      "dt": np.asarray(durations),
                      "main_fuel_used": np.asarray(main_fuel), "rcs_fuel_used": np.asarray(rcs_fuel)}
    return _jsonable(result), trajectory


def evaluate(config: SimConfig, manifest: dict, controller: str | Callable = "scripted",
             controller_config: SimConfig | None = None) -> dict:
    validate_manifest(manifest)
    episodes = []
    start = time.perf_counter()
    for entry in manifest["starts"]:
        result, _ = rollout(config, entry["initial_state"], int(entry["seed"]),
                            controller, controller_config)
        result["start_id"] = entry["id"]
        episodes.append(result)
    safe = [e for e in episodes if e["is_success"]]
    count = len(episodes)
    summary = {"schema_version": 1, "controller": controller if isinstance(controller, str) else "callable",
               "config": config.to_dict(), "controller_config": (controller_config or config).to_dict(),
               "split": manifest.get("split"), "starts_sha256": manifest.get("sha256"),
               "n_episodes": count, "safe_landing_rate": len(safe) / count,
               "crash_rate": sum(e["status"] == "crash" for e in episodes) / count,
               "safe_in_target_rate": sum(e["safe_in_target"] for e in episodes) / count,
               "status_counts": dict(Counter(e["status"] for e in episodes)),
               "transitions": sum(e["transitions"] for e in episodes),
               "wall_time_seconds": time.perf_counter() - start,
               "episodes": episodes}
    for field in ("horizontal_error", "main_fuel_used", "rcs_fuel_used", "total_fuel_used"):
        summary[f"mean_{field}_safe"] = float(np.mean([e[field] for e in safe])) if safe else None
    # Wilson 95% interval states sampling uncertainty without hiding trial count.
    p, n, z = summary["safe_landing_rate"], count, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    summary["safe_landing_rate_wilson95"] = [float(max(0, center-half)), float(min(1, center+half))]
    return _jsonable(summary)


def parameter_sweep(config: SimConfig, manifest: dict,
                    scales=(0.8, 1.0, 1.2), controller: str = "scripted") -> dict:
    """Change one physical parameter at a time; keep starts/controller fixed."""
    validate_manifest(manifest)
    scales = tuple(float(s) for s in scales)
    if not scales or not all(np.isfinite(s) and s > 0 for s in scales):
        raise ValueError("scales must be nonempty positive finite values")
    # Nominal always present even if callers request only held-out magnitudes.
    scales = tuple(sorted(set(scales) | {1.0}))
    nominal = evaluate(config, manifest, controller, controller_config=config)
    runs = []
    for parameter in ("mass", "inertia", "thrust", "gravity"):
        for scale in scales:
            evaluation = nominal if scale == 1.0 else evaluate(
                config.perturbed(parameter, scale), manifest, controller, controller_config=config)
            runs.append({"parameter": parameter, "scale": scale, "evaluation": evaluation})
    return {"schema_version": 1, "controller": controller, "fixed_nominal_controller": True,
            "starts_sha256": manifest.get("sha256"), "scales": list(scales),
            "unique_evaluation_runs": 1 + 4 * (len(scales)-1),
            "unique_episodes": len(manifest["starts"]) * (1 + 4 * (len(scales)-1)),
            "nominal_reused_across_parameters": True, "runs": runs}


def benchmark(config: SimConfig, steps: int = 2000, seed: int = 7) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    env = LanderEnv(config)
    policy = ScriptedController(config)
    obs, _ = env.reset(seed=seed)
    # Warm up before timing; rendering and file IO excluded.
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(policy(obs))
        if term or trunc:
            obs, _ = env.reset()
    start = time.perf_counter()
    simulated_time = 0.0
    resets = 0
    for _ in range(steps):
        obs, _, term, trunc, info = env.step(policy(obs))
        simulated_time += info["dt"]
        if term or trunc:
            resets += 1
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return {"steps": steps, "wall_time_seconds": elapsed, "transitions_per_second": steps/elapsed,
            "simulated_seconds": simulated_time, "real_time_factor": simulated_time/elapsed,
            "resets": resets, "includes_controller": True, "includes_rendering": False,
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__, "config": config.to_dict()}
