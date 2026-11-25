"""Agent-side vertical diagnostic starts, actions and paired closed-loop evaluations."""

import hashlib
import json
from pathlib import Path

import numpy as np
from mun_lander import SimConfig
from mun_lander.evaluation import evaluate, generate_initial_states, validate_manifest

from .collect import collect
from .policy_rollouts import AppliedPolicy, collect_from_manifest, compare_returns, policy_trajectories, write_manifest_once


def vertical_manifest(*, split, seed, episodes):
    """Project the nominal starts, preserving sampled yaw, altitude, vertical speed and fuel."""
    manifest = generate_initial_states(SimConfig(), episodes, seed, split)
    manifest["source_manifest_sha256"] = manifest.pop("sha256")
    manifest["diagnostic"] = "vertical"
    for entry in manifest["starts"]:
        state = entry["initial_state"]
        state["position"][:2] = [0., 0.]
        state["velocity"][:2] = [0., 0.]
        state["angular_velocity"] = [0., 0., 0.]
        # The source quaternion is tilt * yaw; zeroing x,y extracts its yaw factor.
        quat = np.array(state["quaternion"])
        quat[1:3] = 0.
        state["quaternion"] = (quat / np.linalg.norm(quat)).tolist()
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    validate_manifest(manifest)
    return manifest


def load_validation(path):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    if manifest.get("diagnostic") != "vertical" or manifest["split"] != "validation":
        raise ValueError("expected a vertical validation manifest")
    for entry in manifest["starts"]:
        state = entry["initial_state"]
        if any((*state["position"][:2], *state["velocity"][:2],
                *state["angular_velocity"], *state["quaternion"][1:3])):
            raise ValueError("manifest contains a nonvertical initial state")
    return manifest


def prepare_validation(path, *, episodes=100, seed=2026):
    """Create once or load the saved vertical projection without regenerating its starts."""
    write_manifest_once(path, lambda: vertical_manifest(split="validation", seed=seed, episodes=episodes))
    manifest = load_validation(path)
    if len(manifest["starts"]) != episodes or manifest["seed"] != seed:
        raise ValueError("existing diagnostic manifest has a different seed or size")
    return manifest


def vertical_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., 1:] = 0.
    return action


def collect_prefill(*, seed=230, episodes=200):
    manifest = vertical_manifest(split="train", seed=seed, episodes=episodes)
    return collect(seed=seed, episodes=episodes, noise_levels=(0., .05, .1, .2), hard_fraction=.6,
                   random_fraction=.05, manifest=manifest, action_transform=vertical_action)


def collect_policy(policy, *, seed, episodes, noise_std=.1):
    manifest = vertical_manifest(split="train", seed=seed, episodes=episodes)
    return collect_from_manifest(policy, manifest, noise_std=noise_std, action_transform=vertical_action)


def evaluate_vertical(policy, manifest):
    if manifest.get("diagnostic") != "vertical" or manifest["split"] != "validation":
        raise ValueError("expected a vertical validation manifest")
    return evaluate(SimConfig(), manifest, AppliedPolicy(policy, action_transform=vertical_action))


def paired_returns(model, actor, policy, manifest, *, horizon=32):
    validate_manifest(manifest)
    if manifest.get("diagnostic") != "vertical" or manifest["split"] != "validation":
        raise ValueError("expected a vertical validation manifest")
    wrapped = AppliedPolicy(policy, action_transform=vertical_action)
    return compare_returns(model, actor, manifest, policy_trajectories(wrapped, manifest),
                           horizon=horizon, throttle_only=True)
