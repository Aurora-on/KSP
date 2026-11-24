"""Agent-side vertical diagnostic starts, actions and paired closed-loop evaluations."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from mun_lander import LanderEnv, SimConfig
from mun_lander.evaluation import evaluate, generate_initial_states, validate_manifest

from .collect import _collect_episode, collect
from .interface import to_internal_action, to_native_action
from .replay import Replay


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
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        projected = vertical_manifest(split="validation", seed=seed, episodes=episodes)
        # Publish the entire manifest atomically without replacing a concurrent winner.
        with NamedTemporaryFile(mode="w", dir=path.parent) as temporary:
            json.dump(projected, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            try:
                os.link(temporary.name, path)
            except FileExistsError:
                pass
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


class _VerticalPolicy:
    def __init__(self, policy, noise_std=0., rng=None):
        self.policy = policy
        self.noise_std = noise_std
        self.rng = rng

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def __call__(self, obs):
        action = self.policy(obs)
        if self.noise_std:
            internal = to_internal_action(action) + self.rng.normal(0., self.noise_std, size=4)
            action = to_native_action(np.clip(internal, -1., 1.))
        return vertical_action(action)


def collect_policy(policy, *, seed, episodes, noise_std=.1):
    """Collect current-policy transitions using independent internal-coordinate action noise."""
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    manifest = vertical_manifest(split="train", seed=seed, episodes=episodes)
    env = LanderEnv(SimConfig())
    collected, rows = [], []
    try:
        for start in manifest["starts"]:
            noise_seed = [start["seed"], 6]
            wrapped = _VerticalPolicy(policy, noise_std, np.random.default_rng(noise_seed))
            ep, outcome = _collect_episode(env, start, wrapped, 0., None)
            collected.append(ep)
            rows.append(dict(start_id=start["id"], reset_seed=start["seed"], steps=len(ep), controller="policy",
                             action_rng_seed=noise_seed, noise_std=noise_std, **outcome))
    finally:
        env.close()
    return Replay(collected), {
        "split": "train", "seed": seed, "episodes": episodes, "manifest": manifest,
        "train_manifest_sha256": manifest["sha256"], "reset_seeds": [s["seed"] for s in manifest["starts"]],
        "status_counts": dict(Counter(row["status"] for row in rows)), "episode_provenance": rows,
        "env_steps": sum(map(len, collected)),
    }


def evaluate_vertical(policy, manifest):
    if manifest.get("diagnostic") != "vertical" or manifest["split"] != "validation":
        raise ValueError("expected a vertical validation manifest")
    return evaluate(SimConfig(), manifest, _VerticalPolicy(policy))


def _return_summary(records):
    if not records:
        return {"count": 0, "real_mean": None, "imagined_mean": None, "mean_gap": None, "mae": None}
    gaps = np.array([r["gap"] for r in records])
    return {"count": len(records), "real_mean": float(np.mean([r["real_return"] for r in records])),
            "imagined_mean": float(np.mean([r["imagined_return"] for r in records])),
            "mean_gap": float(gaps.mean()), "mae": float(np.mean(np.abs(gaps)))}


def paired_returns(model, actor, policy, manifest, *, horizon=32):
    """Pair simulator and expected model returns while independently re-evaluating the same actor."""
    import torch
    from mun_lander.evaluation import rollout

    from .interface import episode_features
    from .task_reward import task_reward

    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    validate_manifest(manifest)
    if manifest.get("diagnostic") != "vertical" or manifest["split"] != "validation":
        raise ValueError("expected a vertical validation manifest")
    wrapped = _VerticalPolicy(policy)
    # Keep one row per (episode, phase), allowing distinct phases to share a start
    # in a short episode. The terminal window includes the true terminal reward.
    starts, real_first_actions, records, episodes = [], [], [], []
    config = SimConfig()
    for entry in manifest["starts"]:
        outcome, traj = rollout(config, entry["initial_state"], entry["seed"], wrapped, record=True)
        length = outcome["transitions"]
        features = episode_features(traj["observations"], model.spec)
        last_full = max(0, length - horizon)
        origins = {"reset": 0, "middle": min(length // 2, last_full), "terminal": last_full}
        height = traj["observations"][:, 2]
        contact = np.flatnonzero((height[:-1] >= model.spec["ground_epsilon_m"])
                                 & (height[1:] < model.spec["ground_epsilon_m"]))
        if contact.size:
            origins["first_contact"] = min(last_full, max(0, int(contact[0]) - horizon // 2))
        touchdown = outcome["touchdown"]
        episodes.append({"start_id": entry["id"], "status": outcome["status"],
                         "touchdown_speed": touchdown["speed"] if touchdown else None})
        for phase, origin in origins.items():
            stop = min(length, origin + horizon)
            starts.append(features[origin])
            real_first_actions.append(to_internal_action(traj["actions"][origin]))
            records.append({"start_id": entry["id"], "reset_seed": entry["seed"], "phase": phase,
                            "origin_step": origin, "height_m": float(height[origin]),
                            "real_steps": stop - origin, "real_terminal": stop == length,
                            "episode_status": outcome["status"], "real_return": float(traj["rewards"][origin:stop].sum())})

    with torch.inference_mode():
        ref = next(model.parameters())
        state = torch.as_tensor(np.stack(starts), dtype=ref.dtype, device=ref.device)
        actor_ref = next(actor.parameters())
        survival = torch.ones_like(state[:, 0])
        total = torch.zeros_like(survival)
        for step in range(horizon):
            # Both branches re-evaluate the actor on their own state at each step.
            action = actor(state.to(actor_ref)).to(state)
            action = torch.cat((action[..., :1], torch.zeros_like(action[..., 1:])), dim=-1)
            if step == 0 and not np.allclose(action.cpu().numpy(), np.stack(real_first_actions), atol=1e-5, rtol=1e-5):
                raise ValueError("simulator policy and imagined actor disagree at an identical origin")
            following, probs, _ = model.step(state, action)
            reward, continuation = task_reward(state, action, following, probs, model.spec)
            total += survival * reward
            survival *= continuation
            state = following
        for row, value, cont in zip(records, total.cpu().tolist(), survival.cpu().tolist(), strict=True):
            row.update(imagined_return=value, gap=value - row["real_return"], imagined_end_survival=cont)

    statuses = Counter(ep["status"] for ep in episodes)
    return {
        "horizon": horizon, "manifest_sha256": manifest["sha256"], "episodes": len(episodes),
        "definition": "Undiscounted rewards at paired real starts. Each branch runs the actor on its own observations or predicted states. Imagined rewards use incoming survival. Real rewards after termination are zero.",
        "status_counts": dict(statuses),
        "failed_status_counts": {k: v for k, v in statuses.items() if k != "safe_landing"},
        "summary": _return_summary(records),
        "by_phase": {phase: _return_summary([r for r in records if r["phase"] == phase])
                     for phase in sorted({r["phase"] for r in records})},
        "by_episode_status": {status: _return_summary([r for r in records if r["episode_status"] == status])
                              for status in sorted(statuses)},
        "episode_outcomes": episodes, "windows": records,
    }
