"""Fit eight-step transition sequences and select a checkpoint on disjoint development episodes."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from mun_lander import SimConfig

from .dataset import load_dataset
from .deterministic import DeterministicModel
from .fit import code_version
from .interface import deterministic_spec, episode_features, physical_fields, to_internal_action


class EpisodeWindows:
    """Vectorized episode indexing with equally covered transition and terminal targets."""
    def __init__(self, replay, record, spec, device):
        self.lengths = np.array([len(ep) for ep in replay.episodes])
        self.offsets = np.r_[0, self.lengths.cumsum()[:-1]]
        frames = [episode_features(ep.obs, spec) for ep in replay.episodes]
        self.mean = np.concatenate(frames)[:, :16].mean(0, dtype=np.float64).tolist()
        self.status = [p["status"] for p in record["episode_provenance"]]
        labels = []
        for ep, status in zip(replay.episodes, self.status, strict=True):
            y = np.zeros(len(ep), dtype=np.int64)
            if status in ("safe_landing", "crash"):
                y[-1] = 1 if status == "safe_landing" else 2
            elif status in ("timeout", "out_of_bounds", "fuel_exhausted"):
                y[-1] = -1
            else:
                raise ValueError(f"unknown episode outcome: {status}")
            labels.append(y)
        def cat(arrays):
            return torch.as_tensor(np.concatenate(arrays), device=device)
        self.current = cat([f[:-1] for f in frames])
        self.following = cat([f[1:] for f in frames])
        self.actions = cat([to_internal_action(ep.action) for ep in replay.episodes])
        self.labels = cat(labels)
        self.rewards = cat([ep.reward for ep in replay.episodes])

    def sample(self, rng, count, length):
        # Negative starts are left padded so the first transition is covered
        # just as often as an interior (or final) transition.
        limits = np.cumsum(self.lengths + length - 1)
        draws = rng.integers(limits[-1], size=count)
        ids = np.searchsorted(limits, draws, side="right")
        starts = draws - np.r_[0, limits[:-1]][ids] - length + 1
        steps = starts[:, None] + np.arange(length)
        addresses = self.offsets[ids, None] + np.clip(steps, 0, self.lengths[ids, None] - 1)
        mask = (steps >= 0) & (steps < self.lengths[ids, None])
        index = torch.as_tensor(addresses, device=self.current.device)
        return (self.current[index[:, 0]], self.actions[index], self.following[index],
                self.labels[index], self.rewards[index], torch.as_tensor(mask, device=index.device))


@torch.inference_mode()
def dev_score(model, data):
    """Fixed dev score: 15-step tracking, first-event mistakes and terminal-window return."""
    from .task_reward import task_reward
    spec = model.spec
    device = data.current.device
    rng = np.random.default_rng(0)
    horizon = 15
    starts = np.concatenate([offset + np.arange(length - horizon + 1)
                             for offset, length in zip(data.offsets, data.lengths) if length >= horizon])
    if not len(starts):
        raise ValueError("development set has no complete 15-step window")
    probe = rng.choice(starts, min(len(starts), 2048), replace=False)
    eligible = data.lengths >= horizon
    tails = data.offsets[eligible] + data.lengths[eligible] - horizon
    index = torch.as_tensor(np.r_[probe, tails], device=device)
    state = data.current[index]
    expected = torch.zeros(len(index), device=device)
    survival = torch.ones_like(expected)

    for t in range(horizon):
        action = data.actions[index + t]
        following, probs, _ = model.step(state, action)
        reward, continuation = task_reward(state, action, following, probs, spec)
        expected += reward * survival
        survival *= continuation
        state = following
    n = len(probe)
    actual = data.following[index[:n] + horizon - 1]
    error = (state[:n, :6] - actual[:, :6]).cpu().numpy()
    pos = float(np.linalg.norm(error[:, :3] * spec["scales"]["position"], axis=-1).mean())
    vel = float(np.linalg.norm(error[:, 3:] * spec["scales"]["velocity"], axis=-1).mean())
    predicted_tilt = physical_fields(state[:n, :16].cpu().numpy(), spec)
    true_tilt = physical_fields(actual[:, :16].cpu().numpy(), spec)
    tilt = float(np.abs(predicted_tilt["tilt_deg"] - true_tilt["tilt_deg"]).mean())
    tail_true = sum(data.rewards[index[n:] + t] for t in range(horizon))
    tail_error = (expected[n:] - tail_true).abs()
    tail_pass = (tail_error <= 25).float().mean().item()

    # Score learned events here. Rule-based aborts are evaluated in the full report.
    probs = torch.cat([model.step(state, action)[1] for state, action in
                       zip(data.current.split(2048), data.actions.split(2048), strict=True)])
    decisions = probs.argmax(-1).cpu().numpy()
    event_errors = []
    for name, target in (("safe_landing", 1), ("crash", 2)):
        agreements = []
        for offset, length, status in zip(data.offsets, data.lengths, data.status):
            if status != name:
                continue
            segment = decisions[offset:offset + length]
            times = np.flatnonzero(segment != 0)
            agreements.append(bool(len(times) and times[0] >= length - 2 and segment[times[0]] == target))
        if agreements:
            event_errors.append(1 - float(np.mean(agreements)))
    tracking = (pos / 0.5 + vel / 0.2 + tilt) / 3
    score = tracking + sum(event_errors) + 2 * (1 - tail_pass)
    return dict(score=score, position_m=pos, speed_m_s=vel, tilt_deg=tilt,
                class_error=event_errors, terminal_window_pass=tail_pass,
                terminal_window_mae=tail_error.mean().item())


def fit_deterministic(train_path, dev_path, out, config, seed=31, device="cpu"):
    start = time.monotonic()
    if config.get("model_type") != "deterministic" or config["length"] != 8:
        raise ValueError("expected deterministic configuration with k=8")
    if not 0 < config["max_seconds"] <= 1800:
        raise ValueError("fit budget must be in (0, 1800] seconds")
    for key in ("batch_size", "updates", "width", "checkpoint_every", "cpu_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "event_weight"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    spec = deterministic_spec(SimConfig())
    train_replay, train_record = load_dataset(train_path)
    dev_replay, dev_record = load_dataset(dev_path)
    if set(train_record["reset_seeds"]) & set(dev_record["reset_seeds"]):
        raise ValueError("training and development reset seeds overlap")
    train = EpisodeWindows(train_replay, train_record, spec, device)
    dev = EpisodeWindows(dev_replay, dev_record, spec, device)
    model = DeterministicModel(spec, width=config["width"], event_weight=config["event_weight"],
                               kinematic=config.get("kinematic", False),
                               contact_margin=config.get("contact_margin", False)).to(device)
    model.delta_std.copy_((train.following[:, :16] - train.current[:, :16]).std(0, unbiased=False).clamp_min(1e-6))
    dynamics_params = [*model.dynamics.parameters(), *model.event.parameters()]
    groups = [{"params": [*dynamics_params, *model.diagnostic_reward.parameters()]}]
    if model.kinematic:
        model.initialize_liftoff(train.current, train.actions, train.following)
        groups.append({"params": model.liftoff.parameters(), "lr": config["learning_rate"] * 0.001})
        dynamics_params.extend(model.liftoff.parameters())
    optimizer = torch.optim.Adam(groups, lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["updates"],
                                                          eta_min=0.0 if model.kinematic else config["learning_rate"] * 0.1)
    provenance = dict(model_type="deterministic", schema_version=1, config=config, seed=seed,
                      device=str(device), code=code_version(), env_steps=train_record["env_steps"])
    for role, record in (("train", train_record), ("dev", dev_record)):
        provenance.update({role + "_reset_seeds": record["reset_seeds"],
                           role + "_manifest_sha256": record["train_manifest_sha256"],
                           role + "_replay_sha256": record["replay_sha256"],
                           role + "_collection_seed": record["seed"]})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best_score = float("inf")
    best_update = None
    with (out / "metrics.jsonl").open("x") as log:
        for update in range(1, config["updates"] + 1):
            batch = train.sample(rng, config["batch_size"], config["length"])
            losses = model.losses(*batch)
            optimizer.zero_grad(set_to_none=True)
            sum(losses).backward()
            torch.nn.utils.clip_grad_norm_(dynamics_params,
                                           1.0, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(model.diagnostic_reward.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            elapsed = time.monotonic() - start
            done = elapsed >= config["max_seconds"] or update == config["updates"]
            validate = done or update % config["checkpoint_every"] == 0
            if update == 1 or update % 100 == 0 or validate:
                row = dict(update=update, elapsed_s=elapsed, updates_per_second=update / elapsed,
                           dynamics_loss=losses[0].item(), event_loss=losses[1].item(),
                           reward_loss=losses[2].item())
                if validate:
                    row["dev"] = dev_score(model, dev)
                    if row["dev"]["score"] < best_score:
                        best_update = update
                        best_score = row["dev"]["score"]
                        snapshot = dict(provenance, interface_spec=spec, model=model.state_dict(),
                                        updates=update, train_mean=train.mean, dev=row["dev"])
                        tmp = out / "best.tmp"
                        torch.save(snapshot, tmp)
                        tmp.replace(out / "best.pt")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
            if done:
                break
    summary = dict(updates=update, elapsed_s=time.monotonic()-start, best_update=best_update,
                   best_dev_score=best_score, checkpoint="best.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
