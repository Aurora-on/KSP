"""Diagnose the world model and critic on the two real replay sources of a policy snapshot."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import load_dataset
from .deterministic_fit import EpisodeWindows
from .deterministic_report import PhysicalTotals
from .imagination import Critic
from .interface import physical_fields
from .replay import Episode, Replay
from .task_reward import task_reward
from .train import load_training_world_model
from .wm_report import _distribute, _errors


def _distribution(values):
    return _distribute([np.asarray(values, dtype=float)] if len(values) else [])


def _endpoints(model, pool, horizon):
    totals = PhysicalTotals()
    starts = np.concatenate([offset + np.arange(max(0, n - horizon + 1))
                             for offset, n in zip(pool.offsets, pool.lengths)])
    indices = torch.as_tensor(starts, device=pool.current.device, dtype=torch.long)
    for chunk in indices.split(2048):
        if not len(chunk):
            continue
        state = pool.current[chunk]
        for step in range(horizon):
            state = model.advance(state, pool.actions[chunk + step])
        truth = pool.following[chunk + horizon - 1]
        actual = physical_fields(truth[:, :16].cpu().numpy(), model.spec)
        predicted = physical_fields(state[:, :16].cpu().numpy(), model.spec)
        totals.add(_errors(predicted, actual), np.ones(len(chunk), dtype=bool))
    return totals.summary()


def _tail_returns(model, pool, horizon):
    eligible = pool.lengths >= horizon
    starts = pool.offsets[eligible] + pool.lengths[eligible] - horizon
    if not len(starts):
        return {"episodes": 0}
    origin = torch.as_tensor(starts, device=pool.current.device)
    state = pool.current[origin]
    predicted = torch.zeros_like(state[:, 0])
    actual = torch.zeros_like(predicted)
    survival = torch.ones_like(predicted)
    for step in range(horizon):
        action = pool.actions[origin + step]
        following, probs, _ = model.step(state, action)
        reward, continuation = task_reward(state, action, following, probs, model.spec)
        predicted += survival * reward
        survival *= continuation
        actual += pool.rewards[origin + step]
        state = following
    gap = (predicted - actual).cpu().numpy()
    statuses = np.asarray(pool.status)[eligible]
    return {"episodes": len(gap), "signed_gap": _distribution(gap),
            "absolute_gap": _distribution(np.abs(gap)),
            "signed_gap_by_status": {s: _distribution(gap[statuses == s]) for s in sorted(set(statuses))}}


def _outcomes(critic, replay, pool, rows, gamma):
    values = torch.cat([critic(chunk) for chunk in pool.current.split(2048)]).cpu().numpy()
    by_status = defaultdict(list)
    for ep, row, offset in zip(replay.episodes, rows, pool.offsets, strict=True):
        heights = ep.obs[:-1, 2]
        locations = {"reset": 0, "terminal_origin": len(ep) - 1}
        near = np.flatnonzero(heights < 10.)
        if len(near):
            locations["first_below_10m"] = int(near[0])
        phases = {}
        for name, t in locations.items():
            observed = np.dot(gamma**np.arange(len(ep) - t), ep.reward[t:])
            phases[name] = [values[offset + t], observed]
        by_status[row["status"]].append((row.get("touchdown_speed"), phases))
    result = {}
    for status in sorted(set(by_status) | {"safe_landing", "crash"}):
        episodes = by_status[status]
        phases = {}
        for phase in ("reset", "first_below_10m", "terminal_origin"):
            pairs = [p[phase] for _, p in episodes if phase in p]
            phases[phase] = {"critic_value": _distribution([p[0] for p in pairs]),
                             "observed_discounted_return": _distribution([p[1] for p in pairs])}
        result[status] = {"episodes": len(episodes), "values_by_phase": phases,
                          "first_touchdown_speed_m_s": _distribution([s for s, _ in episodes if s is not None])}
    return result


@torch.inference_mode()
def source_diagnostics(model, critic, replay, rows, gamma):
    if len(replay.episodes) != len(rows):
        raise ValueError("diagnostics require one outcome per episode")
    result = {"episodes": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
              "transitions": sum(map(len, replay.episodes))}
    if not rows:
        return result
    pool = EpisodeWindows(replay, {"episode_provenance": rows}, model.spec, next(model.parameters()).device)
    result.update(open_loop={str(h): _endpoints(model, pool, h) for h in (1, 32)},
                  terminal_window_32=_tail_returns(model, pool, 32),
                  by_outcome=_outcomes(critic, replay, pool, rows, gamma))
    return result


def diagnose(checkpoint, data, device="cpu"):
    # latest.pt can be replaced by the trainer during this call. Hash and read
    # a single open file description so the report always identifies its weights.
    with Path(checkpoint).open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    prefill, metadata = load_dataset(data)
    source = saved["provenance"]
    if source["prefill_replay_sha256"] != metadata["replay_sha256"]:
        raise ValueError("prefill differs from checkpoint training provenance")
    config = source["config"]
    model = load_training_world_model(saved, device)
    critic = Critic(config["critic_width"], config["value_scale"], config.get("critic_upper_bound"),
                    config.get("yaw_invariant", False)).to(device)
    critic.load_state_dict(saved["critic"])
    critic.eval()
    fifo = Replay(Episode(**{k: v.numpy() for k, v in ep.items()}) for ep in saved["fifo"])
    return {"iteration": saved["progress"]["iteration"], "checkpoint_sha256": sha,
            "seed": source["seed"], "prefill_replay_sha256": metadata["replay_sha256"],
            "definition": "All eligible real origins, recorded-action open loops. 32-step return gap is expected minus real undiscounted reward from the last 32 steps. Critic values use original reward units. Observed discounted returns follow the logged behavior. Each phase contributes at most once per episode.",
            "prefill": source_diagnostics(model, critic, prefill, metadata["episode_provenance"], config["gamma"]),
            "policy_fifo": source_diagnostics(model, critic, fifo, saved["fifo_provenance"], config["gamma"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="checkpoint's permanent prefill")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = diagnose(args.checkpoint, args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, allow_nan=False, indent=2) + "\n"
    with args.out.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"iteration": result["iteration"], "checkpoint_sha256": result["checkpoint_sha256"],
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
