"""Online model fitting and short-horizon policy training for the vertical diagnostic."""
import json
import os
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .dataset import file_hash, load_dataset
from .deterministic import DeterministicModel, load_deterministic
from .deterministic_fit import EpisodeWindows
from .fit import code_version
from .imagination import Actor, Critic, update_actor_critic
from .policy import Policy, policy_checkpoint
from .replay import Episode, Replay
from .vertical import collect_policy, evaluate_vertical, load_validation


def _write_json(path, value, *, lines=False):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as output:
        if lines:
            for row in value:
                output.write(json.dumps(row, allow_nan=False) + "\n")
        else:
            output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _serialize_episode(episode):
    return {key: torch.tensor(value) for key, value in vars(episode).items() if value is not None}


def _check_config(c):
    if c["task"] != "vertical" or not 0 < c["max_seconds"] <= 7200:
        raise ValueError("S2 needs the vertical task and a budget of at most two hours")
    for name in ("iterations", "cpu_threads", "collect_episodes", "fifo_episodes", "wm_batch", "wm_updates",
                 "imagined_batch", "actor_updates", "actor_width", "critic_width", "critic_epochs",
                 "horizon", "validate_every"):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("actor_lr", "critic_lr", "wm_lr", "value_scale", "near_altitude_m"):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 <= c["noise_std"] or not math.isfinite(c["noise_std"]):
        raise ValueError("invalid exploration noise")
    if not 0 < c["prefill_fraction"] < 1 or min(c["wm_batch"], c["imagined_batch"]) < 2:
        raise ValueError("both replay sources need positive representation")
    for k in ("gamma", "lambda", "near_fraction"):
        if not 0 <= c[k] <= 1:
            raise ValueError(f"{k} must be a probability")


def _counts(size, fraction):
    first = min(size - 1, max(1, round(size * fraction)))
    return first, size - first


def _sample_starts(pool, strata, rng, count, fraction):
    # Uniform origins and reset distributions are sampled independently within each replay source.
    n_near = round(count * fraction) if len(strata[0]) else 0
    parts = [rng.integers(len(pool.current), size=count - n_near)]
    # Optionally dedicate a third of the near-origin quota to approaching contact
    # and a third to stable support close enough to pay the bonus within H.
    remaining = n_near
    for indices in strata[1:]:
        if len(indices):
            quota = n_near // 3
            parts.append(rng.choice(indices, quota))
            remaining -= quota
    parts.append(rng.choice(strata[0], remaining))
    selected = np.concatenate(parts)
    return pool.current[torch.as_tensor(selected, device=pool.current.device)]


def _near(pool, spec, config):
    frames = pool.current.detach().cpu().numpy()
    height = frames[:, 2] * spec["scales"]["position"]
    near = height < config["near_altitude_m"]
    groups = [np.flatnonzero(near)]
    if config.get("stratify_contact", False):
        horizon = config["horizon"]
        vz = frames[:, 5] * spec["scales"]["velocity"]
        approaching = (frames[:, 16] == 0) & (vz < 0) & (height <= -vz * horizon * spec["control_dt"])
        stable = frames[:, 16] > max(0., 1. - horizon / spec["ground_history_cap"])
        groups.extend((np.flatnonzero(near & approaching), np.flatnonzero(near & stable)))
    return groups


def load_training_world_model(checkpoint, device="cpu"):
    saved = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("training_schema") != 1:
        raise ValueError("expected an online training checkpoint")
    model = DeterministicModel(saved["interface_spec"], **saved["wm_arch"])
    model.load_state_dict(saved["world_model"])
    return model.to(device).eval()


class BudgetExpired(Exception):
    pass


class _TimedPolicy:
    """Check the wall clock even inside long simulator evaluations."""
    def __init__(self, policy, check):
        self.policy = policy
        self.check = check

    def reset(self):
        self.policy.reset()

    def __call__(self, obs):
        self.check()
        return self.policy(obs)


def _device_rng(device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _recover_rng(saved, rng, device):
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["device_rng"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(saved["device_rng"])


def _publish_files(out, progress, improved):
    # latest.pt is the single commit point. Replay publication after interruption
    # repairs either log or a missing new best without a duplicate validation point.
    if improved:
        best = out / "checkpoints" / "best.pt"
        tmp = best.with_suffix(".tmp")
        shutil.copyfile(out / "checkpoints" / "latest.pt", tmp)
        tmp.replace(best)
    _write_json(out / "metrics.jsonl", progress["metrics"], lines=True)
    _write_json(out / "validation.jsonl", progress["validations"], lines=True)


def train_policy(config, prefill_path, world_model_path, validation_path, out, seed=31, device="cpu",
                 *, resume=False):
    """Protect the run's checkpoints/logs against simultaneous writers."""
    # macOS, Linux and WSL release the advisory lock automatically on crash.
    import fcntl

    _check_config(config)
    out = Path(out)
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    directory = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"training is already running in {out}") from error
        return _train(config, prefill_path, world_model_path, validation_path, out, seed, device, resume)
    finally:
        os.close(directory)


def _train(config, prefill_path, world_model_path, validation_path, out, seed, device, resume):
    """Train with a persistent prefill and an episode FIFO; commit each completed iteration."""
    began = time.monotonic()
    c = config
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(c["cpu_threads"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(out)
    prefill, meta = load_dataset(prefill_path)
    manifest = load_validation(validation_path)
    if meta["manifest"].get("diagnostic") != "vertical" or any(np.any(ep.action[:, 1:]) for ep in prefill.episodes):
        raise ValueError("S2 requires vertical prefill episodes")
    if {s["seed"] for s in manifest["starts"]}.intersection(meta["reset_seeds"]):
        raise ValueError("prefill overlaps the validation manifest")
    model, initial = load_deterministic(world_model_path, device)
    spec = model.spec
    wm_arch = {k:initial["config"].get(k, False) for k in ("width", "event_weight", "kinematic", "contact_margin")}
    static = EpisodeWindows(prefill, meta, spec, device)
    near_static = _near(static, spec, c)
    actor = Actor(c["actor_width"], vertical_only=True).to(device)
    critic = Critic(c["critic_width"], c["value_scale"], upper_bound=c.get("critic_upper_bound")).to(device)
    policy = Policy(actor, spec)
    wm_params = [*model.dynamics.parameters(), *model.event.parameters()]
    groups = [{"params": [*wm_params, *model.diagnostic_reward.parameters()]}]
    if model.kinematic:
        groups.append({"params": model.liftoff.parameters(), "lr": c["wm_lr"] * .001})
        wm_params.extend(model.liftoff.parameters())
    opt_wm = torch.optim.Adam(groups, lr=c["wm_lr"])
    opt_a = torch.optim.Adam(actor.parameters(), lr=c["actor_lr"])
    opt_v = torch.optim.Adam(critic.parameters(), lr=c["critic_lr"])
    provenance = dict(config=c, seed=seed, device=str(device), prefill_replay_sha256=meta["replay_sha256"],
                      prefill_manifest_sha256=meta["train_manifest_sha256"], pretrained_sha256=file_hash(world_model_path),
                      validation_sha256=manifest["sha256"])
    checkpoints = out / "checkpoints"
    latest = checkpoints / "latest.pt"
    fifo, outcomes = [], []
    progress = dict(iteration=-1, actor_updates=0, model_updates=0, env_steps=meta["env_steps"],
                    elapsed_s=0., best_rate=-1., best_iteration=None, metrics=[], validations=[])
    if resume:
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["training_schema"] != 1 or saved["provenance"] != provenance:
            raise ValueError("resume requires identical configuration, seeds, device and input hashes")
        model.load_state_dict(saved["world_model"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        for opt, key in ((opt_wm,"wm_optimizer"), (opt_a,"actor_optimizer"), (opt_v,"critic_optimizer")):
            opt.load_state_dict(saved[key])
        fifo = [Episode(**{k:v.numpy() for k,v in record.items()}) for record in saved["fifo"]]
        outcomes = saved["fifo_provenance"]
        progress = saved["progress"]
        _recover_rng(saved, rng, device)
        _publish_files(out, progress, progress["best_iteration"] == progress["iteration"])
    else:
        checkpoints.mkdir()
        _write_json(out / "config.json", dict(provenance, code=code_version(), training_source_sha256=file_hash(Path(__file__)), wm_arch=wm_arch,
                                               prefill_path=str(prefill_path), validation_path=str(validation_path),
                                               world_model_path=str(world_model_path)))
    prior_time = progress["elapsed_s"]
    def elapsed():
        return prior_time + time.monotonic() - began

    def check_budget():
        if elapsed() >= c["max_seconds"]:
            raise BudgetExpired

    timed_policy = _TimedPolicy(policy, check_budget)

    def commit(iteration, improved):
        progress["iteration"] = iteration
        progress["elapsed_s"] = elapsed()
        state = dict(policy_checkpoint(actor,spec), training_schema=1, wm_arch=wm_arch,
                     provenance=provenance, world_model=model.state_dict(), critic=critic.state_dict(),
                     wm_optimizer=opt_wm.state_dict(), actor_optimizer=opt_a.state_dict(),
                     critic_optimizer=opt_v.state_dict(), progress=progress,
                     fifo=[_serialize_episode(ep) for ep in fifo], fifo_provenance=outcomes,
                     numpy_rng=rng.bit_generator.state, cpu_rng=torch.get_rng_state(), device_rng=_device_rng(device))
        tmp = latest.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(latest)
        _publish_files(out, progress, improved)

    def validate(iteration):
        result = evaluate_vertical(timed_policy, manifest)
        row = {k:v for k,v in result.items() if k not in ("config", "controller_config", "episodes")}
        row.update(iteration=iteration, elapsed_s=elapsed(),
                   **{k:progress[k] for k in ("env_steps","actor_updates","model_updates")})
        progress["validations"].append(row)
        if row["safe_landing_rate"] <= progress["best_rate"]:
            return False
        progress.update(best_rate=row["safe_landing_rate"], best_iteration=iteration)
        return True

    if not resume:
        commit(-1, False)
    try:
        if progress["iteration"] < 0:
            commit(0, validate(0))

        for iteration in range(progress["iteration"] + 1, c["iterations"] + 1):
            if progress["best_rate"] >= .80:
                break
            check_budget()
            collect_seed = seed * 1_000_000 + iteration
            batch, record = collect_policy(timed_policy, seed=collect_seed, episodes=c["collect_episodes"], noise_std=c["noise_std"])
            fifo.extend(batch.episodes)
            outcomes.extend(record["episode_provenance"])
            fifo = fifo[-c["fifo_episodes"]:]
            outcomes = outcomes[-c["fifo_episodes"]:]
            progress["env_steps"] += record["env_steps"]
            live = EpisodeWindows(Replay(fifo), {"episode_provenance":outcomes}, spec, device)
            near_live = _near(live, spec, c)
            wm_counts = _counts(c["wm_batch"], c["prefill_fraction"])
            totals = Counter()
            wm_time = time.monotonic()
            for _ in range(c["wm_updates"]):
                check_budget()
                samples = [source.sample(rng, n, 8) for source,n in zip((static,live),wm_counts,strict=True)]
                losses = model.losses(*(torch.cat(parts) for parts in zip(*samples,strict=True)))
                opt_wm.zero_grad(set_to_none=True)
                sum(losses).backward()
                torch.nn.utils.clip_grad_norm_(wm_params, 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.diagnostic_reward.parameters(), 1., error_if_nonfinite=True)
                opt_wm.step()
                for loss,name in zip(losses,("dynamics_loss","event_loss","reward_loss"),strict=True):
                    totals[name] += loss.item()/c["wm_updates"]
                progress["model_updates"] += 1
            wm_hz = c["wm_updates"] / (time.monotonic() - wm_time)
            counts = _counts(c["imagined_batch"], c["prefill_fraction"])
            gradient_time = time.monotonic()
            for _ in range(c["actor_updates"]):
                check_budget()
                starts = torch.cat([_sample_starts(pool,near,rng,n,c["near_fraction"]) for pool,near,n in
                                    zip((static,live),(near_static,near_live),counts,strict=True)])
                metrics = update_actor_critic(model,actor,critic,starts,opt_a,opt_v,horizon=c["horizon"],
                                              gamma=c["gamma"],lam=c["lambda"],critic_epochs=c["critic_epochs"])
                for key,value in metrics.items():
                    totals[key] += value/c["actor_updates"]
                progress["actor_updates"] += 1
            actor_hz = c["actor_updates"]/(time.monotonic()-gradient_time)
            row = dict(totals, iteration=iteration, elapsed_s=elapsed(), actor_updates_per_second=actor_hz,
                       model_updates_per_second=wm_hz,
                       env_steps=progress["env_steps"], actor_updates=progress["actor_updates"],
                       model_updates=progress["model_updates"], collect_seed=collect_seed,
                       collect_status_counts=record["status_counts"], fifo_size=len(fifo),
                       collection_steps=record["env_steps"], fifo_status_counts=dict(Counter(r["status"] for r in outcomes)))
            progress["metrics"].append(row)
            due = iteration % c["validate_every"] == 0 or iteration == c["iterations"]
            improved = validate(iteration) if due and elapsed() < c["max_seconds"] else False
            commit(iteration, improved)
    except BudgetExpired:
        # Discard the unfinished iteration as one transaction, including RNG draws.
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        progress = saved["progress"]
        progress["elapsed_s"] = elapsed()
        progress["time_budget_exhausted"] = True
        tmp = latest.with_suffix(".tmp")
        torch.save(saved, tmp)
        tmp.replace(latest)
        _publish_files(out, progress, progress["best_iteration"] == progress["iteration"])
    summary = {k:v for k,v in progress.items() if k not in ("metrics","validations")}
    summary["first_passing_validation"] = next((v for v in progress["validations"] if v["safe_landing_rate"] >= .8), None)
    summary["threshold_on_supplied_manifest"] = progress["best_rate"] >= .80
    official = manifest["seed"] == 2026 and len(manifest["starts"]) == 100
    summary["official_validation"] = official
    summary["passed"] = official and summary["threshold_on_supplied_manifest"]
    summary["time_budget_s"] = c["max_seconds"]
    _write_json(out / "summary.json", summary)
    return summary
