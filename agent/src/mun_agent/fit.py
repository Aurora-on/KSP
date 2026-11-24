"""Fit the world model on fixed simulator replay and produce a portable checkpoint."""
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mun_lander import SimConfig

from .dataset import load_dataset
from .interface import interface_spec, preprocess, to_internal_action, validate_interface_spec
from .world_model import WorldModel


@dataclass(frozen=True)
class FitConfig:
    batch_size: int
    length: int
    updates: int
    learning_rate: float

    def __post_init__(self):
        for field in (self.batch_size, self.length, self.updates):
            if type(field) is not int or field <= 0:
                raise ValueError("batch_size, length and updates must be positive integers")
        if (type(self.learning_rate) not in (float, int) or not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")


def computation_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cpu":
        # Small sequential GRU updates spend more time scheduling large thread pools.
        torch.set_num_threads(1)
    return torch.device(requested)


def code_version():
    directory = Path(__file__).resolve().parent
    try:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=directory, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def tensors(replay, config, rng, spec, device):
    batch = replay.sample(config.batch_size, config.length, rng)
    arrays = (preprocess(batch["obs"], spec), to_internal_action(batch["action"]),
              batch["reward"], batch["is_terminal"], batch["mask"])
    return tuple(torch.as_tensor(a, device=device) for a in arrays)


def fit(settings, dataset_path, out, seed, device):
    config = FitConfig(**settings["fit"])
    replay, record = load_dataset(dataset_path)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = WorldModel(**settings["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    spec = interface_spec(SimConfig())
    # Keep identical real transitions for both measurements, independent of training draws.
    probe = tensors(replay, config, np.random.default_rng([seed, 1]), spec, device)
    provenance = {"config": settings, "seed": seed, "device": str(device), "code": code_version(),
                  "train_manifest_sha256": record["train_manifest_sha256"],
                  "train_replay_sha256": record["replay_sha256"], "env_steps": record["env_steps"]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    start = time.perf_counter()
    with torch.no_grad():
        before = {k: v.item() for k, v in model.loss(*probe, sample=False).items()}
    with (out / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for update in range(1, config.updates + 1):
            losses = model.loss(*tensors(replay, config, rng, spec, device))
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0, error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.perf_counter() - start
            metrics = {name: value.detach().item() for name, value in losses.items()}
            metrics.update(update=update, env_steps=record["env_steps"], elapsed_s=elapsed,
                           updates_per_second=update / elapsed)
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
    with torch.no_grad():
        after = {k: v.item() for k, v in model.loss(*probe, sample=False).items()}
    summary = {"updates": config.updates, "probe_loss_before": before["loss"],
               "probe_loss_after": after["loss"], "probe_before": before, "probe_after": after,
               "probe_valid_steps": int(probe[-1].sum().item()),
               "probe_terminal_steps": int((probe[-1] & probe[-2]).sum().item()),
               "elapsed_s": time.perf_counter() - start}
    checkpoint = dict(provenance, interface=spec, model=model.state_dict(),
                      train_reset_seeds=[s["seed"] for s in record["manifest"]["starts"]],
                      schema_version=1, **summary)
    path = out / "world_model.pt"
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def load_world_model(path, device, env_config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("unsupported world-model checkpoint")
    validate_interface_spec(checkpoint["interface"], env_config)
    model = WorldModel(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
