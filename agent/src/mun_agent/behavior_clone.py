"""Initialize the actor from low-noise scripted demonstrations in permanent replay."""
import math

import numpy as np
import torch
from mun_lander import SimConfig

from .interface import deterministic_spec


def fit_clone(actor, static, meta, config, rng, check_budget):
    """Fit deterministic internal-coordinate actions without changing world-model or critic weights."""
    updates, batch, lr = config["bc_updates"], config["bc_batch"], config["bc_lr"]
    if type(updates) is not int or updates < 1 or type(batch) is not int or batch < 2:
        raise ValueError("behavior cloning requires positive updates and batch >= 2")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("bc_lr must be finite and positive")
    spans = []
    for row, offset, size in zip(meta["episode_provenance"], static.offsets, static.lengths, strict=True):
        if row["controller"] == "scripted" and row["noise_std"] is not None and row["noise_std"] <= .1:
            spans.append(np.arange(offset, offset + size))
    if not spans:
        raise ValueError("behavior cloning needs scripted demonstrations with native noise <= 0.1")
    eligible = np.concatenate(spans)
    scale = deterministic_spec(SimConfig())["scales"]["position"]
    heights = static.current[:, 2].detach().cpu().numpy() * scale
    near = eligible[heights[eligible] < 10.]
    device = static.current.device
    channels = 1 if actor.vertical_only else 4

    def sample(random, count):
        near_count = count // 2 if len(near) else 0
        indices = random.choice(eligible, count - near_count)
        if near_count:
            indices = np.concatenate((indices, random.choice(near, near_count)))
        return torch.as_tensor(indices, device=device)

    def error(indices):
        predicted = actor(static.current[indices])[:, :channels]
        return predicted - static.actions[indices, :channels]

    # These fixed, within-prefill measurements quantify imitation fit.
    # Policy performance is evaluated separately on the disjoint simulator manifest.
    probe = sample(np.random.default_rng(0), min(2048, max(len(eligible), 2)))
    with torch.no_grad():
        before = error(probe).abs().mean().item()
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    total_loss = 0.
    for _ in range(updates):
        check_budget()
        loss = error(sample(rng, batch)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    with torch.no_grad():
        after = error(probe).abs().mean().item()
    return {"updates": updates, "batch": batch, "learning_rate": lr, "selected_episodes": len(spans),
            "selected_transitions": len(eligible), "selected_near_transitions": len(near),
            "probe_count": len(probe), "probe_mae_before": before, "probe_mae_after": after,
            "mean_training_mse": total_loss / updates}
