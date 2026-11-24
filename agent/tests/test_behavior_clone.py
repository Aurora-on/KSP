from types import SimpleNamespace

import numpy as np
import torch

from mun_agent.behavior_clone import fit_clone
from mun_agent.imagination import Actor


def test_clone_uses_source_and_noise_not_outcome_and_fits_active_action():
    torch.manual_seed(19)
    actor = Actor(width=16, vertical_only=True)
    state = torch.zeros(20, 18)
    state[::2, 2] = 3.
    target = torch.zeros(20, 4)
    target[:8, 0] = .6
    target[8:, 0] = -.8
    target[:, 1:] = 1.
    static = SimpleNamespace(current=state, actions=target, offsets=np.arange(5)*4, lengths=np.full(5, 4))
    metadata = {"episode_provenance": [
        {"controller": "scripted", "noise_std": 0., "status": "safe_landing"},
        {"controller": "scripted", "noise_std": .1, "status": "crash"},
        {"controller": "scripted", "noise_std": .2, "status": "safe_landing"},
        {"controller": "hard_landing", "noise_std": 0., "status": "safe_landing"},
        {"controller": "random", "noise_std": None, "status": "safe_landing"}]}
    checks = []
    summary = fit_clone(actor, static, metadata, {"bc_updates": 90, "bc_batch": 8, "bc_lr": .01},
                        np.random.default_rng(2), lambda: checks.append(True))
    assert len(checks) == 90
    assert summary["selected_episodes"] == 2
    assert summary["selected_transitions"] == 8 and summary["selected_near_transitions"] == 4
    assert summary["probe_mae_before"] > .5 and summary["probe_mae_after"] < .05
    assert (actor(state)[:, 1:] == 0.).all()


def test_clone_samples_uniformly_when_demonstrations_stay_airborne():
    actor = Actor(width=8)
    state = torch.zeros(4, 18)
    state[:, 2] = 2.
    static = SimpleNamespace(current=state, actions=torch.zeros(4, 4),
                             offsets=np.array([0]), lengths=np.array([4]))
    metadata = {"episode_provenance": [{"controller": "scripted", "noise_std": 0.}]}
    summary = fit_clone(actor, static, metadata, {"bc_updates": 1, "bc_batch": 8, "bc_lr": .001},
                        np.random.default_rng(2), lambda: None)
    assert summary["selected_near_transitions"] == 0
    assert np.isfinite(summary["probe_mae_after"])
