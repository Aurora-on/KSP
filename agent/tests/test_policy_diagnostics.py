import numpy as np
import pytest
import torch
from mun_lander import SimConfig
from torch import nn

from mun_agent.interface import deterministic_spec
from mun_agent.policy_diagnostics import source_diagnostics
from mun_agent.replay import Episode, Replay


def test_diagnostics_keep_horizons_and_outcomes_separate():
    class StaticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.device_ref = nn.Parameter(torch.tensor(0.))
            self.spec = deterministic_spec(SimConfig())

        def advance(self, state, action):
            return state

        def step(self, state, action):
            probs = torch.zeros((*state.shape[:-1], 3), device=state.device)
            probs[..., 0] = 1.
            return state, probs, None

    class AltitudeCritic(nn.Module):
        def forward(self, state):
            return 20 * state[..., 2]

    episodes = []
    for length in (33, 5):
        frames = np.zeros((length + 1, 14), dtype=np.float32)
        frames[:, 2] = np.arange(length + 1)[::-1]
        frames[:, 6] = 1
        frames[:, 13] = 500
        flags = np.arange(length) == length - 1
        episodes.append(Episode(frames, np.zeros((length, 4)), np.ones(length), flags, np.zeros_like(flags)))
    rows = [{"status": "crash", "touchdown_speed": 5.}, {"status": "safe_landing", "touchdown_speed": .7}]
    report = source_diagnostics(StaticModel(), AltitudeCritic(), Replay(episodes), rows, .5)
    assert report["open_loop"]["1"]["count"] == 38
    assert report["open_loop"]["32"]["count"] == 2
    assert report["open_loop"]["32"]["position_error_m"] == pytest.approx(32)
    assert report["terminal_window_32"]["episodes"] == 1
    assert report["by_outcome"]["safe_landing"]["first_touchdown_speed_m_s"]["mean"] == .7
    phase = report["by_outcome"]["crash"]["values_by_phase"]["first_below_10m"]
    assert phase["critic_value"]["mean"] == 9
    assert phase["observed_discounted_return"]["mean"] == pytest.approx(sum(.5**np.arange(9)))
