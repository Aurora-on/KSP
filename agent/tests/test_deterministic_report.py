import json

import numpy as np
import pytest
import torch
from mun_lander import SimConfig

from mun_agent.deterministic_report import deterministic_report
from mun_agent.interface import deterministic_spec, preprocess
from mun_agent.replay import Episode, Replay


class Oracle(torch.nn.Module):
    """Action signals the final event; assign small hazards along the way."""
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def step(self, state, action):
        new = state.clone()
        new[:, 2] = (state[:, 2] - 1 / self.spec["scales"]["position"]).clamp_min(0)
        probs = state.new_tensor([.996, .002, .002]).expand(len(state), 3).clone()
        probs[action[:, 0] > .5] = state.new_tensor([0., 1., 0.])
        probs[action[:, 0] < -.5] = state.new_tensor([0., 0., 1.])
        return new, probs, torch.full_like(state[:, 0], 7.)


def _episode(status, early=False):
    length = 45
    obs = np.zeros((length + 1, 14), np.float32)
    obs[:, 2] = np.maximum(14 - np.arange(length + 1), 0)
    obs[:, 6] = 1.
    obs[:, 13] = 100.
    action = np.zeros((length, 4), np.float32)
    action[:, 0] = .5
    action[-1, 0] = 1. if status == "safe_landing" else 0.
    if early:
        action[20, 0] = 1.
    end = np.arange(length) == length - 1
    reward = np.full(length, -.002, np.float32)
    reward[-1] += 100 if status == "safe_landing" else -100.
    return Episode(obs, action, reward, end, np.zeros_like(end))


def test_s1_oracle_squared_survival_first_alarm_and_gate_denominators():
    spec = deterministic_spec(SimConfig())
    episodes = [_episode("safe_landing"), _episode("crash", early=True)]
    replay = Replay(episodes)
    model = Oracle(spec)
    report = deterministic_report(model, replay, spec, train_mean=preprocess(episodes[0].obs[0], spec),
                                  provenance=[{"status": name} for name in ("safe_landing", "crash")])
    assert model.training and json.loads(json.dumps(report, allow_nan=False)) == report
    gates = report["gates"]
    assert gates["W1"]["pass"] and gates["W2"]["pass"]
    for h in (1, 5, 15, 32):
        row = report["horizons"][str(h)]
        expected = 2 * (len(episodes[0]) - h + 1)
        assert all(value["count"] == expected for value in row["all"].values())
        assert row["near_ground"]["deterministic"]["count"] == 2 * np.sum(episodes[0].obs[h:, 2] < 10)
    for group in gates["W3"]["by_final_status"].values():
        assert group["count"] == 1 and group["velocity_error_m_s"] == 0
    assert gates["W4"]["overall"] == {"hits": 1, "total": 2, "rate": .5}
    assert gates["W4"]["nonterminal_end_probability"]["mean"] == pytest.approx((.004 * 87 + 1) / 88)
    safe, crash = report["terminal_episodes"]
    assert safe["w4"] == {"first_step": 45, "first_status": "safe_landing", "hit": True}
    assert crash["w4"] == {"first_step": 21, "first_status": "safe_landing", "hit": False}
    for record, bonus in ((safe, 100), (crash, -100)):
        expected = sum(-.002 * .996 ** k for k in range(15)) + bonus * .996 ** 14
        assert record["w5"]["predicted_return"] == pytest.approx(expected, rel=1e-5)
    assert gates["W5"]["pass"] and not report["acceptance"]["pass"]
    assert gates["W6"]["terminal_steps"]["mean"] == pytest.approx(100)
    near = report["survival_32_nonterminal_windows"]["near_ground"]
    assert near["max"] == pytest.approx(.996 ** 32)
    assert near["min"] == 0.
