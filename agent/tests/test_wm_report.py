import json

import numpy as np
import pytest
import torch
from mun_lander import SimConfig

from mun_agent.interface import interface_spec
from mun_agent.replay import Episode, Replay
from mun_agent.wm_report import open_loop_report


class CountingModel(torch.nn.Module):
    """Know the elapsed step from posterior history, with biased position dynamics."""

    def __init__(self, degenerate=False):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0, dtype=torch.float64))
        self.degenerate = degenerate

    def observe_initial(self, features, sample=True):
        assert not sample and not self.training and features.dtype == self.anchor.dtype
        return features, torch.zeros_like(features[:, 0])

    def observe_step(self, state, action, next_features, sample=True):
        assert not sample
        return next_features, state[1] + 1

    def imagine_step(self, state, action, sample=True):
        assert not sample
        old, clock = state
        new = old.clone()
        new[:, 0] += action[:, 0] / interface_spec(SimConfig())["scales"]["position"]
        return new, clock + 1

    def predict(self, state):
        features, clock = state
        if self.degenerate:
            features = features.clone()
            features[:, 6:12] = 0
        return features, clock, torch.where(clock % 3 == 0, 0.1, 0.5)


def make_episode(length, truncated):
    t = np.arange(length + 1)
    obs = np.zeros((length + 1, 14), np.float32)
    obs[:, 0] = t * t / 16
    obs[:, 3] = 0.125 * t
    obs[:, 10] = 0.25 * t
    obs[:, 13] = 180 - 0.5 * t
    # Every transition flips between 0 and 30 degrees of tilt about x.
    half_angle = np.deg2rad(15 * (t % 2))
    obs[:, 6] = np.cos(half_angle)
    obs[:, 7] = np.sin(half_angle)
    action = np.zeros((length, 4), np.float32)
    action[:, 0] = (t[:-1] % 4) / 4
    end_flags = np.zeros(length, bool)
    end_flags[-1] = True
    return Episode(obs, action, 2 * t[1:], end_flags & ~np.bool_(truncated), end_flags & truncated)


def test_every_open_loop_origin_against_independent_scalar_oracle():
    # Different lengths exercise shrinking batches. L=140 also crosses a tile boundary.
    replay = Replay([make_episode(51, True), make_episode(140, False), make_episode(8, True)])
    model = CountingModel()
    report = open_loop_report(model, replay, interface_spec(SimConfig()))
    assert model.training and report["episodes"] == 3
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for h in (1, 5, 15, 50):
        estimated = report["horizons"][str(h)]
        position_errors, reward_errors = [], []
        tp = fp = tn = fn = 0
        for ep in replay.episodes:
            for start in range(len(ep) - h + 1):
                end = start + h
                prediction = ep.obs[start, 0] + (2 * ep.action[start:end, 0] - 1).sum()
                position_errors.append(abs(prediction - ep.obs[end, 0]))
                reward_errors.append(abs(end - ep.reward[end - 1]))
                predicted_end = end % 3 == 0
                done = end == len(ep)
                tp += int(predicted_end and done)
                fp += int(predicted_end and not done)
                tn += int(not predicted_end and not done)
                fn += int(not predicted_end and done)
        assert estimated["count"] == len(position_errors)
        assert estimated["position_error_m"] == pytest.approx(np.mean(position_errors))
        assert estimated["velocity_error_m_s"] == pytest.approx(0.125 * h)
        assert estimated["angular_velocity_error_rad_s"] == pytest.approx(0.25 * h)
        assert estimated["tilt_mae_deg"] == pytest.approx(30 * (h % 2), abs=1e-5)
        assert estimated["tilt_valid_count"] == estimated["count"]
        assert estimated["fuel_mae_kg"] == pytest.approx(0.5 * h, abs=1e-5)
        assert estimated["reward_mae"] == pytest.approx(np.mean(reward_errors))
        assert estimated["terminal"] == {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall": tp / (tp + fn),
            "false_positive_rate": fp / (fp + tn),
        }


def test_empty_denominators_and_degenerate_tilt():
    replay = Replay([make_episode(1, True)])
    model = CountingModel(degenerate=True).eval()
    horizons = open_loop_report(model, replay, interface_spec(SimConfig()))["horizons"]
    assert not model.training
    assert horizons["1"]["tilt_valid_count"] == 0
    assert horizons["1"]["tilt_invalid_count"] == 1
    assert horizons["1"]["tilt_mae_deg"] is None
    assert horizons["1"]["terminal"]["false_positive_rate"] is None
    assert horizons["5"]["count"] == 0 and horizons["5"]["reward_mae"] is None
    assert horizons["5"]["terminal"]["recall"] is None


class GateOracle(torch.nn.Module):
    """Prior reward grows with steps since last observation; posterior uses a separate head value."""
    def observe_initial(self, features, sample=False):
        assert features.shape[-1] == 17
        return features[:, :16], torch.zeros_like(features[:, 0])

    def observe_step(self, state, action, next_features, sample=False):
        return self.observe_initial(next_features, sample)

    def imagine_step(self, state, action, sample=False):
        return state[0], state[1] + 1

    def predict(self, state):
        features, imagined_steps = state
        sign = torch.sign(features[:, 0])
        reward = sign * torch.where(imagined_steps > 0, imagined_steps, 50)
        return features, reward, torch.where(sign > 0, 0.25, 0.75)


def test_baselines_near_ground_endpoints_status_and_last15_reward_sum():
    from mun_agent.interface import preprocess

    spec = interface_spec(SimConfig(), ground_history=True)
    episodes, provenance = [], []
    for i in range(10):
        length = 29 + i
        ep = make_episode(length, False)
        obs = ep.obs.copy()
        obs[:, 0] = 1 if i < 5 else -1
        obs[:, 2] = np.maximum(28 - np.arange(length + 1), 0)
        obs[:, 3:6] = 0
        obs[:, 6:10] = (1, 0, 0, 0)
        # Seven out of ten meet G5, with one exactly on the 50% boundary.
        reward = np.zeros(length, np.float32)
        reward[-1] = np.sign(obs[0, 0]) * (240 if i < 7 else 241)
        episodes.append(Episode(obs, ep.action, reward, ep.terminated, ep.truncated))
        status = "safe_landing" if i < 5 else "crash"
        if i in (4, 5):
            status = "crash" if i == 4 else "safe_landing"  # recorded status governs G4
        provenance.append({"status": status, "start_id": f"held-{i}"})

    train_mean = preprocess(episodes[0].obs[0], spec)
    report = open_loop_report(GateOracle(), Replay(episodes), spec, train_mean=train_mean,
                              provenance=provenance, horizons=(1, 5, 15))
    records = report["terminal_episodes"]
    for i, (ep, row) in enumerate(zip(episodes, records, strict=True)):
        assert row["start_id"] == f"held-{i}"
        sign = np.sign(ep.obs[0, 0])
        assert row["posterior_reward"] == 50 * sign
        assert row["prior_rewards"] == {str(h): h * sign for h in (1, 5, 15)}
        assert row["g5_predicted_sum"] == 120 * sign
        assert row["g5_true_sum"] == ep.reward[-1]
        assert row["g5_hit"] == (i < 7)

    gates = report["gates"]
    assert all(g["pass"] for g in gates.values())
    assert not report["acceptance"]["pass"]  # fewer than 50 held-out episodes
    assert gates["G4"]["groups"]["safe_landing"] == {"hits": 4, "total": 5, "rate": .8}
    assert gates["G4"]["groups"]["crash"] == {"hits": 4, "total": 5, "rate": .8}
    assert gates["G5"]["hits"] == 7 and gates["G5"]["total"] == 10
    assert gates["G5"]["rate"] == .7
    assert report["posterior"]["position_error_m"] == pytest.approx(0)
    assert report["horizons"]["15"]["continue_probability"]["terminal"]["median"] == .5

    h = 15
    row = report["horizons"][str(h)]
    expected_hold, expected_mean = [], []
    for ep in episodes:
        endpoints = ep.obs[h:]
        mask = endpoints[:, 2] < 10
        expected_hold.extend(np.linalg.norm(ep.obs[:-h][mask, :3] - endpoints[mask, :3], axis=1))
        expected_mean.extend(np.linalg.norm(episodes[0].obs[0, :3] - endpoints[mask, :3], axis=1))
    ground = row["near_ground"]
    assert 0 < ground["count"] < row["count"] and ground["count"] == len(expected_hold)
    assert ground["baselines"]["persistence"]["position_error_m"] == pytest.approx(np.mean(expected_hold))
    assert ground["baselines"]["train_mean"]["position_error_m"] == pytest.approx(np.mean(expected_mean))
    assert gates["G3"]["count"] == ground["count"]
    assert row["reward_by_endpoint"]["terminal"]["zero_reward_mae"] == pytest.approx(
        np.mean([abs(ep.reward[-1]) for ep in episodes]))


def test_missing_provenance_and_short_episodes_have_unavailable_gates():
    replay = Replay([make_episode(10, True)])
    report = open_loop_report(CountingModel(), replay, interface_spec(SimConfig()), horizons=(1,))
    assert report["gates"]["G4"]["pass"] is None
    assert report["gates"]["G4"]["known_status_count"] == 0
    assert {k: report["gates"][k]["pass"] for k in ("G2", "G3", "G5")} == dict.fromkeys(("G2", "G3", "G5"))
    assert report["gates"]["G5"]["total"] == 1 and report["gates"]["G5"]["evaluated_count"] == 0
    assert report["horizons"]["1"]["baselines"]["train_mean"] is None
    assert report["terminal_episodes"][0]["g5_predicted_sum"] is None
