from dataclasses import replace

import numpy as np
import pytest
import torch
from mun_lander import LanderEnv, SimConfig
from mun_lander.state import LanderState

from mun_agent.interface import deterministic_spec, episode_features, to_internal_action, validate_deterministic_spec
from mun_agent.replay import Episode, Replay
from mun_agent.task_reward import advance_history, reward_parity, rule_outcome, task_reward


def test_causal_history_matches_real_and_imagined_frames_and_detaches_counters():
    spec = deterministic_spec(SimConfig())
    eps = spec["ground_epsilon_m"]
    heights = np.array([2*eps, eps/2, 0, eps, 0])
    obs = np.zeros((len(heights), 14), np.float32)
    obs[:, 2] = heights
    obs[:, 6] = 1
    states = torch.from_numpy(episode_features(obs, spec))
    torch.testing.assert_close(states[:, 16] * spec["ground_history_cap"], torch.tensor([0., 1., 2., 0., 1.]))
    current = states[:-1].clone().requires_grad_()
    physical = states[1:, :16].clone().requires_grad_()
    advanced = advance_history(current, physical, spec)
    torch.testing.assert_close(advanced, states[1:])
    advanced.sum().backward()
    assert current.grad is None
    torch.testing.assert_close(physical.grad, torch.ones_like(physical))
    validate_deterministic_spec(spec, SimConfig())
    with pytest.raises(ValueError):
        validate_deterministic_spec(dict(spec, episode_step_cap=1), SimConfig())


def test_abort_override_and_gradients_with_upright_attitude():
    spec = deterministic_spec(SimConfig())
    obs = np.zeros((4, 14), dtype=np.float32)
    obs[:, 6] = 1
    obs[:, 2] = [1, 1, 0, 1]
    obs[1, 0] = 2000
    obs[-1, 13] = 2
    states = torch.from_numpy(episode_features(obs, spec))
    states[-1, 17] = 1
    assert rule_outcome(states, spec).tolist() == [3, 2, 0, 1]
    y = states.clone().requires_grad_()
    probs = torch.tensor([[0.8, 0.15, 0.05]] * 4, requires_grad=True)
    actions = torch.zeros(4, 4, requires_grad=True)
    result, continuation = task_reward(states, actions, y, probs, spec)
    assert continuation.tolist() == pytest.approx([0., 0., 0.8, 0.])
    assert result[2].item() == pytest.approx(10 - 0.002)
    assert (result[[0, 1, 3]] <= -50).all()
    result.sum().backward()
    assert y.grad.isfinite().all() and actions.grad.isfinite().all()
    assert probs.grad[:, 1].tolist() == [0, 0, 100, 0]


def test_reward_and_recorded_duration_at_impact(tmp_path):
    co = SimConfig()
    env = LanderEnv(co)
    initial = LanderState(np.array([2., 0., 0.23]), np.array([0., 0., -2.1]),
                          np.array([1., 0., 0., 0.]), np.zeros(3), 100.)
    obs, _ = env.reset(options={"initial_state": initial})
    frames = [obs]
    rewards, durations, actions = [], [], []
    while True:
        action = np.array([0.05, 0.1, 0.2, -0.2], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        frames.append(next_obs)
        rewards.append(reward)
        durations.append(info["dt"])
        actions.append(action)
        if terminated or truncated:
            break
    env.close()
    assert len(rewards) == 2 and info["status"] == "crash" and durations[-1] < 0.1
    episode = Episode(frames, actions, rewards, [False, True], [False, False], dt=durations)
    replay = Replay([episode])
    replay.save(tmp_path / "replay.npz")
    restored = Replay.load(tmp_path / "replay.npz")
    np.testing.assert_array_equal(restored.episodes[0].dt, episode.dt)
    legacy = replace(episode, dt=None)
    Replay([legacy, episode]).save(tmp_path / "mixed.npz")
    mixed = Replay.load(tmp_path / "mixed.npz").episodes
    assert mixed[0].dt is None
    np.testing.assert_array_equal(mixed[1].dt, episode.dt)
    metadata = {"episode_provenance": [{"status": "crash"}]}
    spec = deterministic_spec(co)
    with pytest.raises(ValueError, match="dt"):
        reward_parity(Replay([legacy]), metadata, spec)
    parity = reward_parity(restored, metadata, spec)
    assert all(dist["max"] < 1e-5 and dist["count"] == 1 for dist in parity.values())
    states = torch.from_numpy(episode_features(episode.obs, spec))
    actions = torch.from_numpy(to_internal_action(episode.action))
    probs = torch.tensor([0., 0., 1.])
    full_dt_reward, _ = task_reward(states[-2], actions[-1], states[-1], probs, spec)
    assert abs(full_dt_reward.item() - rewards[-1]) > 1e-3


def test_parity_for_episode_with_only_a_terminal_step():
    spec = deterministic_spec(SimConfig())
    obs = np.zeros((2, 14), dtype=np.float32)
    obs[:, 6] = 1
    ep = Episode(obs, [[0, 0, 0, 0]], [-100], [True], [False], dt=[0.])
    result = reward_parity(Replay([ep]), {"episode_provenance": [{"status": "crash"}]}, spec)
    assert result["nonterminal_process_abs_error"]["count"] == 0
    assert result["nonterminal_process_abs_error"]["mean"] is None
    assert result["terminal_total_abs_error"]["max"] == 0
