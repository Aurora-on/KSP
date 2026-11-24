"""Contracts for deterministic rollout fitting, probability correction and model gradients."""
import numpy as np
import pytest
import torch
from mun_lander import SimConfig
from mun_agent.deterministic import DeterministicModel
from mun_agent.deterministic_fit import EpisodeWindows
from mun_agent.interface import deterministic_spec
from mun_agent.replay import Episode, Replay


def small_model():
    torch.manual_seed(31)
    return DeterministicModel(deterministic_spec(SimConfig()), width=16, event_weight=40.0)


def test_step_corrects_weighted_logits_and_advances_history():
    model = small_model()
    for head in (model.event, model.dynamics):
        for p in head.parameters():
            p.data.zero_()
    model.event[-1].bias.data.copy_(torch.tensor([0.0, np.log(40), np.log(40)]))
    state = torch.zeros(2, 18)
    state[1, 2] = 1
    next_state, probs, reward = model.step(state, torch.zeros(2, 4))
    torch.testing.assert_close(probs, torch.full((2, 3), 1/3))
    torch.testing.assert_close(next_state[:, :16], state[:, :16])
    assert next_state[0, 16] > 0 and next_state[1, 16] == 0
    assert reward.shape == (2,)


@pytest.mark.parametrize("contact_margin", [False, True])
def test_reward_is_detached_and_step_remains_differentiable(contact_margin):
    model = DeterministicModel(deterministic_spec(SimConfig()), width=16,
                               kinematic=contact_margin, contact_margin=contact_margin)
    state = torch.rand(2, 18, requires_grad=True)
    action = torch.rand(2, 4, requires_grad=True)
    next_state, probs, reward = model.step(state, action)
    reward.sum().backward(retain_graph=True)
    assert all(p.grad is None for p in [*model.dynamics.parameters(), *model.event.parameters()])
    assert action.grad is None
    (next_state[:, :16].sum() + probs[:, 1].sum()).backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_padding_does_not_advance_initial_or_change_loss():
    model = small_model()
    first = torch.rand(3, 18)
    actions, targets = torch.rand(3, 8, 4), torch.rand(3, 8, 18)
    labels = torch.zeros(3, 8, dtype=torch.long)
    labels[:, 4] = torch.tensor([1, 2, -1])
    rewards = torch.rand(3, 8)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[:, 2:5] = True
    full = model.losses(first, actions, targets, labels, rewards, mask)
    cut = model.losses(first, actions[:, 2:5], targets[:, 2:5], labels[:, 2:5],
                       rewards[:, 2:5], mask[:, 2:5])
    for observed, expected in zip(full, cut):
        torch.testing.assert_close(observed, expected)


def test_training_windows_cover_ends_and_preserve_episode_labels():
    ep_list = []
    for n in (2, 5):
        flags = np.arange(n) == n - 1
        obs = np.zeros((n + 1, 14))
        obs[:, 6] = 1
        obs[:, 0] = len(ep_list) + 1
        ep_list.append(Episode(obs, np.zeros((n, 4)), np.arange(n), flags, np.zeros(n, dtype=bool)))
    provenance = {"episode_provenance": [{"status": "timeout"}, {"status": "crash"}]}
    cache = EpisodeWindows(Replay(ep_list), provenance, deterministic_spec(SimConfig()), "cpu")
    initial, _, targets, labels, _, mask = cache.sample(np.random.default_rng(0), 5000, 8)
    assert torch.allclose(initial[:, None, 0], targets[:, :, 0])
    # Two terminal transitions have the same expected coverage as any other transition.
    assert ((labels[mask] == -1).float().mean().item()) == pytest.approx(1 / 7, abs=0.01)
    assert ((labels[mask] == 2).float().mean().item()) == pytest.approx(1 / 7, abs=0.01)


def test_ground_branch_learns_liftoff_and_preserves_action_gradient():
    model = DeterministicModel(deterministic_spec(SimConfig()), width=16, kinematic=True)
    state = torch.zeros(8, 18)
    state[:, 6] = state[:, 10] = 1
    state[:, 15] = torch.linspace(0.8, 1.0, 8)
    state[:, 16] = 0.5
    actions = torch.zeros(8, 4)
    actions[:, 0] = torch.linspace(0.1, 0.9, 8)
    targets = state.clone()
    weights = torch.tensor([-0.032, 0.1, -0.034])
    targets[:, 5] = model.lift_inputs(state, actions) @ weights
    targets[:, 2] = 0.01
    model.initialize_liftoff(state, actions, targets)
    probe = actions.clone()
    probe[:4, 0] = -1
    probe.requires_grad_()
    predicted = model.advance(state, probe)
    assert (predicted[:4, [2, 5]] == 0).all()
    torch.testing.assert_close(predicted[4:, 5], targets[4:, 5])
    assert (predicted[4:, 2] > 0).all()
    predicted[:, 2].sum().backward()
    assert (probe.grad[4:, 0] > 0).all()
