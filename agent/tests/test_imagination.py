import numpy as np
import pytest
import torch
from torch import nn
from mun_lander import SimConfig

from mun_agent import imagination
from mun_agent.imagination import Actor, Critic, lambda_returns, update_actor_critic
from mun_agent.interface import deterministic_spec, episode_features
from mun_agent.policy import Policy, load_policy, policy_checkpoint


@pytest.mark.parametrize("lam", [0, .45, 1])
def test_lambda_return_matches_explicit_n_step_mixture(lam):
    r = torch.tensor([[1., 2.], [4., 8.], [16., 32.]])
    c = torch.tensor([[.7, 1.], [0., .5], [1., 0.]])
    v = torch.tensor([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])
    gamma = .9
    result = lambda_returns(r, c, v, gamma, lam)
    for t in range(len(r)):
        expected = torch.zeros_like(r[0])
        for n in range(1, len(r) - t + 1):
            total, product = torch.zeros_like(r[0]), torch.ones_like(r[0])
            for j in range(t, t + n):
                total += product * r[j]
                product *= gamma * c[j]
            total += product * v[t + n]
            mix = lam**(n - 1) * (1 if n == len(r) - t else 1 - lam)
            expected += mix * total
        torch.testing.assert_close(result[t], expected)


def test_actor_gradient_includes_survival_and_terminal_value(monkeypatch):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.75))
            self.spec = {}

        def step(self, state, action):
            following = state + self.weight * action[:, :1]
            continuation = torch.sigmoid(action[:, 0] - state[:, 0])
            return following, continuation, None

    class ToyActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(.2))

        def forward(self, state):
            return (self.bias + state[:, :1]).expand(-1, 4)

    class ToyCritic(Critic):
        def forward(self, state):
            return self.network[-1].bias + state[..., 0] * .3

    def reward(s, a, ns, prob, spec):
        return ns[:, 0] * .1, prob
    monkeypatch.setattr(imagination, "task_reward", reward)
    wm, actor, critic = ToyModel(), ToyActor(), ToyCritic(width=8)
    starts = torch.zeros(3, 18)

    def objective(bias):
        state, c, total = 0., 1., 0.
        for t in range(2):
            action = bias + state
            next_state = state + .75 * action
            total += .9**t * c * next_state * .1
            c *= 1 / (1 + np.exp(-(action - state)))
            state = next_state
        return total + .9**2 * c * (.3 * state)

    finite_difference = (objective(.2001) - objective(.1999)) / .0002
    original = wm.weight.detach().clone()
    stats = update_actor_critic(wm, actor, critic, starts, torch.optim.SGD(actor.parameters(), lr=.1),
                                torch.optim.SGD(critic.parameters(), lr=.1), horizon=2, gamma=.9)
    assert stats["actor_grad_norm"] == pytest.approx(finite_difference, rel=1e-4)
    assert stats["actor_loss"] == pytest.approx(-objective(.2), rel=1e-5)
    assert actor.bias > .2
    assert actor.bias.grad.abs() <= .100001
    assert wm.weight.requires_grad and wm.weight.grad is None
    torch.testing.assert_close(wm.weight, original)


def test_observation_history_and_cpu_policy_checkpoint(tmp_path):
    spec = deterministic_spec(SimConfig())
    actor = Actor(width=12, vertical_only=True)
    frames = np.zeros((9, 14), dtype=np.float32)
    frames[:, 6] = 1
    frames[:, 2] = [2, 1, 0, 0, 0, 1e-3, 0, 0, 0]
    policy = Policy(actor, spec)
    actual = np.stack([policy.state(obs) for obs in frames])
    np.testing.assert_array_equal(actual, episode_features(frames, spec))
    policy.reset()
    path = tmp_path / "policy.pt"
    torch.save(policy_checkpoint(actor, spec), path)
    loaded = load_policy(path, device="cpu")
    actions = np.array([policy(obs) for obs in frames])
    np.testing.assert_array_equal(actions, [loaded(obs) for obs in frames])
    assert ((actions[:, 0] >= 0) & (actions[:, 0] <= 1)).all()
    np.testing.assert_array_equal(actions[:, 1:], 0)
    loaded.reset()
    np.testing.assert_array_equal(actions[0], loaded(frames[0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA checkpoint")
def test_cuda_policy_checkpoint_loads_on_cpu(tmp_path):
    spec = deterministic_spec(SimConfig())
    actor = Actor(width=8).cuda()
    path = tmp_path / "cuda-policy.pt"
    torch.save(policy_checkpoint(actor, spec), path)
    policy = load_policy(path, device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6] = 1
    assert np.isfinite(policy(obs)).all()
    assert next(policy.actor.parameters()).device.type == "cpu"


def test_optional_critic_bound_preserves_negative_tail_and_checkpoint_keys():
    critic = Critic(width=8, upper_bound=100.)
    legacy = Critic(width=8)
    legacy.load_state_dict(critic.state_dict())
    state = torch.zeros(1, 18)
    with torch.no_grad():
        critic.network[-1].bias.fill_(-3.)
    assert critic(state).item() == pytest.approx(-300.)
    with torch.no_grad():
        critic.network[-1].bias.fill_(1.2)
    value = critic(state)
    assert value.item() < 100.
    value.backward()
    assert critic.network[-1].bias.grad.item() > 0


@pytest.mark.parametrize("vertical_first_column", [False, True])
def test_yaw_frame_changes_neither_actor_nor_critic(vertical_first_column):
    from mun_agent.imagination import yaw_canonical_state
    torch.manual_seed(73)
    state = torch.randn(10, 18)
    if vertical_first_column:
        state[:, 6:8] = 0
    angle = torch.linspace(-3., 3., len(state))[:, None]
    rotated = state.clone()
    for i in (0, 3, 6, 9):
        x, y = state[:, i:i + 1], state[:, i + 1:i + 2]
        rotated[:, i:i + 1] = x * angle.cos() - y * angle.sin()
        rotated[:, i + 1:i + 2] = x * angle.sin() + y * angle.cos()
    torch.testing.assert_close(yaw_canonical_state(rotated), yaw_canonical_state(state))
    actor = Actor(width=16, yaw_invariant=True)
    critic = Critic(width=16, yaw_invariant=True)
    nn.init.normal_(critic.network[-1].weight)
    for net in (actor, critic):
        torch.testing.assert_close(net(state), net(rotated), atol=1e-5, rtol=1e-5)


def test_score_function_uses_fixed_active_samples_and_ignores_post_abort(monkeypatch):
    class MeanOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.vertical_only = True
            self.bias = nn.Parameter(torch.zeros(4))

        def location(self, states):
            return self.bias.expand(*states.shape[:-1], 4)

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(1.))
            self.spec = {}
            self.actions = []

        def step(self, state, action):
            assert not torch.is_grad_enabled()
            assert not state.requires_grad and not action.requires_grad
            self.actions.append(action.clone())
            assert not action[..., 1:].any()
            return state, self.gain * action[:, 0], None

    monkeypatch.setattr(imagination, "task_reward", lambda s, a, ns, p, spec: (p, torch.zeros_like(p)))
    torch.manual_seed(12)
    actor, critic, model = MeanOnly(), Critic(width=8), RecordingModel()
    std = .3
    starts = torch.zeros(64, 18, requires_grad=True)
    stats = imagination.update_sampled_actor_critic(model, actor, critic, starts,
        torch.optim.SGD(actor.parameters(), lr=.1), torch.optim.SGD(critic.parameters(), lr=0),
        horizon=3, action_std=std)
    first = model.actions[0][:, 0]
    # All later actions are after the first rule abort, with zero score weight.
    standardized = (first - first.mean()) / first.std(unbiased=False)
    analytic = (standardized * first.atanh() / std**2).mean()
    assert stats["actor_grad_norm"] == pytest.approx(analytic.item(), rel=1e-5)
    assert stats["value_target_mean"] == pytest.approx(first.mean().item(), abs=1e-6)
    assert stats["end_survival"] == 0
    assert actor.bias[0] > 0 and not actor.bias[1:].any()
    assert model.gain.grad is None and starts.grad is None
