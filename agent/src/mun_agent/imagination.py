"""Pathwise actor gradients and lambda-return value fitting on frozen model rollouts."""
from contextlib import contextmanager
import math

import torch
from torch import nn

from .deterministic import mlp
from .task_reward import task_reward


class Actor(nn.Module):
    """Deterministic action in the agent's symmetric action coordinates."""
    def __init__(self, width=128, vertical_only=True):
        super().__init__()
        self.width = width
        self.vertical_only = vertical_only
        self.network = mlp(18, width, 2, 4)
        nn.init.uniform_(self.network[-1].weight, -.003, .003)
        nn.init.zeros_(self.network[-1].bias)

    def project(self, action):
        if self.vertical_only:
            return torch.cat((action[..., :1], torch.zeros_like(action[..., 1:])), dim=-1)
        return action

    def forward(self, state):
        return self.project(self.network(state).tanh())


class Critic(nn.Module):
    """Return values in reward units, with a normalized regression coordinate."""
    def __init__(self, width=128, value_scale=100.0, upper_bound=None):
        super().__init__()
        if not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("value_scale must be positive and finite")
        if upper_bound is not None and (not math.isfinite(upper_bound) or upper_bound <= 0):
            raise ValueError("upper_bound must be positive and finite")
        self.upper_bound = upper_bound
        self.value_scale = float(value_scale)
        self.network = mlp(18, width, 2, 1)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state):
        value = self.value_scale * self.network(state).squeeze(-1)
        if self.upper_bound is None:
            return value
        # Smooth upper envelope: values far below it remain in original reward
        # units and retain unit slope, including arbitrarily negative outcomes.
        temperature = 0.1 * self.value_scale
        return self.upper_bound - temperature * nn.functional.softplus((self.upper_bound - value) / temperature)


@contextmanager
def frozen_parameters(module):
    """Preserve the input Jacobian while excluding model weights from the gradient tape."""
    params = tuple(module.parameters())
    flags = [p.requires_grad for p in params]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def lambda_returns(rewards, continuations, values, gamma=.997, lam=.95):
    """Time-major tensors: rewards/continuations[H,B], values[H+1,B]."""
    if (rewards.shape != continuations.shape or values.shape != (len(rewards) + 1, *rewards.shape[1:])
            or not len(rewards)):
        raise ValueError("lambda-return needs H rewards and continuations and H+1 values")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must lie in [0, 1]")
    last = values[-1]
    reversed_returns = []
    for t in reversed(range(len(rewards))):
        last = rewards[t] + gamma * continuations[t] * ((1 - lam) * values[t + 1] + lam * last)
        reversed_returns.append(last)
    return torch.stack(reversed_returns[::-1])


def update_actor_critic(model, actor, critic, starts, actor_optimizer, critic_optimizer, *,
                        horizon=32, gamma=.997, lam=.95, critic_epochs=2):
    """One BPTT actor update and value regression on the same detached, fixed targets."""
    if (type(horizon) is not int or horizon < 1 or type(critic_epochs) is not int or critic_epochs < 1
            or starts.ndim != 2 or starts.shape[1] != 18 or not len(starts)):
        raise ValueError("positive horizon, critic_epochs and a batch of 18-D states are required")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must lie in [0, 1]")
    state = starts.detach()
    survival = torch.ones_like(state[:, 0])
    states = [state]
    rewards, continues, weights = [], [], []
    with frozen_parameters(model), frozen_parameters(critic):
        trajectory_return = torch.zeros_like(survival)
        for t in range(horizon):
            action = actor(state)
            following, probs, _ = model.step(state, action)
            reward, continuation = task_reward(state, action, following, probs, model.spec)
            trajectory_return = trajectory_return + gamma**t * survival * reward
            weights.append(survival.detach())
            rewards.append(reward.detach())
            continues.append(continuation.detach())
            # Exact zero survival occurs after a rule abort. Keep those
            # rows at a bounded state; their subsequent terms carry zero weight.
            state = torch.where((survival > 0).unsqueeze(-1), following, state)
            survival = survival * continuation
            states.append(state.detach())

        objective = (trajectory_return + gamma**horizon * survival * critic(state)).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        nn.utils.clip_grad_value_(actor.parameters(), .1)
        actor_optimizer.step()

    visited = torch.stack(states)
    weights = torch.stack(weights)
    with torch.no_grad():
        targets = lambda_returns(torch.stack(rewards), torch.stack(continues), critic(visited), gamma, lam)
    for _ in range(critic_epochs):
        values = critic(visited[:-1])
        errors = (values - targets) / critic.value_scale
        critic_loss = (weights * errors.square()).sum() / weights.sum().clamp_min(1)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
        critic_optimizer.step()
    return {
        "actor_loss": -objective.detach().item(), "critic_loss": critic_loss.detach().item(),
        "actor_grad_norm": actor_norm.item(), "critic_grad_norm": critic_norm.item(),
        "imagined_return": trajectory_return.detach().mean().item(),
        "end_survival": survival.detach().mean().item(),
        "value_target_mean": (weights * targets).sum().item() / weights.sum().item(),
    }
