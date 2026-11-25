"""Categorical recurrent state-space model for the 16 normalized lander features."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def symlog(x: Tensor) -> Tensor:
    return x.sign() * x.abs().log1p()


def symexp(x: Tensor) -> Tensor:
    return x.sign() * x.abs().expm1()


def two_hot(target: Tensor, centers: Tensor) -> Tensor:
    """Interpolate between adjacent ordered bin centers in original reward units."""
    value = target.to(centers.dtype).clamp(centers[0], centers[-1])
    upper = torch.searchsorted(centers, value.contiguous(), right=True).clamp(1, len(centers) - 1)
    lower = upper - 1
    width = centers[upper] - centers[lower]
    lower_weight = (centers[upper] - value) / width
    upper_weight = (value - centers[lower]) / width
    probs = value.new_zeros((*value.shape, len(centers)))
    probs.scatter_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    return probs.scatter_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))


def decode_two_hot(probs: Tensor, centers: Tensor) -> Tensor:
    """Expectation on the symmetric symexp grid, in original reward units."""
    # Subtract masses at opposite centers before multiplying by their shared
    # magnitude. Equal tail masses cancel exactly even at ±4.85e8 in float32.
    half = len(centers) // 2
    difference = probs[..., -half:] - probs[..., :half].flip((-1,))
    return (difference * centers[-half:]).sum(dim=-1)


def kl_losses(posterior: Tensor, prior: Tensor, free_nats: float = 1.0):
    """Per-state KL terms for factored categorical probability tensors [...,S,C]."""
    log_q, log_p = posterior.log(), prior.log()
    dynamics = (posterior.detach() * (log_q.detach() - log_p)).sum(dim=(-1, -2))
    representation = (posterior * (log_q - log_p.detach())).sum(dim=(-1, -2))
    return dynamics.clamp_min(free_nats), representation.clamp_min(free_nats)


def _mlp(input_size, hidden, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, output_size),
    )


class WorldModel(nn.Module):
    def __init__(self, deter: int, stoch: int, classes: int, hidden: int, bins: int = 255,
                 *, legacy_reward: bool = False, dynamics_scale: float = 1.0,
                 representation_scale: float = 0.1, free_nats: float = 1.0,
                 event_weight: float = 1.0, ground_history: bool = False):
        super().__init__()
        if min(deter, stoch, hidden) < 1 or min(classes, bins) < 2:
            raise ValueError("invalid world model dimensions")
        scales = (dynamics_scale, representation_scale, free_nats, event_weight)
        if any(not math.isfinite(k) or k < 0 for k in scales) or event_weight <= 0:
            raise ValueError("loss scales must be finite and nonnegative, event_weight positive")
        self.dynamics_scale = dynamics_scale
        self.representation_scale = representation_scale
        self.free_nats = free_nats
        self.event_weight = event_weight
        self.ground_history = ground_history
        self.input_size = 17 if ground_history else 16
        self.deter, self.stoch, self.classes = deter, stoch, classes
        stochastic_size = stoch * classes
        state_size = deter + stochastic_size
        self.encoder = _mlp(self.input_size, hidden, hidden)
        self.gru = nn.GRUCell(stochastic_size + 4, deter)
        self.prior = _mlp(deter, hidden, stochastic_size)
        self.posterior = _mlp(deter + hidden, hidden, stochastic_size)
        self.decoder = _mlp(state_size, hidden, 16)
        self.reward_head = _mlp(state_size, hidden, bins)
        self.continue_head = _mlp(state_size, hidden, 1)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)
        self.legacy_reward = legacy_reward
        if legacy_reward:
            # Preserve the original key and arithmetic for format-1 checkpoints.
            self.register_buffer("centers", torch.linspace(-20, 20, bins))
        else:
            # Mirror the negative half exactly, including the two central zero
            # bins for an even-sized head in the reference implementation.
            negative = symexp(torch.linspace(-20, 0, (bins + 1) // 2))
            mirrored = negative[:-1] if bins % 2 else negative
            self.register_buffer("reward_bins", torch.cat((negative, -mirrored.flip((0,)))))

    def initial(self, batch_size):
        return (self.gru.bias_hh.new_zeros((batch_size, self.deter)),
                self.gru.bias_hh.new_zeros((batch_size, self.stoch, self.classes)))

    def _distribution(self, logits):
        probs = logits.unflatten(-1, (self.stoch, self.classes)).softmax(dim=-1)
        return 0.99 * probs + 0.01 / self.classes

    def _one_hot(self, probs, sample):
        if sample:
            index = torch.multinomial(probs.reshape(-1, self.classes), 1)
            index = index.reshape(probs.shape[:-1])
        else:
            index = probs.argmax(dim=-1)
        hard = F.one_hot(index, self.classes).to(probs.dtype)
        return hard + (probs - probs.detach())

    def _posterior_probs(self, h, embedding):
        return self._distribution(self.posterior(torch.cat((h, embedding), dim=-1)))

    def _advance(self, state, action):
        h, z = state
        return self.gru(torch.cat((z.flatten(-2), action), dim=-1), h)

    def observe_initial(self, features, sample=False):
        h, _ = self.initial(len(features))
        embedding = self.encoder(symlog(features))
        return h, self._one_hot(self._posterior_probs(h, embedding), sample)

    def observe_step(self, state, action_internal, next_features, sample=False):
        h = self._advance(state, action_internal)
        embedding = self.encoder(symlog(next_features))
        return h, self._one_hot(self._posterior_probs(h, embedding), sample)

    def imagine_step(self, state, action_internal, sample=False):
        h = self._advance(state, action_internal)
        return h, self._one_hot(self._distribution(self.prior(h)), sample)

    def _heads(self, state):
        h, z = state
        latent = torch.cat((h, z.flatten(-2)), dim=-1)
        return (self.decoder(latent), self.reward_head(latent),
                self.continue_head(latent).squeeze(-1))

    def predict(self, state):
        features, reward_logits, continue_logits = self._heads(state)
        probs = reward_logits.softmax(dim=-1)
        if self.legacy_reward:
            reward = symexp((probs * self.centers).sum(dim=-1))
        else:
            reward = decode_two_hot(probs, self.reward_bins)
        return symexp(features), reward, continue_logits.sigmoid()

    def loss(self, features, actions, rewards, terminals, mask, *, sample=True):
        """At a real transition t predict x[t+1], r[t] and 1-done[t] from s[t+1]."""
        if self.legacy_reward:
            raise RuntimeError("historical reward head supports inference only")
        batch, length = mask.shape
        if (features.shape != (batch, length + 1, self.input_size) or actions.shape != (batch, length, 4)
                or rewards.shape != mask.shape or terminals.shape != mask.shape):
            raise ValueError("inconsistent world model sequence shapes")

        # Replay guarantees one contiguous, nonempty valid span per row. Move it
        # to the left before recurrent inference; no padded prefix can consume
        # actions, alter the hidden state, or draw samples before the first input.
        mask = mask.bool()
        counts = mask.sum(dim=1)
        first = mask.long().argmax(dim=1)
        offset = torch.arange(length, device=mask.device)
        relative = offset - first[:, None]
        if not torch.equal(mask, (relative >= 0) & (relative < counts[:, None])) or (counts == 0).any():
            raise ValueError("each sampled row must have one nonempty contiguous valid span")
        valid = offset < counts[:, None]
        row = torch.arange(batch, device=mask.device)
        index = first[:, None] + offset.clamp_max((counts - 1)[:, None])
        source_obs = features[row, first]
        next_obs = features[row[:, None], index + 1]
        act = actions[row[:, None], index]
        embeddings = self.encoder(symlog(next_obs))
        state = self.observe_initial(source_obs, sample)

        hs, zs, posteriors = [], [], []
        for i in range(length):
            h = self._advance(state, act[:, i])
            q = self._posterior_probs(h, embeddings[:, i])
            z = self._one_hot(q, sample)
            state = h, z
            hs.append(h)
            zs.append(z)
            posteriors.append(q)

        h = torch.stack(hs, dim=1)[valid]
        z = torch.stack(zs, dim=1)[valid]
        q = torch.stack(posteriors, dim=1)[valid]
        prior = self._distribution(self.prior(h))
        dynamics, representation = kl_losses(q, prior, self.free_nats)
        features_hat, reward_logits, continue_logits = self._heads((h, z))
        target_reward = rewards[row[:, None], index][valid]
        target_continue = 1.0 - terminals[row[:, None], index][valid].to(features.dtype)
        target_probs = two_hot(target_reward, self.reward_bins)
        # Optional event weighting changes the effective training distribution
        # of these two heads. Its inference still uses only the RSSM state.
        weights = target_continue + (1 - target_continue) * self.event_weight
        weights = weights / weights.sum()
        reward_ce = -(target_probs * reward_logits.log_softmax(dim=-1)).sum(dim=-1)
        continue_bce = F.binary_cross_entropy_with_logits(
            continue_logits, target_continue, reduction="none")
        # Reconstruct the 16 physical features. The optional causal contact
        # history is additional encoder context, not a reconstruction target.
        error = features_hat - symlog(next_obs[valid][..., :16])
        losses = {
            "reconstruction": error.square().sum(dim=-1).mean(),
            "reward": (weights * reward_ce).sum(),
            "continue": (weights * continue_bce).sum(),
            "dynamics": dynamics.mean(),
            "representation": representation.mean(),
        }
        losses["loss"] = (losses["reconstruction"] + losses["reward"] + losses["continue"]
                          + self.dynamics_scale * losses["dynamics"]
                          + self.representation_scale * losses["representation"])
        return losses
