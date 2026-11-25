"""Differentiable residual dynamics and event probabilities in observation coordinates."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mun_lander import SimConfig

from .interface import validate_deterministic_spec
from .task_reward import advance_history


def mlp(input_dim, width, layers, output_dim):
    modules = []
    for _ in range(layers):
        modules.extend([nn.Linear(input_dim, width), nn.SiLU()])
        input_dim = width
    return nn.Sequential(*modules, nn.Linear(width, output_dim))


class DeterministicModel(nn.Module):
    def __init__(self, spec, *, width=256, event_weight=40.0, kinematic=False, contact_margin=False):
        super().__init__()
        self.spec = spec
        self.kinematic = kinematic
        self.contact_margin = contact_margin
        if contact_margin and not kinematic:
            raise ValueError("contact margins require the fitted normal acceleration head")
        if kinematic:
            self.liftoff = nn.Linear(3, 1, bias=False)
        self.dynamics = mlp(24, width, 3, 16)
        head_inputs = 51 if contact_margin else 46
        self.event = mlp(head_inputs, 64, 2, 3)
        self.diagnostic_reward = mlp(head_inputs, 64, 2, 1)
        self.register_buffer("delta_std", torch.ones(16))
        self.register_buffer("class_weights", torch.tensor([1.0, event_weight, event_weight]))

    def contact_features(self, state):
        """Smooth proximity and linearized clearance in units of vertical travel per step."""
        height = state[..., 2] * self.spec["scales"]["position"]
        travel = state[..., 5] * self.spec["scales"]["velocity"] * self.spec["control_dt"]
        clearance = (height + travel) / (0.01 + travel.abs())
        return torch.stack((torch.exp(-height.clamp_min(0) / 0.2), clearance.clamp(-3, 3)), -1)

    @staticmethod
    def lift_inputs(state, action):
        # The two thrust terms allow a first-order dependence on remaining fuel.
        up = torch.linalg.cross(state[..., 6:9], state[..., 9:12])
        vertical_throttle = (action[..., 0] + 1) * 0.5 * up[..., 2]
        return torch.stack((torch.ones_like(vertical_throttle), vertical_throttle,
                            vertical_throttle * (state[..., 15] - 1)), -1)

    def initialize_liftoff(self, state, action, following):
        # Learn the positive branch from observed liftoff velocities. This
        # avoids the flat gradient of a rectified head initialized below zero.
        liftoff = (state[..., 16] > 0) & (following[..., 2] > 0)
        if liftoff.sum() >= 3:
            x = self.lift_inputs(state[liftoff], action[liftoff]).cpu().double()
            y = following[liftoff, 5:6].cpu().double()
            weights = torch.linalg.lstsq(x, y, driver="gelsd").solution.T
            with torch.no_grad():
                self.liftoff.weight.copy_(weights)

    def advance(self, state, action):
        inputs = torch.cat((state, action, self.contact_features(state)), -1)
        delta = self.delta_std * self.dynamics(inputs)
        physical = state[..., :16] + delta
        if self.kinematic:
            grounded = state[..., 16:17] > 0
            free_vz = physical[..., 5:6]
            supported_vz = self.liftoff(self.lift_inputs(state, action)).clamp_min(0)
            vz = torch.where(grounded, supported_vz, free_vz)
            velocity = torch.cat((physical[..., 3:5], vz), -1)
            # The trapezoid uses velocity and position in the model's units.
            # A learned displacement corrects impact and partial terminal steps.
            factor = self.spec["control_dt"] * self.spec["scales"]["velocity"] / self.spec["scales"]["position"]
            trapezoid = (state[..., 3:6] + velocity) * (0.5 * factor)
            position = state[..., :3] + trapezoid + 0.05 * delta[..., :3]
            z = torch.where(grounded, supported_vz * (0.5 * factor), position[..., 2:3])
            physical = torch.cat((position[..., :2], z, velocity, physical[..., 6:]), -1)
        # Only height has a geometric floor. Positive predicted displacements allow liftoff.
        physical = torch.cat((physical[..., :2], physical[..., 2:3].clamp_min(0), physical[..., 3:]), -1)
        return advance_history(state, physical, self.spec)

    def motion_margin(self, state):
        limits = self.spec["source_config"]["landing"]
        speed = state[..., 3:6].norm(dim=-1) * self.spec["scales"]["velocity"]
        spin = state[..., 12:15].norm(dim=-1) * self.spec["scales"]["angular_velocity"]
        up = torch.linalg.cross(state[..., 6:9], state[..., 9:12])
        z_axis = torch.where((up == 0).all(-1), 1.0, up[..., 2])
        tilt = torch.atan2(up[..., :2].norm(dim=-1), z_axis)
        ratios = torch.stack((speed / limits["max_touchdown_speed"],
                              spin / limits["max_angular_speed"],
                              tilt / math.radians(limits["max_tilt_deg"])), -1)
        return ratios.amax(-1) - 1.0

    def contact_margins(self, state, action, following):
        # Approximate the phase of first contact within a control interval.
        # Ratios supplied to the learned logits are continuous across impact.
        scales = self.spec["scales"]
        height = state[..., 2] * scales["position"]
        change_vz = self.liftoff(self.lift_inputs(state, action)).squeeze(-1)
        travel = (state[..., 5] + 0.5 * change_vz) * scales["velocity"] * self.spec["control_dt"]
        crossing = -(height + travel) / (0.01 + travel.abs())
        current = self.motion_margin(state)
        next_margin = self.motion_margin(following)
        joint = torch.minimum(crossing, torch.maximum(current, next_margin))
        # Spread the narrow boundary in input coordinates; the MLP still fits
        # the conditional distribution over all three outcomes by cross entropy.
        return (8 * torch.stack((crossing, current, next_margin, joint,
                                 state[..., 16] - 1), -1)).clamp(-8, 8)

    def heads(self, state, action, following):
        cap = self.spec["ground_history_cap"]
        countdown = ((1 - state[..., 16]) * cap).clamp(0, 5)
        speed = state[..., 3:6].norm(dim=-1) * self.spec["scales"]["velocity"]
        spin = state[..., 12:15].norm(dim=-1) * self.spec["scales"]["angular_velocity"]
        axis = torch.linalg.cross(state[..., 6:9], state[..., 9:12])
        sin_tilt = axis[..., :2].norm(dim=-1)
        inputs = torch.cat((state, action, following, self.contact_features(state),
                            torch.stack((countdown, speed, spin, sin_tilt), -1)), -1)
        if self.contact_margin:
            inputs = torch.cat((inputs, self.contact_margins(state, action, following)), -1)
        logits = self.event(inputs)
        reward = self.diagnostic_reward(inputs.detach()).squeeze(-1)
        return logits, reward

    def step(self, state, action):
        following = self.advance(state, action)
        logits, reward = self.heads(state, action, following)
        probs = (logits - self.class_weights.log()).softmax(-1)
        return following, probs, decode_reward(reward)

    def losses(self, initial, actions, targets, labels, rewards, mask):
        """Unroll only through live transitions. Padded leading steps retain the real seed."""
        state = initial
        starts, predictions = [], []
        for t in range(actions.shape[1]):
            starts.append(state)
            state = torch.where(mask[:, t, None], self.advance(state, actions[:, t]), state)
            predictions.append(state)
        predictions = torch.stack(predictions, 1)
        logits, reward_hat = self.heads(torch.stack(starts, 1), actions, predictions)
        error = (predictions[..., :16] - targets[..., :16]) / self.delta_std
        # Sum the eight transition losses and average over windows, with boundary masks.
        denom = mask.shape[0]
        dyn_loss = (mask * error.square().mean(-1)).sum() / denom
        cross_entropy = F.cross_entropy(logits.transpose(1, 2), labels, weight=self.class_weights,
                                        ignore_index=-1, reduction="none")
        event_loss = (cross_entropy * mask).sum() / denom
        # Predict symlog reward to balance dense and terminal magnitudes.
        symlog_reward = rewards.sign() * rewards.abs().log1p()
        reward_weights = torch.where(labels > 0, self.class_weights[1], 1.0)
        reward_loss = ((reward_hat - symlog_reward).square() * mask * reward_weights).sum() / denom
        return dyn_loss, event_loss, reward_loss


def decode_reward(symlog):
    return symlog.sign() * symlog.abs().expm1()


def load_deterministic(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "deterministic" or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported deterministic checkpoint")
    spec = checkpoint["interface_spec"]
    validate_deterministic_spec(spec, SimConfig())
    settings = checkpoint["config"]
    model = DeterministicModel(spec, width=settings["width"], event_weight=settings["event_weight"],
                               kinematic=settings.get("kinematic", False),
                               contact_margin=settings.get("contact_margin", False))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
