"""Observation-only callable for deterministic simulator policies."""
import numpy as np
import torch
from mun_lander import SimConfig

from .imagination import Actor
from .interface import preprocess, to_native_action, validate_deterministic_spec


class Policy:
    def __init__(self, actor, spec):
        validate_deterministic_spec(spec, SimConfig())
        self.actor = actor
        self.spec = spec
        self.reset()

    def reset(self):
        self.ground_steps = 0
        self.steps = 0

    def state(self, obs):
        """Consume one observed frame and return its causal 18-dimensional state."""
        physical = preprocess(obs, self.spec)
        if physical.shape != (16,):
            raise ValueError("policy expects one raw 14-dimensional observation")
        cap = self.spec["ground_history_cap"]
        if obs[2] < self.spec["ground_epsilon_m"]:
            self.ground_steps = min(cap, self.ground_steps + 1)
        else:
            self.ground_steps = 0
        clock_cap = self.spec["episode_step_cap"]
        history = np.array([self.ground_steps / cap, min(self.steps, clock_cap) / clock_cap],
                           dtype=np.float32)
        self.steps += 1
        return np.concatenate((physical, history))

    @torch.inference_mode()
    def __call__(self, obs):
        device = next(self.actor.parameters()).device
        state = torch.from_numpy(self.state(obs)).to(device)
        return to_native_action(self.actor(state).cpu().numpy())


def policy_checkpoint(actor, spec):
    return {"policy_schema_version": 1, "interface_spec": spec,
            "actor_spec": {"width": actor.width, "vertical_only": actor.vertical_only,
                           "yaw_invariant": actor.yaw_invariant},
            "actor": actor.state_dict()}


def load_policy(checkpoint_path, device="cpu"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("policy_schema_version") != 1:
        raise ValueError("unsupported policy checkpoint")
    actor = Actor(**saved["actor_spec"])
    actor.load_state_dict(saved["actor"])
    return Policy(actor.to(device).eval(), saved["interface_spec"])
