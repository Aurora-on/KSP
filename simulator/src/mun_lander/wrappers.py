"""Explicit opt-in learning transformations. Raw SI API remains the default."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NormalizedObservation(gym.ObservationWrapper):
    def __init__(self, env, scale=None):
        super().__init__(env)
        self.scale = np.asarray(scale if scale is not None else [100]*3+[10]*3+[1]*4+[0.5]*3+[env.unwrapped.config.vehicle.fuel_capacity], dtype=np.float32)
        if self.scale.shape != (14,) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("scale must be 14 finite positive values")
        self.observation_space = spaces.Box(env.observation_space.low/self.scale, env.observation_space.high/self.scale, dtype=np.float32)

    def observation(self, observation):
        return (observation/self.scale).astype(np.float32)


class SymmetricAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def action(self, action):
        a = np.asarray(action, dtype=np.float64).copy()
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError("action must be a finite vector of shape (4,)")
        a[0] = (a[0]+1)*0.5
        return a

