"""MunLander-v0: six degree of freedom source-domain landing environment."""
from gymnasium.envs.registration import register, registry
from .config import SimConfig, VehicleConfig, InitialConfig, LandingConfig, RewardConfig
from .state import LanderState
from .env import LanderEnv
from .wrappers import NormalizedObservation, SymmetricAction

if "MunLander-v0" not in registry:
    register(id="MunLander-v0", entry_point="mun_lander.env:LanderEnv")

__all__ = ["LanderEnv", "LanderState", "SimConfig", "VehicleConfig", "InitialConfig",
           "LandingConfig", "RewardConfig", "NormalizedObservation", "SymmetricAction"]
