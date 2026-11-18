"""Importable KSP environment adapter.

Diagnostics remain in ``scripts/``; this package is the library used by the
future task environment and training code.
"""

from .contracts import (
    Action,
    ConfigurationError,
    EnvConfig,
    InfrastructureError,
    KSPConnectionError,
    KSPEnvError,
    Observation,
    ResetError,
    ResetInfo,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)
from .env import KSPEnv

__all__ = [
    "Action",
    "ConfigurationError",
    "EnvConfig",
    "InfrastructureError",
    "KSPConnectionError",
    "KSPEnv",
    "KSPEnvError",
    "Observation",
    "ResetError",
    "ResetInfo",
    "StepResult",
    "TickError",
    "VesselError",
    "is_tbd",
    "load_config",
    "require_defined",
    "require_number",
]
