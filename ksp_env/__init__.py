"""Importable KSP environment adapter.

Diagnostics remain in ``scripts/``; this package is the library used by the
future task environment and training code.
"""

from .contracts import (
    Action,
    ConfigurationError,
    EnvConfig,
    EpisodeDoneError,
    InfrastructureError,
    KSPConnectionError,
    KSPEnvError,
    Observation,
    ResetError,
    ResetInfo,
    SnapshotError,
    SnapshotInfo,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)
from .env import KSPEnv
from .initial_states import InitialState, sample_bank, sample_initial_state

__all__ = [
    "Action",
    "ConfigurationError",
    "EnvConfig",
    "EpisodeDoneError",
    "InfrastructureError",
    "KSPConnectionError",
    "KSPEnv",
    "KSPEnvError",
    "InitialState",
    "Observation",
    "ResetError",
    "ResetInfo",
    "SnapshotError",
    "SnapshotInfo",
    "StepResult",
    "TickError",
    "VesselError",
    "is_tbd",
    "load_config",
    "require_defined",
    "require_number",
    "sample_bank",
    "sample_initial_state",
]
