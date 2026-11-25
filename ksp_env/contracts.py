"""Stable data contracts for the KSP environment adapter.

This module deliberately has no dependency on kRPC, NumPy or Gymnasium. The
adapter builds these objects from kRPC telemetry, and a future fast simulator
can build the same objects from its own state.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mun_landing.toml"
TBD = "TBD"

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


class KSPEnvError(Exception):
    """Base class for adapter errors."""


class ConfigurationError(KSPEnvError):
    """A required decision or value is missing, including any ``TBD`` field."""


class InfrastructureError(KSPEnvError):
    """The connection, reset or physics-tick transport failed.

    These must not be converted into a successful or failed landing.
    """


class KSPConnectionError(InfrastructureError):
    """The kRPC connection could not be established or was lost."""


class VesselError(InfrastructureError):
    """No usable active vessel exists in the current scene."""


class ResetError(InfrastructureError):
    """A named save could not be loaded and verified."""


class TickError(InfrastructureError):
    """A held physics tick did not complete as expected."""


class EpisodeDoneError(KSPEnvError):
    """``step()`` was called after a terminal or truncated episode."""


class SnapshotError(KSPEnvError):
    """A debug-generated initial save could not be created or verified."""


def is_tbd(value: object) -> bool:
    """Return whether a configuration value is the explicit ``TBD`` marker."""

    return isinstance(value, str) and value.strip().upper() == TBD


def require_defined(value: Any, name: str) -> Any:
    """Return a value or raise if it is absent or still marked ``TBD``."""

    if value is None or is_tbd(value):
        raise ConfigurationError(f"{name} is not defined (TBD)")
    return value


def require_number(value: Any, name: str) -> float:
    """Return a finite number or raise a clear configuration error."""

    value = require_defined(value, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be finite")
    return number


def _finite_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must be a sequence of {length} numbers")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        numbers.append(number)
    return tuple(numbers)


@dataclass(frozen=True)
class Observation:
    """Numeric observation defined by ``docs/task-spec.md``.

    ``position_m``, ``velocity_m_s``, ``attitude_quaternion`` and
    ``angular_velocity_rad_s`` are expressed in the target-relative surface
    frame once that frame is fixed.  kRPC returns quaternions as
    ``(x, y, z, w)``.

    ``main_engine_fuel`` is the configured main-engine resource amount.  Its
    physical unit and normalization remain ``TBD`` until the lander is fixed.
    """

    position_m: Vector3
    velocity_m_s: Vector3
    attitude_quaternion: Quaternion
    angular_velocity_rad_s: Vector3
    main_engine_fuel: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _finite_tuple(self.position_m, 3, "position_m"))
        object.__setattr__(self, "velocity_m_s", _finite_tuple(self.velocity_m_s, 3, "velocity_m_s"))
        object.__setattr__(
            self, "attitude_quaternion", _finite_tuple(self.attitude_quaternion, 4, "attitude_quaternion")
        )
        object.__setattr__(
            self, "angular_velocity_rad_s", _finite_tuple(self.angular_velocity_rad_s, 3, "angular_velocity_rad_s")
        )
        if isinstance(self.main_engine_fuel, bool) or not isinstance(self.main_engine_fuel, (int, float)):
            raise ValueError("main_engine_fuel must be a number")
        fuel = float(self.main_engine_fuel)
        if not math.isfinite(fuel) or fuel < 0:
            raise ValueError("main_engine_fuel must be finite and nonnegative")
        object.__setattr__(self, "main_engine_fuel", fuel)
        norm = math.sqrt(sum(component * component for component in self.attitude_quaternion))
        if norm == 0:
            raise ValueError("attitude_quaternion must not be the zero quaternion")


@dataclass(frozen=True)
class Action:
    """Normalized control command.

    RCS values use kRPC's normalized input convention in ``[-1, 1]`` and
    throttle uses ``[0, 1]``.  Whether those inputs represent a torque command
    or a normalized rate remains a task decision; :meth:`KSPEnv.step` refuses to
    run until ``[action].rcs_semantics`` is set.
    """

    rcs_pitch: float = 0.0
    rcs_yaw: float = 0.0
    rcs_roll: float = 0.0
    main_engine_throttle: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rcs_pitch", "rcs_yaw", "rcs_roll"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between -1 and 1")
            object.__setattr__(self, name, value)

        throttle = self.main_engine_throttle
        if isinstance(throttle, bool) or not isinstance(throttle, (int, float)):
            raise ValueError("main_engine_throttle must be a number")
        throttle = float(throttle)
        if not math.isfinite(throttle) or not 0.0 <= throttle <= 1.0:
            raise ValueError("main_engine_throttle must be finite and between 0 and 1")
        object.__setattr__(self, "main_engine_throttle", throttle)


@dataclass(frozen=True)
class StepResult:
    """One adapter step over a fixed control interval.

    ``terminated`` and ``truncated`` follow Gymnasium semantics.  The adapter
    evaluates a provisional contact/stability outcome, but ``reward`` remains
    ``None`` because the task contract does not define a reward yet.
    """

    observation: Observation
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
    reward: float | None = None


@dataclass(frozen=True)
class ResetInfo:
    """Metadata for a successful named-save reset."""

    save_name: str
    ut_s: float
    game_state_loads: int
    scene: str
    vessel_name: str
    elapsed_s: float | None = None


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata for a debug-generated initial save."""

    save_name: str
    ut_s: float
    game_state_loads: int
    vessel_name: str
    body_name: str
    target_position: tuple[float, float, float]
    requested: Mapping[str, Any]
    observed: Mapping[str, Any] | None
    can_save_before: bool
    can_save_after: bool
    verified: bool
    save_path: str | None = None
    save_sha256: str | None = None


@dataclass(frozen=True)
class EnvConfig:
    """Connection settings plus the raw task configuration.

    The raw mapping keeps ``TBD`` values intact.  Callers must use
    :func:`require_defined`/:func:`require_number` before relying on a value
    that is still open in the task contract.
    """

    address: str
    rpc_port: int
    stream_port: int
    client_version: str
    server_version: str
    game_version: str
    timeout_s: float = 10.0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


def load_config(path: str | Path | None = None) -> EnvConfig:
    """Load ``configs/mun_landing.toml`` and validate its connection block."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"could not read configuration: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {config_path}") from exc

    connection = raw.get("connection")
    if not isinstance(connection, dict):
        raise ConfigurationError("configuration is missing the [connection] section")

    address = connection.get("address")
    if not isinstance(address, str) or not address.strip():
        raise ConfigurationError("[connection].address must be a non-empty string")

    ports = {}
    for key in ("rpc_port", "stream_port"):
        value = connection.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ConfigurationError(f"[connection].{key} must be an integer between 1 and 65535")
        ports[key] = value

    versions = {}
    for key in ("client_version", "server_version", "game_version"):
        value = connection.get(key)
        if not isinstance(value, str) or not value.strip() or is_tbd(value):
            raise ConfigurationError(f"[connection].{key} must be defined")
        versions[key] = value

    return EnvConfig(
        address=address,
        rpc_port=ports["rpc_port"],
        stream_port=ports["stream_port"],
        client_version=versions["client_version"],
        server_version=versions["server_version"],
        game_version=versions["game_version"],
        raw=raw,
    )
