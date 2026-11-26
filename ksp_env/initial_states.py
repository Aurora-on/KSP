"""Pure samplers for the frozen KSP descent initial-state distribution.

The sampler uses the ranges in ``configs/mun_landing.toml``.  It only creates
target-frame requests; applying them to KSP and recording the observed state is
the live bank generator's job.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import ConfigurationError, is_tbd
from .geometry import Vector3

@dataclass(frozen=True)
class InitialState:
    """A requested initial state in the target frame.

    ``position_m`` is ``(east, south, up)`` relative to the target surface.
    ``body_z_direction`` is the requested task body +z axis in the target frame.
    """

    clearance_m: float
    position_m: Vector3
    velocity_m_s: Vector3
    body_z_direction: Vector3
    roll_deg: float
    angular_velocity_rad_s: Vector3

    def __post_init__(self) -> None:
        for name in (
            "position_m",
            "velocity_m_s",
            "body_z_direction",
            "angular_velocity_rad_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ValueError(f"{name} must contain three numbers")
            numbers = tuple(float(component) for component in value)
            if not all(math.isfinite(component) for component in numbers):
                raise ValueError(f"{name} must contain finite numbers")
            object.__setattr__(self, name, numbers)
        if not math.isfinite(self.clearance_m) or self.clearance_m <= 0.0:
            raise ValueError("clearance_m must be finite and positive")
        if not math.isfinite(self.roll_deg):
            raise ValueError("roll_deg must be finite")


def _pair(value: Any, name: str) -> tuple[float, float]:
    if is_tbd(value) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigurationError(f"{name} must contain two numbers")
    low, high = float(value[0]), float(value[1])
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ConfigurationError(f"{name} must be a finite ordered range")
    return low, high


def _number(value: Any, name: str) -> float:
    if is_tbd(value) or not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be finite")
    return number


def _unit_sphere(rng: random.Random) -> tuple[float, float, float]:
    z = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return (radius * math.cos(azimuth), radius * math.sin(azimuth), z)


def _cone_direction(rng: random.Random, max_tilt_deg: float) -> tuple[float, float, float]:
    """Sample a thrust axis uniformly in solid angle inside a cone."""

    max_tilt = math.radians(max_tilt_deg)
    cos_theta = rng.uniform(math.cos(max_tilt), 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    return (
        sin_theta * math.cos(azimuth),
        sin_theta * math.sin(azimuth),
        cos_theta,
    )


def sample_initial_state(
    rng: random.Random,
    scenario: Mapping[str, Any],
) -> InitialState:
    """Draw one state from the frozen task distribution.

    The ranges are read from the ``[scenario]`` mapping.  No target values or
    observed KSP state are involved.
    """

    descent = scenario.get("descent_altitude_m")
    clearance = rng.uniform(*_pair(descent, "[scenario].descent_altitude_m"))

    position_range = scenario.get("initial_position_range")
    if not isinstance(position_range, Mapping):
        raise ConfigurationError("[scenario].initial_position_range must be a table")
    x_range = _pair(position_range.get("x_m"), "[scenario].initial_position_range.x_m")
    y_range = _pair(position_range.get("y_m"), "[scenario].initial_position_range.y_m")
    position = (
        rng.uniform(*x_range),
        rng.uniform(*y_range),
        clearance,
    )

    velocity_range = scenario.get("initial_velocity_range")
    if not isinstance(velocity_range, Mapping):
        raise ConfigurationError("[scenario].initial_velocity_range must be a table")
    vx_range = _pair(velocity_range.get("x_m_s"), "[scenario].initial_velocity_range.x_m_s")
    vy_range = _pair(velocity_range.get("y_m_s"), "[scenario].initial_velocity_range.y_m_s")
    vz_range = _pair(velocity_range.get("z_m_s"), "[scenario].initial_velocity_range.z_m_s")
    velocity = (
        rng.uniform(*vx_range),
        rng.uniform(*vy_range),
        rng.uniform(*vz_range),
    )

    attitude_range = scenario.get("initial_attitude_range")
    if not isinstance(attitude_range, Mapping):
        raise ConfigurationError("[scenario].initial_attitude_range must be a table")
    max_tilt_deg = _number(
        attitude_range.get("max_tilt_deg"),
        "[scenario].initial_attitude_range.max_tilt_deg",
    )
    body_z = _cone_direction(rng, max_tilt_deg)
    roll_deg = rng.uniform(0.0, 360.0)

    angular_range = scenario.get("initial_angular_velocity_range")
    if not isinstance(angular_range, Mapping):
        raise ConfigurationError(
            "[scenario].initial_angular_velocity_range must be a table"
        )
    max_norm = _number(
        angular_range.get("max_norm_rad_s"),
        "[scenario].initial_angular_velocity_range.max_norm_rad_s",
    )
    direction = _unit_sphere(rng)
    radius = max_norm * rng.random() ** (1.0 / 3.0)
    angular = tuple(radius * component for component in direction)

    return InitialState(
        clearance_m=clearance,
        position_m=position,
        velocity_m_s=velocity,
        body_z_direction=body_z,
        roll_deg=roll_deg,
        angular_velocity_rad_s=angular,
    )


def sample_bank(
    scenario: Mapping[str, Any],
    *,
    seed: int,
    count: int,
) -> list[InitialState]:
    """Sample a deterministic list of states from one seed."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return [sample_initial_state(rng, scenario) for _ in range(count)]
