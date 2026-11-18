"""Validated, serializable experiment settings. Defaults are pilot assumptions."""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path


def _positive(name, value, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def _triple(name, values):
    if len(values) != 3:
        raise ValueError(f"{name} must have three components")
    for value in values:
        _positive(name, value)


@dataclass(frozen=True)
class VehicleConfig:
    gravity: float = 1.63
    dry_mass: float = 1200.0
    fuel_capacity: float = 600.0
    max_thrust: float = 9000.0
    main_isp: float = 310.0
    inertia: tuple = (1200.0, 1200.0, 800.0)
    rcs_torque: tuple = (900.0, 900.0, 400.0)
    rcs_isp: float = 240.0
    rcs_lever_arm: float = 1.4

    def __post_init__(self):
        for name in ("dry_mass", "fuel_capacity", "max_thrust", "main_isp", "rcs_isp", "rcs_lever_arm"):
            _positive(name, getattr(self, name))
        _positive("gravity", self.gravity, allow_zero=True)
        for name in ("inertia", "rcs_torque"):
            _triple(name, getattr(self, name))
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if 2 * max(self.inertia) > sum(self.inertia) + 1e-10:
            raise ValueError("principal moments must obey the rigid-body triangle inequality")


@dataclass(frozen=True)
class InitialConfig:
    horizontal_position: float = 30.0
    altitude: tuple = (80.0, 120.0)
    horizontal_velocity: float = 2.0
    vertical_velocity: tuple = (-8.0, -3.0)
    max_tilt_deg: float = 12.0
    max_angular_speed: float = 0.05

    def __post_init__(self):
        for name in ("horizontal_position", "horizontal_velocity", "max_tilt_deg", "max_angular_speed"):
            _positive(name, getattr(self, name), allow_zero=True)
        for name in ("altitude", "vertical_velocity"):
            values = getattr(self, name)
            if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
                raise ValueError(f"invalid {name} interval")
            object.__setattr__(self, name, tuple(values))
        if self.altitude[0] <= 0 or self.max_tilt_deg >= 90:
            raise ValueError("initial altitude must be above ground and tilt below 90 degrees")


@dataclass(frozen=True)
class LandingConfig:
    max_touchdown_speed: float = 2.0
    max_tilt_deg: float = 10.0
    max_angular_speed: float = 0.2
    stability_time: float = 5.0
    target_radius: float = 5.0
    ground_friction: float = 0.8
    ground_angular_damping: float = 8.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _positive(name, getattr(self, name), allow_zero=name in ("ground_friction", "ground_angular_damping"))
        if self.max_tilt_deg >= 90:
            raise ValueError("landing tilt must be below 90 degrees")


@dataclass(frozen=True)
class RewardConfig:
    time_cost: float = 0.02
    horizontal_cost: float = 0.01
    speed_cost: float = 0.005
    tilt_cost: float = 0.02
    main_fuel_cost: float = 0.1
    rcs_fuel_cost: float = 1.0
    safe_bonus: float = 100.0
    crash_cost: float = 100.0
    abort_cost: float = 50.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _positive(name, getattr(self, name), allow_zero=True)


@dataclass(frozen=True)
class SimConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    landing: LandingConfig = field(default_factory=LandingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    control_dt: float = 0.1
    physics_dt: float = 0.02
    max_time: float = 90.0
    max_horizontal_distance: float = 1000.0
    max_altitude: float = 2000.0

    def __post_init__(self):
        for name, typ in (("vehicle", VehicleConfig), ("initial", InitialConfig), ("landing", LandingConfig), ("reward", RewardConfig)):
            if not isinstance(getattr(self, name), typ):
                raise TypeError(f"{name} must be {typ.__name__}; use SimConfig.from_dict for JSON data")
        for name in ("control_dt", "physics_dt", "max_time", "max_horizontal_distance", "max_altitude"):
            _positive(name, getattr(self, name))
        if self.physics_dt > self.control_dt:
            raise ValueError("physics_dt must not exceed control_dt")
        if self.max_altitude <= self.initial.altitude[1]:
            raise ValueError("max_altitude must exceed initial altitude range")
        if self.initial.horizontal_position * math.sqrt(2) > self.max_horizontal_distance:
            raise ValueError("horizontal boundary must contain the initial position box")

    def to_dict(self):
        # JSON canonical form also avoids list/tuple checkpoint incompatibility.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        for key, typ in (("vehicle", VehicleConfig), ("initial", InitialConfig), ("landing", LandingConfig), ("reward", RewardConfig)):
            if key in values:
                values[key] = typ(**values[key])
        return cls(**values)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def perturbed(self, parameter, scale):
        _positive("scale", scale)
        v = self.vehicle
        if parameter == "mass":
            # Add/remove dry ballast: fuel amount, Isp, thrust and inertia stay fixed.
            changes = {"dry_mass": (v.dry_mass + v.fuel_capacity) * scale - v.fuel_capacity}
        elif parameter == "inertia":
            changes = {"inertia": tuple(scale * x for x in v.inertia)}
        elif parameter == "thrust":
            changes = {"max_thrust": v.max_thrust * scale}
        elif parameter == "gravity":
            changes = {"gravity": v.gravity * scale}
        else:
            raise ValueError("parameter must be mass, inertia, thrust or gravity")
        return replace(self, vehicle=replace(v, **changes))

