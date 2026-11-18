from dataclasses import dataclass
import numpy as np
from .math3d import normalize_quaternion, quat_multiply


@dataclass
class LanderState:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    fuel: float

    def __post_init__(self):
        for name in ("position", "velocity", "angular_velocity"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            setattr(self, name, value)
        self.quaternion = normalize_quaternion(self.quaternion)
        self.fuel = float(self.fuel)
        if not np.isfinite(self.fuel) or self.fuel < 0:
            raise ValueError("fuel must be finite and nonnegative")

    def to_dict(self):
        return {"position": self.position.tolist(), "velocity": self.velocity.tolist(),
                "quaternion": self.quaternion.tolist(), "angular_velocity": self.angular_velocity.tolist(),
                "fuel": self.fuel}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def observation(self):
        return np.concatenate((self.position, self.velocity, self.quaternion,
                               self.angular_velocity, [self.fuel])).astype(np.float32)

    def copy(self):
        return LanderState.from_dict(self.to_dict())


def sample_initial_state(rng, config):
    """Uniform boxes, uniform solid-angle thrust cone, uniform yaw and rate ball."""
    c = config.initial
    xy = rng.uniform(-c.horizontal_position, c.horizontal_position, 2)
    z = rng.uniform(*c.altitude)
    vxy = rng.uniform(-c.horizontal_velocity, c.horizontal_velocity, 2)
    vz = rng.uniform(*c.vertical_velocity)
    tilt = np.arccos(rng.uniform(np.cos(np.deg2rad(c.max_tilt_deg)), 1))
    azimuth, yaw = rng.uniform(-np.pi, np.pi, 2)
    qt = np.array([np.cos(tilt/2), np.cos(azimuth)*np.sin(tilt/2), np.sin(azimuth)*np.sin(tilt/2), 0])
    qy = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    omega = axis * c.max_angular_speed * rng.uniform() ** (1/3)
    return LanderState(np.r_[xy,z], np.r_[vxy,vz], quat_multiply(qt,qy), omega, config.vehicle.fuel_capacity)

