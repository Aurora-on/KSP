"""Engineering reference controller. It consumes the public observation only."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .math3d import rotation_matrix


class ScriptedController:
    """Cascaded position, velocity and thrust-axis controller.

    This is a deliberately small, untrained engineering baseline. It estimates
    mass from observed fuel and the nominal dry mass. During parameter sweeps,
    keep this nominal configuration fixed to avoid privileged parameter access.
    Yaw is unconstrained; all three body angular velocities are damped.
    """

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

    def reset(self) -> None:
        """Stateless, retained for compatibility with stateful policies."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (14,) or not np.all(np.isfinite(obs)):
            raise ValueError("Expected finite raw observation of shape (14,)")
        vehicle = self.config.vehicle
        position, velocity = obs[:3], obs[3:6]
        rotation = rotation_matrix(obs[6:10])
        omega = obs[10:13]
        mass = vehicle.dry_mass + max(0.0, obs[13])

        # Smooth descent gives lateral guidance time to converge and reduces
        # touchdown speed without accessing the environment's contact flag.
        desired_vz = -min(6.0, 0.4 + 0.12 * max(0.0, position[2]))
        vertical_acceleration = 1.2 * (desired_vz - velocity[2])
        thrust_acceleration_z = max(0.15, vehicle.gravity + vertical_acceleration)
        lateral_acceleration = -0.065 * position[:2] - 0.65 * velocity[:2]
        max_lateral = min(0.8, np.tan(np.deg2rad(20.0)) * thrust_acceleration_z)
        lateral_norm = np.linalg.norm(lateral_acceleration)
        if lateral_norm > max_lateral:
            lateral_acceleration *= max_lateral / lateral_norm
        desired_axis = np.array([*lateral_acceleration, thrust_acceleration_z])
        desired_axis /= np.linalg.norm(desired_axis)
        # Cut at 8 cm clearance to avoid relaunch during the unobserved
        # remainder of the control interval containing the first impact.
        on_ground = position[2] <= 0.08 and velocity[2] <= 0.25
        if on_ground:
            desired_axis = np.array([0.0, 0.0, 1.0])
        axis_error_body = rotation.T @ np.cross(rotation[:, 2], desired_axis)
        inertia = np.asarray(vehicle.inertia)
        desired_torque = inertia * (4.0 * axis_error_body - 3.8 * omega)
        desired_torque += np.cross(omega, inertia * omega)
        rcs = np.clip(desired_torque / np.asarray(vehicle.rcs_torque), -1.0, 1.0)
        throttle = mass * thrust_acceleration_z / (vehicle.max_thrust * max(0.25, rotation[2, 2]))
        if on_ground:
            throttle = 0.0
        return np.array([np.clip(throttle, 0.0, 1.0), *rcs], dtype=np.float32)
