"""Shared reward definition for the source simulator and target adapter."""
import numpy as np
from .math3d import tilt_angle


def compute_reward(state, main_used, rcs_used, dt, status, config):
    c = config.reward
    terminal = c.safe_bonus if status == "safe_landing" else -c.crash_cost if status == "crash" else -c.abort_cost if status in ("fuel_exhausted", "timeout", "out_of_bounds") else 0.0
    return {"time": -c.time_cost*dt,
            "horizontal": -c.horizontal_cost*float(np.linalg.norm(state.position[:2]))*dt,
            "speed": -c.speed_cost*float(np.linalg.norm(state.velocity))*dt,
            "tilt": -c.tilt_cost*tilt_angle(state.quaternion)*dt,
            "main_fuel": -c.main_fuel_cost*main_used,
            "rcs_fuel": -c.rcs_fuel_cost*rcs_used,
            "terminal": terminal}
