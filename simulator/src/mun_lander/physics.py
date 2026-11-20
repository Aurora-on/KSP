"""6-DOF rigid-body free flight, RK4 and exact fuel-burn duration splitting."""
import numpy as np
from .state import LanderState

G0 = 9.80665


def _rhs(y, thrust, torque, v, burn_rate):
    q = y[6:10]
    qw, qx, qy, qz = q / np.linalg.norm(q)
    wx, wy, wz = y[10:13]
    ix, iy, iz = v.inertia
    dy = np.empty(14)
    dy[:3] = y[3:6]
    accel = thrust / (v.dry_mass + max(y[13], 0))
    dy[3] = 2 * (qx*qz + qw*qy) * accel
    dy[4] = 2 * (qy*qz - qw*qx) * accel
    dy[5] = (1 - 2*(qx*qx + qy*qy)) * accel - v.gravity
    dy[6:10] = 0.5*np.array([-qx*wx-qy*wy-qz*wz, qw*wx+qy*wz-qz*wy,
                            qw*wy-qx*wz+qz*wx, qw*wz+qx*wy-qy*wx])
    dy[10] = (torque[0] + (iy-iz)*wy*wz)/ix
    dy[11] = (torque[1] + (iz-ix)*wz*wx)/iy
    dy[12] = (torque[2] + (ix-iy)*wx*wy)/iz
    dy[13] = -burn_rate
    return dy


def _rk4(y, dt, thrust, torque, vehicle, burn_rate):
    a = _rhs(y, thrust, torque, vehicle, burn_rate)
    b = _rhs(y + 0.5*dt*a, thrust, torque, vehicle, burn_rate)
    c = _rhs(y + 0.5*dt*b, thrust, torque, vehicle, burn_rate)
    d = _rhs(y + dt*c, thrust, torque, vehicle, burn_rate)
    result = y + dt/6 * (a+2*b+2*c+d)
    result[6:10] /= np.linalg.norm(result[6:10])
    result[13] = max(0.0, result[13])
    return result


def integrate_free(state, action, vehicle, dt):
    """Integrate one small step; no ground interaction. Action must be bounded."""
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    u = np.asarray(action, dtype=np.float64)
    if u.shape != (4,) or not np.all(np.isfinite(u)) or u[0] < 0 or u[0] > 1 or np.any(np.abs(u[1:]) > 1):
        raise ValueError("physics action outside [0,1] x [-1,1]^3")
    torque = u[1:] * np.asarray(vehicle.rcs_torque)
    thrust = float(u[0]) * vehicle.max_thrust
    burn_rate = thrust / (vehicle.main_isp * G0)
    burn_time = min(dt, state.fuel / burn_rate) if burn_rate else 0.0
    y = np.concatenate((state.position, state.velocity, state.quaternion, state.angular_velocity, [state.fuel]))
    if burn_time > 0:
        y = _rk4(y, burn_time, thrust, torque, vehicle, burn_rate)
    if burn_time < dt:
        y = _rk4(y, dt-burn_time, 0.0, torque, vehicle, 0.0)
    rcs_used = float(np.sum(np.abs(torque))) * dt / (vehicle.rcs_lever_arm * vehicle.rcs_isp * G0)
    return LanderState(y[:3], y[3:6], y[6:10], y[10:13], y[13]), rcs_used

