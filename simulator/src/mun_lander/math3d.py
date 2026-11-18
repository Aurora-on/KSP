"""Right-handed rotations; scalar-first Hamilton quaternions, body to world."""
import numpy as np


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be a finite nonzero 4-vector")
    return q / norm


def quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def rotation_matrix(q):
    w, x, y, z = normalize_quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion_from_euler(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return normalize_quaternion([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                                 cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def tilt_angle(q):
    w, x, y, z = normalize_quaternion(q)
    return float(np.arccos(np.clip(1 - 2*(x*x+y*y), -1, 1)))

