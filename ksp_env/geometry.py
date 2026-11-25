"""Small quaternion helpers used by the KSP adapter.

Quaternions use the kRPC component order ``(x, y, z, w)``.  The fixed
``body_to_task`` rotation maps vectors in the KSP vessel body frame to vectors
in the shared task body frame.  Keeping it separate from the live quaternion
avoids hard-coding a craft's assembly orientation.
"""

from __future__ import annotations

import math

Quaternion = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("quaternion must not be zero")
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    q = normalize_quaternion(quaternion)
    x, y, z, w = q
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def transform_attitude(
    krpc_body_to_target: Quaternion, body_to_task: Quaternion
) -> Quaternion:
    """Convert a KSP body-to-target quaternion to the task body frame."""

    return normalize_quaternion(
        quaternion_multiply(
            normalize_quaternion(krpc_body_to_target),
            quaternion_conjugate(normalize_quaternion(body_to_task)),
        )
    )


def transform_body_vector(vector: Vector3, body_to_task: Quaternion) -> Vector3:
    """Convert a vector expressed in the KSP body frame to the task body frame."""

    return rotate_vector(body_to_task, vector)


def inverse_transform_body_vector(vector: Vector3, body_to_task: Quaternion) -> Vector3:
    """Convert a task-body vector to the KSP body frame."""

    return rotate_vector(quaternion_conjugate(normalize_quaternion(body_to_task)), vector)


def quaternion_from_unit_vectors(source: Vector3, target: Vector3) -> Quaternion:
    """Return a shortest-arc rotation from one unit vector to another."""

    sx, sy, sz = source
    tx, ty, tz = target
    dot = max(-1.0, min(1.0, sx * tx + sy * ty + sz * tz))
    if dot >= 1.0 - 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    if dot <= -1.0 + 1e-12:
        # Choose any axis perpendicular to the source vector.
        axis = (1.0, 0.0, 0.0) if abs(sx) < 0.9 else (0.0, 1.0, 0.0)
        cx = sy * axis[2] - sz * axis[1]
        cy = sz * axis[0] - sx * axis[2]
        cz = sx * axis[1] - sy * axis[0]
        return normalize_quaternion((cx, cy, cz, 0.0))
    cx = sy * tz - sz * ty
    cy = sz * tx - sx * tz
    cz = sx * ty - sy * tx
    return normalize_quaternion((cx, cy, cz, 1.0 + dot))
