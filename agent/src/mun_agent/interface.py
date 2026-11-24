"""Transform the lander's native observations and actions at the agent boundary."""

import numpy as np
from mun_lander import SimConfig


def interface_spec(config: SimConfig, *, ground_history=False, legacy=False) -> dict:
    """Return the JSON-serializable preprocessing contract stored in a checkpoint."""
    spec = {
        "attitude_encoding": "rot6d",
        "scales": {
            "position": 100.0 if legacy else 20.0,
            "velocity": 10.0 if legacy else 5.0,
            "angular_velocity": 0.5 if legacy else 0.2,
            "fuel": config.vehicle.fuel_capacity,
        },
        "control_dt": config.control_dt,
        "source_config": config.to_dict(),
    }
    if not legacy:
        spec["ground_history_cap"] = (int(np.ceil(config.landing.stability_time / config.control_dt))
                                      if ground_history else 0)
    return spec


def validate_interface_spec(saved: dict, config: SimConfig, *, ground_history=False, legacy=False) -> None:
    """Reject a checkpoint whose transforms or source simulator configuration differ."""
    expected = interface_spec(config, ground_history=ground_history, legacy=legacy)
    if not isinstance(saved, dict) or saved.keys() != expected.keys():
        raise ValueError("checkpoint interface specification has missing or unknown fields")
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f"checkpoint interface mismatch in {key}")


def _vectors(value, width, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with trailing dimension {width}")
    return array


def preprocess(obs, spec: dict) -> np.ndarray:
    """Convert one observation or a batch [..., 14] to float32 features [..., 16].

    Rotation features concatenate columns 0 and 1 of the body-to-world matrix.
    """
    if spec["attitude_encoding"] != "rot6d":
        raise ValueError("unsupported attitude encoding")
    obs = _vectors(obs, 14, "observation")
    scales = spec["scales"]
    quat = obs[..., 6:10].astype(np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = np.moveaxis(quat / norm, -1, 0)
    rotation = np.stack((
        1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w),
        2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w),
    ), axis=-1)
    return np.concatenate((
        obs[..., :3] / scales["position"],
        obs[..., 3:6] / scales["velocity"],
        rotation,
        obs[..., 10:13] / scales["angular_velocity"],
        obs[..., 13:14] / scales["fuel"],
    ), axis=-1, dtype=np.float32)


def episode_features(obs, spec):
    """Encode a complete episode, counting consecutive zero-clearance frames causally."""
    features = preprocess(obs, spec)
    cap = spec.get("ground_history_cap", 0)
    if not cap:
        return features
    if features.ndim != 2:
        raise ValueError("ground history requires one complete episode in temporal order")
    index = np.arange(1, len(obs) + 1)
    last_air = np.maximum.accumulate(np.where(obs[:, 2] == 0, 0, index))
    duration = np.minimum(index - last_air, cap).astype(np.float32) / cap
    return np.column_stack((features, duration))


def physical_fields(features, spec: dict) -> dict:
    """Decode [..., 16] features for diagnostics in SI units and tilt in degrees.

    Report tilt only where tilt_valid is true. A zero or collinear rot6d pair
    has no defined tilt and receives a finite placeholder (zero degrees).
    """
    features = _vectors(features, 16, "features")
    scales = spec["scales"]
    first = features[..., 6:9].astype(np.float64)
    second = features[..., 9:12].astype(np.float64)
    eps = 1e-8
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = first / np.maximum(first_norm, eps)
    projection = (first_unit * second).sum(axis=-1, keepdims=True)
    orthogonal = second - first_unit * projection
    orthogonal_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    second_unit = orthogonal / np.maximum(orthogonal_norm, eps)
    # The third column is the thrust axis expressed in the world frame.
    up = np.cross(first_unit, second_unit)
    valid = (first_norm[..., 0] > eps) & (
        orthogonal_norm[..., 0] > eps * np.maximum(1.0, np.linalg.norm(second, axis=-1))
    )
    tilt = np.degrees(np.arctan2(np.hypot(up[..., 0], up[..., 1]), up[..., 2]))
    return {
        "position": features[..., :3] * scales["position"],
        "velocity": features[..., 3:6] * scales["velocity"],
        "angular_velocity": features[..., 12:15] * scales["angular_velocity"],
        "fuel": features[..., 15] * scales["fuel"],
        "tilt_deg": np.where(valid, tilt, 0.0),
        "tilt_valid": valid,
    }


def to_native_action(action) -> np.ndarray:
    """Map internal throttle from [-1, 1] to [0, 1] as in SymmetricAction."""
    native = _vectors(action, 4, "action").copy()
    native[..., 0] = (native[..., 0] + 1.0) * 0.5
    return native


def to_internal_action(action) -> np.ndarray:
    """Convert native applied actions, including batches, to the model coordinates."""
    internal = _vectors(action, 4, "action").copy()
    internal[..., 0] = 2.0 * internal[..., 0] - 1.0
    return internal
