"""Transform the lander's native observations and actions at the agent boundary."""

import numpy as np
from mun_lander import SimConfig


def interface_spec(config: SimConfig) -> dict:
    """Return the JSON-serializable preprocessing contract stored in a checkpoint."""
    return {
        "attitude_encoding": "rot6d",
        "scales": {
            "position": 100.0,
            "velocity": 10.0,
            "angular_velocity": 0.5,
            "fuel": config.vehicle.fuel_capacity,
        },
        "control_dt": config.control_dt,
        "source_config": config.to_dict(),
    }


def validate_interface_spec(saved: dict, config: SimConfig) -> None:
    """Reject a checkpoint whose transforms or source simulator configuration differ."""
    expected = interface_spec(config)
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
