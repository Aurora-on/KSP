"""Simulator target domains and observation transforms for transfer evaluations."""
import math
from numbers import Real

import numpy as np
from mun_lander import SimConfig


def target_config(domain: str) -> SimConfig:
    """Change only vehicle inertia, leaving task, reset and policy transforms unchanged."""
    scales = {"P1": 4.0, "P2": 0.75 / 0.07}
    if domain not in scales:
        raise ValueError(f"unknown simulator target domain {domain!r}; expected P1 or P2")
    return SimConfig().perturbed("inertia", scales[domain])


def rescale_fuel(obs, *, source_capacity_kg, target_capacity_kg):
    """Express target fuel in equivalent source kg at the same tank fill fraction."""
    for capacity in (source_capacity_kg, target_capacity_kg):
        if (not isinstance(capacity, Real) or isinstance(capacity, bool)
                or not math.isfinite(capacity) or capacity <= 0):
            raise ValueError("fuel capacities must be positive finite masses in kg")
    values = np.asarray(obs)
    if values.ndim == 0 or values.shape[-1] != 14 or values.dtype.kind not in "fiu":
        raise ValueError("expected numeric observations with trailing dimension 14")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")

    result = values.astype(np.result_type(values.dtype, np.float32), copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        result[..., 13] *= float(source_capacity_kg) / float(target_capacity_kg)
    if not np.isfinite(result[..., 13]).all():
        raise ValueError("rescaled fuel is outside the representable range")
    return result
