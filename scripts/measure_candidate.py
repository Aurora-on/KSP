#!/usr/bin/env python3
"""Measure a candidate KSP landing scene without changing the game or task.

This script is read-only.  It records the active craft's mass, inertia,
resources, engines and RCS parts, plus a local terrain sample around the
configured target.  The output is evidence for selecting a craft and site; it
does not fill the ``TBD`` fields in ``configs/mun_landing.toml``.

Example:
    .venv/bin/python scripts/measure_candidate.py \
        --config configs/ksp_mun_test.toml \
        --output reports/candidate-scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ksp_env import KSPConnectionError, KSPEnv


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _vector(value) -> list[float]:
    return [float(item) for item in value]


def _safe_float(value) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense system with partial pivoting."""

    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular plane fit")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][size] for index in range(size)]


def fit_plane(rows: list[dict]) -> dict | None:
    """Fit ``height = a*north + b*east + c`` to valid terrain samples."""

    valid = [row for row in rows if row["valid"]]
    if len(valid) < 3:
        return None

    count = len(valid)
    sum_n = sum(row["north_m"] for row in valid)
    sum_e = sum(row["east_m"] for row in valid)
    sum_nn = sum(row["north_m"] ** 2 for row in valid)
    sum_ee = sum(row["east_m"] ** 2 for row in valid)
    sum_ne = sum(row["north_m"] * row["east_m"] for row in valid)
    sum_h = sum(row["height_m"] for row in valid)
    sum_nh = sum(row["north_m"] * row["height_m"] for row in valid)
    sum_eh = sum(row["east_m"] * row["height_m"] for row in valid)
    matrix = [
        [sum_nn, sum_ne, sum_n],
        [sum_ne, sum_ee, sum_e],
        [sum_n, sum_e, count],
    ]
    try:
        gradient_north, gradient_east, intercept = _solve_linear(
            matrix, [sum_nh, sum_eh, sum_h]
        )
    except ValueError:
        return None

    residuals = [
        row["height_m"]
        - (
            gradient_north * row["north_m"]
            + gradient_east * row["east_m"]
            + intercept
        )
        for row in valid
    ]
    slope_rad = math.atan(math.hypot(gradient_north, gradient_east))
    normal = (
        1.0,
        -gradient_north,
        -gradient_east,
    )
    norm = math.sqrt(sum(value * value for value in normal))
    return {
        "gradient_north": gradient_north,
        "gradient_east": gradient_east,
        "height_at_target_m": intercept,
        "slope_deg": math.degrees(slope_rad),
        "normal_surface_frame_xyz": [value / norm for value in normal],
        "rms_residual_m": math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        ),
        "samples_used": len(valid),
    }


def terrain_samples(
    conn,
    body,
    latitude_deg: float,
    longitude_deg: float,
    *,
    step_m: float,
    ray_start_m: float,
) -> dict:
    """Cast short vertical rays around the target and fit a local plane."""

    surface = body.surface_reference_frame(latitude_deg, longitude_deg, 0.0)
    rows = []
    for north in (-step_m, 0.0, step_m):
        for east in (-step_m, 0.0, step_m):
            position = (ray_start_m, north, east)
            direction = (-1.0, 0.0, 0.0)
            part = conn.space_center.raycast_part(position, direction, surface)
            distance = _safe_float(
                conn.space_center.raycast_distance(position, direction, surface)
            )
            height = (
                ray_start_m - distance
                if distance is not None and math.isfinite(distance)
                else None
            )
            rows.append(
                {
                    "north_m": north,
                    "east_m": east,
                    "ray_distance_m": distance,
                    "height_m": height,
                    "hit_part": None if part is None else getattr(part, "name", None),
                    "valid": height is not None and part is None,
                }
            )

    fit = fit_plane(rows)
    return {
        "frame": "body.surface_reference_frame(lat, lon, 0)",
        "frame_axes": "x=up, y=north, z=east",
        "step_m": step_m,
        "ray_start_m": ray_start_m,
        "samples": rows,
        "fit": fit,
    }


def resource_summary(vessel) -> list[dict]:
    grouped: dict[str, dict] = {}
    for resource in vessel.resources.all:
        row = grouped.setdefault(
            resource.name,
            {
                "name": resource.name,
                "amount": 0.0,
                "max": 0.0,
                "density_kg_per_unit": float(resource.density),
            },
        )
        row["amount"] += float(resource.amount)
        row["max"] += float(resource.max)
    for row in grouped.values():
        row["mass_kg"] = row["amount"] * row["density_kg_per_unit"]
        row["max_mass_kg"] = row["max"] * row["density_kg_per_unit"]
    return [
        grouped[name]
        for name in sorted(grouped)
    ]


def engine_summary(vessel) -> dict:
    groups: dict[str, dict] = {}
    engines = []
    for engine in vessel.parts.engines:
        part_name = getattr(engine.part, "name", None)
        propellants = [str(name) for name in engine.propellant_names]
        row = {
            "part_name": part_name,
            "active": bool(engine.active),
            "has_fuel": bool(engine.has_fuel),
            "thrust_n": float(engine.thrust),
            "max_thrust_n": _safe_float(engine.max_thrust),
            "max_vacuum_thrust_n": _safe_float(engine.max_vacuum_thrust),
            "vacuum_isp_s": _safe_float(engine.vacuum_specific_impulse),
            "propellants": propellants,
        }
        engines.append(row)
        group = groups.setdefault(
            part_name,
            {
                "part_name": part_name,
                "count": 0,
                "active_count": 0,
                "current_thrust_n": 0.0,
                "max_thrust_n": 0.0,
                "max_vacuum_thrust_n": 0.0,
                "vacuum_isp_s": row["vacuum_isp_s"],
                "propellants": propellants,
            },
        )
        group["count"] += 1
        group["active_count"] += int(row["active"])
        group["current_thrust_n"] += row["thrust_n"]
        group["max_thrust_n"] += row["max_thrust_n"] or 0.0
        group["max_vacuum_thrust_n"] += row["max_vacuum_thrust_n"] or 0.0

    active = [row for row in engines if row["active"]]
    active_propellants = sorted(
        {name for row in active for name in row["propellants"]}
    )
    return {
        "engines": engines,
        "groups": [groups[name] for name in sorted(groups)],
        "active_count": len(active),
        "active_propellants": active_propellants,
        "active_current_thrust_n": sum(row["thrust_n"] for row in active),
        "active_max_thrust_n": sum(
            row["max_thrust_n"] or 0.0 for row in active
        ),
        "active_max_vacuum_thrust_n": sum(
            row["max_vacuum_thrust_n"] or 0.0 for row in active
        ),
    }


def rcs_summary(vessel) -> dict:
    groups: dict[str, dict] = {}
    for part in vessel.parts.rcs:
        part_name = getattr(part.part, "name", None)
        row = groups.setdefault(
            part_name,
            {
                "part_name": part_name,
                "count": 0,
                "has_fuel_count": 0,
                "propellants": sorted(str(name) for name in part.propellants),
            },
        )
        row["count"] += 1
        row["has_fuel_count"] += int(bool(part.has_fuel))
    return {
        "groups": [groups[name] for name in sorted(groups)],
        "part_count": sum(row["count"] for row in groups.values()),
        "has_fuel_part_count": sum(
            row["has_fuel_count"] for row in groups.values()
        ),
        "propellants": sorted(
            {name for row in groups.values() for name in row["propellants"]}
        ),
    }


def build_report(
    env: KSPEnv, *, config_path: str, step_m: float, ray_start_m: float
) -> dict:
    conn = env.connection
    scene = _enum_name(conn.krpc.game_scene)
    vessel = conn.space_center.active_vessel
    if vessel is None:
        raise RuntimeError("no active vessel")
    body = vessel.orbit.body
    latitude_deg, longitude_deg, target_altitude_m = env._target_coordinates()
    terrain = terrain_samples(
        conn,
        body,
        latitude_deg,
        longitude_deg,
        step_m=step_m,
        ray_start_m=ray_start_m,
    )
    resources = resource_summary(vessel)
    engines = engine_summary(vessel)
    rcs = rcs_summary(vessel)
    flight = vessel.flight(body.reference_frame)
    config = env.config.raw
    observation = config.get("observation", {})
    action = config.get("action", {})
    scenario = config.get("scenario", {})
    resource_mass_kg = sum(row["mass_kg"] for row in resources)
    configured_fuel = observation.get("main_engine_fuel_resource")
    active_propellants = engines["active_propellants"]
    active_propellant_mass_kg = sum(
        row["mass_kg"]
        for row in resources
        if row["name"] in active_propellants
    )
    warnings = []

    if config.get("status") != "nominal":
        warnings.append(
            "The source profile is not the final nominal task; measurements do "
            "not change configs/mun_landing.toml or readiness flags."
        )
    if len(active_propellants) > 1 and configured_fuel in active_propellants:
        warnings.append(
            "Active engines consume more than one propellant but the configured "
            "main_engine_fuel_resource names only one; decide whether the "
            "observation should use total propellant mass in kg."
        )
    if terrain["fit"] is None:
        warnings.append(
            "The terrain plane fit had fewer than three valid samples; inspect "
            "the ray grid before using this site."
        )
    if terrain["fit"] is not None:
        center = next(
            (
                row
                for row in terrain["samples"]
                if row["north_m"] == 0.0 and row["east_m"] == 0.0
            ),
            None,
        )
        if center is not None and center["hit_part"] is not None:
            warnings.append(
                "The target-center ray hit the vessel; that sample was excluded "
                "from the terrain plane fit."
            )
    if target_altitude_m == 0.0 and terrain["fit"] is not None:
        warnings.append(
            "The configured target altitude is 0 m; the measured terrain height "
            "at the target differs. Define the clearance reference before reset."
        )

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_path,
        "read_only": True,
        "status": "candidate_scene_measurement_not_task_decision",
        "scene": {
            "name": scene,
            "paused": bool(conn.krpc.paused),
            "ut_s": float(conn.space_center.ut),
        },
        "target": {
            "body": body.name,
            "latitude_deg": latitude_deg,
            "longitude_deg": longitude_deg,
            "target_altitude_m": target_altitude_m,
            "vessel_latitude_deg": float(flight.latitude),
            "vessel_longitude_deg": float(flight.longitude),
            "vessel_surface_altitude_m": float(flight.surface_altitude),
            "terrain": terrain,
        },
        "vessel": {
            "name": vessel.name,
            "situation": _enum_name(vessel.situation),
            "wet_mass_kg": float(vessel.mass),
            "dry_mass_kg": float(vessel.dry_mass),
            "resource_mass_kg": resource_mass_kg,
            "mass_minus_dry_kg": float(vessel.mass) - float(vessel.dry_mass),
            "active_propellant_mass_kg": active_propellant_mass_kg,
            "moment_of_inertia_kg_m2": _vector(vessel.moment_of_inertia),
            "inertia_tensor_kg_m2": _vector(vessel.inertia_tensor),
            "resources": resources,
            "engines": engines,
            "rcs": rcs,
            "control": {
                "throttle": float(vessel.control.throttle),
                "rcs": bool(vessel.control.rcs),
                "sas": bool(vessel.control.sas),
                "reaction_wheels": bool(vessel.control.reaction_wheels),
                "input_mode": _enum_name(vessel.control.input_mode),
                "current_stage": int(vessel.control.current_stage),
            },
        },
        "interface": {
            "status": config.get("status"),
            "task": config.get("task"),
            "craft": scenario.get("craft"),
            "site": scenario.get("site"),
            "save_name": scenario.get("save_name"),
            "observation_fields": observation.get("fields"),
            "observation_frame": observation.get("frame"),
            "main_engine_fuel_resource": configured_fuel,
            "body_to_task_rotation_xyzw": observation.get(
                "body_to_task_rotation_xyzw"
            ),
            "action_fields": action.get("fields"),
            "rcs_semantics": action.get("rcs_semantics"),
            "control_hz": scenario.get("control_hz"),
            "episode_horizon_s": scenario.get("episode_horizon_s"),
            "readiness": {
                "initial_state_ready": config.get("initial_state_ready"),
                "success_thresholds_ready": config.get("success_thresholds_ready"),
                "reset_step_validated": config.get("reset_step_validated"),
            },
        },
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ksp_mun_test.toml")
    parser.add_argument("--terrain-step", type=float, default=5.0)
    parser.add_argument("--ray-start", type=float, default=100.0)
    parser.add_argument("--output", type=Path, help="write the complete report as JSON")
    args = parser.parse_args()

    if args.terrain_step <= 0.0:
        parser.error("--terrain-step must be positive")
    if args.ray_start <= 0.0:
        parser.error("--ray-start must be positive")

    env = KSPEnv.from_config(args.config)
    try:
        env.connect()
    except KSPConnectionError as exc:
        print(f"NOT_READY: {exc}; load a save and start the kRPC server", file=sys.stderr)
        return 2

    try:
        conn = env.connection
        scene = _enum_name(conn.krpc.game_scene)
        if scene != "flight":
            print(f"NOT_READY: Flight scene required, current scene is {scene}", file=sys.stderr)
            return 2
        if conn.space_center.active_vessel is None:
            print("NOT_READY: no active vessel", file=sys.stderr)
            return 2

        report = build_report(
            env,
            config_path=str(args.config),
            step_m=args.terrain_step,
            ray_start_m=args.ray_start,
        )
        target = report["target"]
        vessel = report["vessel"]
        fit = target["terrain"]["fit"] or {}
        engines = vessel["engines"]
        resources = {row["name"]: row for row in vessel["resources"]}

        print("candidate scene measurement (read-only)")
        print(f"vessel: {vessel['name']} ({vessel['situation']})")
        print(
            f"mass: wet {vessel['wet_mass_kg']:.3f} kg, "
            f"dry {vessel['dry_mass_kg']:.3f} kg, "
            f"resource {vessel['resource_mass_kg']:.3f} kg"
        )
        print(
            f"active engines: {engines['active_count']}, "
            f"max thrust {engines['active_max_thrust_n']:.1f} N, "
            f"propellants {engines['active_propellants']}, "
            f"propellant mass {vessel['active_propellant_mass_kg']:.3f} kg"
        )
        for name in ("LiquidFuel", "Oxidizer", "MonoPropellant", "SolidFuel"):
            if name in resources:
                row = resources[name]
                print(
                    f"  {name}: {row['amount']:.3f}/{row['max']:.3f} units, "
                    f"{row['mass_kg']:.3f} kg"
                )
        print(
            f"target: lat {target['latitude_deg']:.8f}, "
            f"lon {target['longitude_deg']:.8f}, "
            f"config altitude {target['target_altitude_m']:.3f} m"
        )
        print(
            f"terrain: slope {fit.get('slope_deg', float('nan')):.4f}°, "
            f"height at target {fit.get('height_at_target_m', float('nan')):.3f} m, "
            f"RMS {fit.get('rms_residual_m', float('nan')):.3f} m"
        )
        print(
            f"vessel surface altitude: "
            f"{target['vessel_surface_altitude_m']:.3f} m"
        )
        if report["warnings"]:
            print("warnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")

        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
            print(f"JSON report: {args.output.resolve()}")
        else:
            print(encoded, end="")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
