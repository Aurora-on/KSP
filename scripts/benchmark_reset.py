#!/usr/bin/env python3
"""Compare named-save loads with an in-memory Debug reset.

This is a measurement harness, not a task reset implementation.  It reports
wall-clock cost and observed-state spread for:

* loading the base snapshot five times;
* loading a fixed bank snapshot five times;
* applying one in-memory state five times;
* applying the same state with bounded correction five times.

The script leaves the game at the base snapshot and paused.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ksp_env import KSPConnectionError, KSPEnv
from ksp_env.geometry import rotate_vector, quaternion_from_unit_vectors


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _vector(value) -> list[float]:
    return [float(item) for item in value]


def snapshot(env: KSPEnv) -> dict:
    conn = env.connection
    vessel = conn.space_center.active_vessel
    flight = vessel.flight(vessel.orbit.body.reference_frame)
    observation = env.observe()
    return {
        "ut_s": float(conn.space_center.ut),
        "position_m": _vector(observation.position_m),
        "velocity_m_s": _vector(observation.velocity_m_s),
        "attitude_quaternion": _vector(observation.attitude_quaternion),
        "angular_velocity_rad_s": _vector(observation.angular_velocity_rad_s),
        "surface_altitude_m": float(flight.surface_altitude),
        "vertical_speed_m_s": float(flight.vertical_speed),
        "main_engine_fuel": float(observation.main_engine_fuel),
        "legs": [
            {
                "part_name": getattr(leg.part, "name", None),
                "state": _enum_name(leg.state),
                "deployed": bool(leg.deployed),
            }
            for leg in vessel.parts.legs
        ],
        "gear": bool(vessel.control.gear),
    }


def load_and_measure(env: KSPEnv, save_name: str) -> dict:
    started = time.perf_counter()
    conn = env.connection
    conn.space_center.load(save_name)
    conn.krpc.paused = True
    elapsed = time.perf_counter() - started
    return {"elapsed_s": elapsed, "observed": snapshot(env)}


def apply_state(env: KSPEnv, desired: dict, *, altitude_offset: float, velocity_offset: float) -> dict:
    conn = env.connection
    vessel = conn.space_center.active_vessel
    frame = env._target_reference_frame()
    body_to_task = env._body_to_task_rotation()
    desired_position = (
        desired["position_m"][0],
        desired["position_m"][1],
        desired["position_m"][2] + altitude_offset,
    )
    desired_velocity = (
        desired["velocity_m_s"][0],
        desired["velocity_m_s"][1],
        desired["velocity_m_s"][2] + velocity_offset,
    )
    task_z = desired["body_z_direction"]
    ksp_y_in_task = rotate_vector(body_to_task, (0.0, 1.0, 0.0))
    task_to_target = quaternion_from_unit_vectors((0.0, 0.0, 1.0), task_z)
    ksp_y_in_target = rotate_vector(task_to_target, ksp_y_in_task)

    started = time.perf_counter()
    conn.krpc.paused = False
    conn.debug.set_velocity(desired_velocity, frame)
    conn.debug.set_direction(ksp_y_in_target, desired["roll_deg"], frame)
    conn.debug.set_angular_velocity(desired["angular_velocity_rad_s"], frame)
    conn.debug.set_position(desired_position, frame)
    conn.krpc.paused = True
    elapsed = time.perf_counter() - started
    return {"elapsed_s": elapsed, "observed": snapshot(env)}


def errors(observed: dict, desired: dict) -> dict:
    dx = observed["position_m"][0] - desired["position_m"][0]
    dy = observed["position_m"][1] - desired["position_m"][1]
    dz = observed["surface_altitude_m"] - desired["clearance_m"]
    dvx = observed["velocity_m_s"][0] - desired["velocity_m_s"][0]
    dvy = observed["velocity_m_s"][1] - desired["velocity_m_s"][1]
    dvz = observed["velocity_m_s"][2] - desired["velocity_m_s"][2]
    return {
        "position_error_m": [dx, dy, dz],
        "horizontal_error_m": math.hypot(dx, dy),
        "velocity_error_m_s": [dvx, dvy, dvz],
        "speed_error_m_s": math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz),
    }


def corrected_apply(
    env: KSPEnv,
    desired: dict,
    *,
    attempts: int = 4,
    tolerance_m: float = 2.0,
    velocity_tolerance_m_s: float = 0.5,
) -> dict:
    altitude_offset = 30.0
    velocity_offset = 10.0
    applied = []
    for attempt in range(attempts):
        result = apply_state(
            env,
            desired,
            altitude_offset=altitude_offset,
            velocity_offset=velocity_offset,
        )
        error = errors(result["observed"], desired)
        applied.append(
            {
                "attempt": attempt + 1,
                "altitude_offset_m": altitude_offset,
                "elapsed_s": result["elapsed_s"],
                **error,
            }
        )
        if (
            error["horizontal_error_m"] <= tolerance_m
            and abs(error["position_error_m"][2]) <= tolerance_m
            and error["speed_error_m_s"] <= velocity_tolerance_m_s
        ):
            break
        # Correct the Debug request from the observed state.  The vertical
        # velocity is deliberately not corrected here; the test only evaluates
        # position/clearance correction.
        altitude_offset += desired["clearance_m"] - result["observed"]["surface_altitude_m"]
    return {
        "attempts": applied,
        "elapsed_s": sum(attempt["elapsed_s"] for attempt in applied),
        "observed": result["observed"],
        "error": errors(result["observed"], desired),
    }


def _summary(rows: list[dict], key: str) -> dict:
    elapsed = [row["elapsed_s"] for row in rows]
    return {
        "count": len(rows),
        "elapsed_mean_s": sum(elapsed) / len(elapsed),
        "elapsed_min_s": min(elapsed),
        "elapsed_max_s": max(elapsed),
        "observed_altitudes_m": [row["observed"]["surface_altitude_m"] for row in rows],
        "observed_positions_m": [row["observed"]["position_m"] for row in rows],
        "observed_vertical_speeds_m_s": [
            row["observed"]["vertical_speed_m_s"] for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ksp_mun_test.toml")
    parser.add_argument("--base-save", default="Mun landing-candidate-140p5-100m-legs")
    parser.add_argument(
        "--bank-save", default="Mun landing-bank-140p5-v3-33-00"
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/reset-benchmark.json")
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    desired = {
        "clearance_m": 100.0,
        "position_m": (5.0, 5.0, 100.0),
        "velocity_m_s": (1.0, -1.0, -5.0),
        "body_z_direction": (0.05, 0.0, math.sqrt(1.0 - 0.05**2)),
        "roll_deg": 30.0,
        "angular_velocity_rad_s": (0.01, -0.02, 0.0),
    }
    env = KSPEnv.from_config(args.config)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "base_save": args.base_save,
        "bank_save": args.bank_save,
        "desired": {
            "clearance_m": desired["clearance_m"],
            "position_m": list(desired["position_m"]),
            "velocity_m_s": list(desired["velocity_m_s"]),
            "body_z_direction": list(desired["body_z_direction"]),
            "roll_deg": desired["roll_deg"],
            "angular_velocity_rad_s": list(desired["angular_velocity_rad_s"]),
        },
        "base_loads": [],
        "bank_loads": [],
        "memory_raw": [],
        "memory_corrected": [],
    }

    try:
        env.connect()
    except KSPConnectionError as exc:
        print(f"NOT_READY: {exc}; load a save and start the kRPC server", file=sys.stderr)
        return 2

    conn = env.connection
    try:
        if _enum_name(conn.krpc.game_scene) != "flight":
            print("NOT_READY: Flight scene required", file=sys.stderr)
            return 2

        for _ in range(args.count):
            report["base_loads"].append(load_and_measure(env, args.base_save))
        for _ in range(args.count):
            report["bank_loads"].append(load_and_measure(env, args.bank_save))

        conn.space_center.load(args.base_save)
        conn.krpc.paused = True
        for _ in range(args.count):
            result = apply_state(
                env, desired, altitude_offset=30.0, velocity_offset=10.0
            )
            report["memory_raw"].append(
                {
                    "elapsed_s": result["elapsed_s"],
                    "observed": result["observed"],
                    "error": errors(result["observed"], desired),
                }
            )

        conn.space_center.load(args.base_save)
        conn.krpc.paused = True
        for _ in range(args.count):
            report["memory_corrected"].append(corrected_apply(env, desired))

        conn.space_center.load(args.base_save)
        conn.krpc.paused = True
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            conn.krpc.paused = True
        except Exception:
            pass
        env.close()

    report["summary"] = {
        "base_loads": _summary(report["base_loads"], "base"),
        "bank_loads": _summary(report["bank_loads"], "bank"),
        "memory_raw": _summary(report["memory_raw"], "raw"),
        "memory_corrected": _summary(report["memory_corrected"], "corrected"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"JSON report: {args.output.resolve()}")
    for name in ("base_loads", "bank_loads", "memory_raw", "memory_corrected"):
        summary = report["summary"][name]
        print(
            f"{name}: n={summary['count']} "
            f"mean={summary['elapsed_mean_s']:.3f}s "
            f"range=({summary['elapsed_min_s']:.3f}, {summary['elapsed_max_s']:.3f}) "
            f"altitudes={[round(value, 2) for value in summary['observed_altitudes_m']]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
