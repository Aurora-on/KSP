#!/usr/bin/env python3
"""Generate a small KSP initial-state bank from the candidate base snapshot.

The sampler is deterministic for a seed. Each state is applied with kRPC
``Debug``, saved as a named snapshot, reloaded and recorded as requested versus
observed. The live run is deliberately a smoke test of the generation module,
not a task-grade initial-state acceptance.

Example:
    .venv/bin/python scripts/generate_initial_state_bank.py \
        --save-dir "/path/to/saves/Munar landing" \
        --count 5 --seed 33 \
        --output reports/initial-state-bank.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ksp_env import KSPConnectionError, KSPEnv, load_config
from ksp_env.initial_states import sample_bank


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _legs(vessel) -> list[dict]:
    return [
        {
            "part_name": getattr(leg.part, "name", None),
            "state": _enum_name(leg.state),
            "deployed": bool(leg.deployed),
            "is_grounded": bool(leg.is_grounded),
        }
        for leg in vessel.parts.legs
    ]


def _tilt_deg(quaternion) -> float:
    x, y, _z, _w = quaternion
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _envelope_check(observed, clearance_m, scenario) -> dict:
    position_range = scenario["initial_position_range"]
    velocity_range = scenario["initial_velocity_range"]
    x_range = [float(value) for value in position_range["x_m"]]
    y_range = [float(value) for value in position_range["y_m"]]
    vx_range = [float(value) for value in velocity_range["x_m_s"]]
    vy_range = [float(value) for value in velocity_range["y_m_s"]]
    vz_range = [float(value) for value in velocity_range["z_m_s"]]
    position = observed["position_m"]
    velocity = observed["velocity_m_s"]
    angular = observed["angular_velocity_rad_s"]
    angular_norm = math.sqrt(sum(value * value for value in angular))
    attitude_tilt = _tilt_deg(observed["attitude_quaternion"])
    checks = {
        "clearance": 80.0 <= clearance_m <= 120.0,
        "position_x": x_range[0] <= position[0] <= x_range[1],
        "position_y": y_range[0] <= position[1] <= y_range[1],
        "velocity_x": vx_range[0] <= velocity[0] <= vx_range[1],
        "velocity_y": vy_range[0] <= velocity[1] <= vy_range[1],
        "velocity_z": vz_range[0] <= velocity[2] <= vz_range[1],
        "attitude_tilt": attitude_tilt <= float(
            scenario["initial_attitude_range"]["max_tilt_deg"]
        ),
        "angular_speed": angular_norm <= float(
            scenario["initial_angular_velocity_range"]["max_norm_rad_s"]
        ),
    }
    return {
        "checks": checks,
        "within_frozen_envelope": all(checks.values()),
        "attitude_tilt_deg": attitude_tilt,
        "angular_speed_rad_s": angular_norm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ksp_mun_test.toml")
    parser.add_argument("--task-config", default="configs/mun_landing.toml")
    parser.add_argument("--base-save", default="Mun landing-candidate-140p5-100m-legs")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--prefix", default="Mun landing-bank-140p5")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--altitude-offset-m",
        type=float,
        default=30.0,
        help="Debug pre-compensation added to requested clearance before saving",
    )
    parser.add_argument(
        "--vertical-velocity-offset-m-s",
        dest="vertical_velocity_offset_m_s",
        type=float,
        default=10.0,
        help="Debug pre-compensation added to requested vertical velocity before saving",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/initial-state-bank.json"))
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")
    if not args.altitude_offset_m >= 0.0:
        parser.error("--altitude-offset-m must be non-negative")
    if not args.vertical_velocity_offset_m_s >= 0.0:
        parser.error("--velocity-offset-m-s must be non-negative")

    task = load_config(args.task_config)
    scenario = task.raw["scenario"]
    states = sample_bank(scenario, seed=args.seed, count=args.count)
    env = KSPEnv.from_config(args.config)
    entries = []
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": False,
        "scope": "initial_state_bank_smoke_test",
        "config": str(args.config),
        "task_config": str(args.task_config),
        "base_save": args.base_save,
        "seed": args.seed,
        "count": args.count,
        "debug_pre_compensation": {
            "altitude_offset_m": args.altitude_offset_m,
            "vertical_velocity_offset_m_s": args.vertical_velocity_offset_m_s,
        },
        "entries": entries,
        "errors": [],
    }

    try:
        env.connect()
    except KSPConnectionError as exc:
        print(f"NOT_READY: {exc}; load a save and start the kRPC server", file=sys.stderr)
        return 2

    conn = env.connection
    try:
        scene = _enum_name(conn.krpc.game_scene)
        if scene != "flight":
            print(f"NOT_READY: Flight scene required, current scene is {scene}", file=sys.stderr)
            return 2

        conn.space_center.load(args.base_save)
        conn.krpc.paused = True
        base_vessel = conn.space_center.active_vessel
        base_legs = _legs(base_vessel)
        report["base_legs"] = base_legs
        report["base_gear"] = bool(base_vessel.control.gear)
        if not base_legs or not all(leg["deployed"] for leg in base_legs):
            report["errors"].append(
                "base snapshot does not have all landing legs deployed; expected "
                "the gear-deployed candidate save"
            )
            print(report["errors"][-1], file=sys.stderr)
            return 2

        for index, state in enumerate(states):
            save_name = f"{args.prefix}-{args.seed}-{index:02d}"
            attempts = []
            try:
                position_offset = [0.0, 0.0, 0.0]
                info = None
                vessel = None
                flight = None
                observed_clearance = None
                observed = None
                envelope = None
                for attempt in range(4):
                    requested_position = (
                        state.position_m[0] + position_offset[0],
                        state.position_m[1] + position_offset[1],
                        state.clearance_m + args.altitude_offset_m + position_offset[2],
                    )
                    requested_velocity = (
                        state.velocity_m_s[0],
                        state.velocity_m_s[1],
                        state.velocity_m_s[2] + args.vertical_velocity_offset_m_s,
                    )
                    conn.space_center.load(args.base_save)
                    conn.krpc.paused = True
                    info = env.create_initial_save(
                        save_name,
                        position_m=requested_position,
                        velocity_m_s=requested_velocity,
                        body_z_direction=state.body_z_direction,
                        roll_deg=state.roll_deg,
                        angular_velocity_rad_s=state.angular_velocity_rad_s,
                        fill_resources=True,
                        verify=True,
                        overwrite=attempt > 0,
                        save_dir=args.save_dir,
                    )
                    vessel = conn.space_center.active_vessel
                    flight = vessel.flight(vessel.orbit.body.reference_frame)
                    observed_clearance = float(flight.surface_altitude)
                    observed = info.observed
                    envelope = _envelope_check(
                        observed, observed_clearance, scenario
                    )
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "requested_position_m": list(requested_position),
                            "observed_surface_altitude_m": observed_clearance,
                            "within_80_120_m": 80.0 <= observed_clearance <= 120.0,
                            "position_within_range": (
                                envelope["checks"]["position_x"]
                                and envelope["checks"]["position_y"]
                            ),
                            "within_frozen_envelope": envelope[
                                "within_frozen_envelope"
                            ],
                        }
                    )
                    if envelope["within_frozen_envelope"]:
                        break
                    position_offset[0] += state.position_m[0] - observed["position_m"][0]
                    position_offset[1] += state.position_m[1] - observed["position_m"][1]
                    position_offset[2] += state.clearance_m - observed_clearance
                entry = {
                    "index": index,
                    "save_name": save_name,
                    "sample": dataclasses.asdict(state),
                    "debug_request": {
                        "position_m": list(requested_position),
                        "velocity_m_s": list(requested_velocity),
                    },
                    "attempts": attempts,
                    "observed": observed,
                    "observed_surface_altitude_m": observed_clearance,
                    "observed_vertical_speed_m_s": float(flight.vertical_speed)
                    if flight is not None
                    else None,
                    "clearance_within_80_120_m": (
                        observed_clearance is not None
                        and 80.0 <= observed_clearance <= 120.0
                    ),
                    "envelope": envelope,
                    "within_frozen_envelope": bool(
                        envelope and envelope["within_frozen_envelope"]
                    ),
                    "legs": _legs(vessel) if vessel is not None else [],
                    "gear": bool(vessel.control.gear) if vessel is not None else None,
                    "verified": bool(info.verified) if info is not None else False,
                    "can_save_before": bool(info.can_save_before) if info is not None else None,
                    "can_save_after": bool(info.can_save_after) if info is not None else None,
                    "save_path": info.save_path if info is not None else None,
                    "save_sha256": info.save_sha256 if info is not None else None,
                }
            except Exception as exc:
                entry = {
                    "index": index,
                    "save_name": save_name,
                    "sample": dataclasses.asdict(state),
                    "attempts": attempts,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["errors"].append(f"{save_name}: {entry['error']}")
            entries.append(entry)
            print(
                f"[{index + 1}/{args.count}] {save_name}: "
                f"clearance={entry.get('observed_surface_altitude_m')} m, "
                f"within_80_120={entry.get('clearance_within_80_120_m')}, "
                f"within_envelope={entry.get('within_frozen_envelope')}, "
                f"legs_deployed={bool(entry.get('legs')) and all(leg['deployed'] for leg in entry['legs'])}"
            )

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

    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"JSON report: {args.output.resolve()}")
    if report["errors"]:
        print(f"completed with {len(report['errors'])} error(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
