#!/usr/bin/env python3
"""Probe main-engine and RCS response with short, reversible control pulses.

The script is read-only unless --execute is given. It holds physics ticks,
applies a short throttle pulse and short RCS pulses, samples thrust/angular
velocity, then restores the original control toggles and zeroes throttle.
It does not stage, save, load or change the orbit intentionally.

Example:
    .venv/bin/python scripts/probe_controls.py \
        --config configs/ksp_mun_test.toml --execute \
        --output reports/control-response.json
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


def _vector(value) -> list[float]:
    return [float(item) for item in value]


def _norm(value) -> float:
    return math.sqrt(sum(item * item for item in value))


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _snapshot(control) -> dict:    return {
        "throttle": float(control.throttle),
        "rcs": bool(control.rcs),
        "sas": bool(control.sas),
        "reaction_wheels": bool(control.reaction_wheels),
        "input_mode": control.input_mode,
        "input_mode_name": _enum_name(control.input_mode),
        "pitch": float(control.pitch),
        "yaw": float(control.yaw),
        "roll": float(control.roll),
        "forward": float(control.forward),
        "right": float(control.right),
        "up": float(control.up),
    }


def _restore(control, snapshot: dict) -> None:
    # Keep throttle at zero for safety even if it was nonzero before the probe.
    control.throttle = 0.0
    control.pitch = 0.0
    control.yaw = 0.0
    control.roll = 0.0
    control.forward = 0.0
    control.right = 0.0
    control.up = 0.0
    control.rcs = snapshot["rcs"]
    control.sas = snapshot["sas"]
    control.reaction_wheels = snapshot["reaction_wheels"]
    control.input_mode = snapshot["input_mode"]


class HeldTick:
    """Hold one physics tick and advance it explicitly."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.held = False

    def __enter__(self) -> "HeldTick":
        self.conn.krpc.hold_tick()
        self.held = True
        return self

    def next(self) -> None:
        self.conn.krpc.next_tick()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.held:
            try:
                self.conn.krpc.release_tick()
            except Exception:
                pass


def stage_once(conn, vessel, timeout_s: float = 3.0) -> dict:
    """Activate exactly one stage and report whether an engine became active.

    This is deliberately a single activation. It never loops through stages,
    because a second activation could fire separation motors or decouplers.
    """

    control = vessel.control
    before_stage = int(control.current_stage)
    before_active = sum(1 for engine in vessel.parts.engines if engine.active)
    control.activate_next_stage()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(engine.active for engine in vessel.parts.engines):
            break
        time.sleep(0.1)
    after_active = sum(1 for engine in vessel.parts.engines if engine.active)
    return {
        "before_stage": before_stage,
        "after_stage": int(control.current_stage),
        "before_active_engines": before_active,
        "after_active_engines": after_active,
    }


def probe_engine(conn, vessel, throttle: float, ticks: int) -> dict:
    active = [engine for engine in vessel.parts.engines if engine.active]
    if not active:
        return {
            "skipped": "no active engine; stage or ignite the intended engine first",
            "engines": [engine.part.name for engine in vessel.parts.engines],
        }

    resource_names = sorted(
        {name for engine in active for name in engine.propellant_names}
    )
    before = {
        "mass_kg": float(vessel.mass),
        "resources": {name: float(vessel.resources.amount(name)) for name in resource_names},
        "thrust_n": float(vessel.thrust),
        "engines": [
            {
                "part_name": engine.part.name,
                "active": bool(engine.active),
                "has_fuel": bool(engine.has_fuel),
                "throttle_locked": bool(engine.throttle_locked),
                "max_thrust_n": float(engine.max_thrust),
            }
            for engine in active
        ],
    }

    samples = []
    zero_samples = []
    with HeldTick(conn) as tick:
        vessel.control.throttle = throttle
        for _ in range(ticks):
            tick.next()
            samples.append(
                {
                    "thrust_n": float(vessel.thrust),
                    "throttle_applied": float(vessel.control.throttle),
                    "engine_thrust_n": [float(engine.thrust) for engine in active],
                }
            )
        vessel.control.throttle = 0.0
        # Engine thrust can persist for a tick after the throttle command. Let
        # it settle before reporting the zero-throttle state.
        for _ in range(3):
            tick.next()
            zero_samples.append(
                {
                    "thrust_n": float(vessel.thrust),
                    "throttle_applied": float(vessel.control.throttle),
                }
            )
        after_zero_thrust_n = zero_samples[-1]["thrust_n"]

    after = {
        "mass_kg": float(vessel.mass),
        "resources": {name: float(vessel.resources.amount(name)) for name in resource_names},
        "thrust_n": float(vessel.thrust),
    }
    return {
        "command_throttle": throttle,
        "ticks": ticks,
        "before": before,
        "samples": samples,
        "zero_throttle_samples": zero_samples,
        "after_zero_throttle_thrust_n": after_zero_thrust_n,
        "after": after,
        "mass_delta_kg": after["mass_kg"] - before["mass_kg"],
        "resource_delta": {
            name: after["resources"][name] - before["resources"][name]
            for name in resource_names
        },
        "peak_thrust_n": max((sample["thrust_n"] for sample in samples), default=0.0),
    }


def probe_rcs(conn, vessel, axis: str, magnitude: float, ticks: int) -> dict:
    rcs_parts = list(vessel.parts.rcs)
    if not rcs_parts:
        return {"skipped": "no RCS parts on the vessel"}
    if not any(part.has_fuel for part in rcs_parts):
        return {"skipped": "RCS parts have no fuel"}

    frame = vessel.orbit.body.non_rotating_reference_frame
    before = _vector(vessel.angular_velocity(frame))
    propellant_names = sorted(
        {
            name
            for part in rcs_parts
            for name in part.propellants
            if name in vessel.resources.names
        }
    )
    before_resources = {
        name: float(vessel.resources.amount(name)) for name in propellant_names
    }
    samples = []
    counter_samples = []

    with HeldTick(conn) as tick:
        control = vessel.control
        # Imported lazily so the script does not require kRPC for --help.
        from krpc.services.spacecenter import ControlInputMode

        control.input_mode = ControlInputMode.override
        control.sas = False
        control.reaction_wheels = False
        control.rcs = True
        control.throttle = 0.0
        control.forward = 0.0
        control.right = 0.0
        control.up = 0.0
        setattr(control, axis, magnitude)

        for _ in range(ticks):
            tick.next()
            velocity = _vector(vessel.angular_velocity(frame))
            samples.append(
                {
                    "input": float(getattr(control, axis)),
                    "angular_velocity_rad_s": velocity,
                    "magnitude_rad_s": _norm(velocity),
                }
            )

        setattr(control, axis, -magnitude)
        for _ in range(ticks):
            tick.next()
            velocity = _vector(vessel.angular_velocity(frame))
            counter_samples.append(
                {
                    "input": float(getattr(control, axis)),
                    "angular_velocity_rad_s": velocity,
                    "magnitude_rad_s": _norm(velocity),
                }
            )

        setattr(control, axis, 0.0)
        tick.next()
        during_zero = _vector(vessel.angular_velocity(frame))

    after = _vector(vessel.angular_velocity(frame))
    after_resources = {
        name: float(vessel.resources.amount(name)) for name in propellant_names
    }
    resource_delta = {
        name: after_resources[name] - before_resources[name]
        for name in propellant_names
    }
    peak = max(
        (sample["magnitude_rad_s"] for sample in samples + counter_samples),
        default=0.0,
    )
    return {
        "axis": axis,
        "command": magnitude,
        "ticks": ticks,
        "before_angular_velocity_rad_s": before,
        "pulse_samples": samples,
        "counter_pulse_samples": counter_samples,
        "after_zero_angular_velocity_rad_s": during_zero,
        "after_release_angular_velocity_rad_s": after,
        "peak_angular_speed_rad_s": peak,
        "delta_angular_velocity_rad_s": [
            after[i] - before[i] for i in range(3)
        ],
        "rcs_parts": [part.part.name for part in rcs_parts],
        "rcs_propellants": propellant_names,
        "before_resources": before_resources,
        "after_resources": after_resources,
        "resource_delta": resource_delta,
        "monopropellant_delta": resource_delta.get("MonoPropellant"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ksp_mun_test.toml")
    parser.add_argument("--engine-throttle", type=float, default=0.05)
    parser.add_argument("--engine-ticks", type=int, default=5)
    parser.add_argument("--rcs-input", type=float, default=0.5)
    parser.add_argument("--rcs-ticks", type=int, default=10)
    parser.add_argument("--axis", choices=("pitch", "yaw", "roll", "all"), default="all")
    parser.add_argument("--execute", action="store_true",
                        help="send short control pulses; without this flag only read state")
    parser.add_argument("--stage-once", action="store_true",
                        help="activate exactly one stage if no engine is active (requires --execute)")
    parser.add_argument("--output", type=Path, help="write the complete report as JSON")
    args = parser.parse_args()

    if args.stage_once and not args.execute:
        parser.error("--stage-once requires --execute")
    if not 0.0 < args.engine_throttle <= 1.0:
        parser.error("--engine-throttle must be in (0, 1]")
    if args.engine_ticks < 1 or args.rcs_ticks < 1:
        parser.error("tick counts must be positive")
    if not 0.0 < args.rcs_input <= 1.0:
        parser.error("--rcs-input must be in (0, 1]")
    axes = ("pitch", "yaw", "roll") if args.axis == "all" else (args.axis,)

    env = KSPEnv.from_config(args.config)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "execute": args.execute,
        "scene": None,
        "vessel": None,
        "body": None,
        "baseline": None,
        "staging": None,
        "engine": None,
        "rcs": {},
        "rcs_total_resource_delta": {},
        "restored_control": None,
    }

    try:
        env.connect()
    except KSPConnectionError as exc:
        print(f"NOT_READY: {exc}; load a save and start the kRPC server", file=sys.stderr)
        return 2

    try:
        conn = env.connection
        scene = _enum_name(conn.krpc.game_scene)
        report["scene"] = scene
        if scene != "flight":
            print(f"NOT_READY: Flight scene required, current scene is {scene}", file=sys.stderr)
            return 2

        vessel = conn.space_center.active_vessel
        if vessel is None:
            print("NOT_READY: no active vessel", file=sys.stderr)
            return 2
        body = vessel.orbit.body
        report["vessel"] = vessel.name
        report["body"] = body.name
        report["baseline"] = env.read_diagnostic_state()

        print(f"scene: {scene}")
        print(f"vessel: {vessel.name} around {body.name}")
        print(f"throttle: {vessel.control.throttle}, RCS: {vessel.control.rcs}, "
              f"SAS: {vessel.control.sas}, stage: {vessel.control.current_stage}")
        print(f"active engines: {sum(1 for engine in vessel.parts.engines if engine.active)}"
              f" / {len(vessel.parts.engines)}")

        if not args.execute:
            print("dry run: no control commands sent; pass --execute to probe")
            return 0

        active_before = sum(1 for engine in vessel.parts.engines if engine.active)
        if args.stage_once and active_before == 0:
            print("staging once to activate the next stage...")
            report["staging"] = stage_once(conn, vessel)
            print(f"stage: {report['staging']['before_stage']} -> "
                  f"{report['staging']['after_stage']}; active engines: "
                  f"{report['staging']['before_active_engines']} -> "
                  f"{report['staging']['after_active_engines']}")
        else:
            report["staging"] = {
                "skipped": "no staging requested" if not args.stage_once else "an engine is already active"
            }

        control = vessel.control
        snapshot = _snapshot(control)
        try:
            print("engine probe...")
            report["engine"] = probe_engine(
                conn, vessel, args.engine_throttle, args.engine_ticks
            )
            engine_report = report["engine"]
            if "skipped" in engine_report:
                print(f"engine skipped: {engine_report['skipped']}")
            else:
                print(f"engine peak thrust: {engine_report['peak_thrust_n']:.1f} N; "
                      f"mass delta: {engine_report['mass_delta_kg']:.4f} kg; "
                      f"resource delta: {engine_report['resource_delta']}")

            for axis in axes:
                print(f"RCS {axis} probe...")
                result = probe_rcs(conn, vessel, axis, args.rcs_input, args.rcs_ticks)
                report["rcs"][axis] = result
                if "skipped" in result:
                    print(f"  skipped: {result['skipped']}")
                else:
                    print(f"  peak angular speed: {result['peak_angular_speed_rad_s']:.6f} rad/s; "
                          f"after: {_norm(result['after_release_angular_velocity_rad_s']):.6f} rad/s; "
                          f"resource delta: {result['resource_delta']}")

            rcs_total = {}
            for result in report["rcs"].values():
                for name, delta in result.get("resource_delta", {}).items():
                    rcs_total[name] = rcs_total.get(name, 0.0) + delta
            report["rcs_total_resource_delta"] = rcs_total
            if rcs_total:
                print(f"RCS total resource delta: {rcs_total}")
            if "MonoPropellant" in rcs_total:
                print(f"Monopropellant consumed: {-rcs_total['MonoPropellant']:.6f} units")
        finally:
            _restore(control, snapshot)
            report["restored_control"] = {
                "throttle": float(control.throttle),
                "rcs": bool(control.rcs),
                "sas": bool(control.sas),
                "reaction_wheels": bool(control.reaction_wheels),
                "input_mode": _enum_name(control.input_mode),
            }

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
