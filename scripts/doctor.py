#!/usr/bin/env python3
"""Check the local KSP/kRPC environment without changing game state by default.

Run with .venv/bin/python scripts/doctor.py --output reports/doctor.json.
--control-test explicitly enables a reversible lights test in the Flight scene.
Exit codes: 0 PASS, 1 FAIL, 2 NOT_READY (未就绪).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def bounded_connect(krpc, args):
    """Keep an I/O deadline as well as the public API's connection deadline.

    kRPC 0.7.0 Connection._open clears its socket timeout after TCP connects.
    This process-local wrapper also bounds the handshake and subsequent RPCs.
    The installed package and the class method are not permanently modified.
    """
    from krpc.connection import Connection

    original = Connection._open

    def open_socket(connection):
        sock = original(connection)
        sock.settimeout(args.timeout)
        return sock

    Connection._open = open_socket
    try:
        return krpc.connect(
            name="KSP environment doctor",
            address=args.address,
            rpc_port=args.rpc_port,
            stream_port=args.stream_port,
            timeout=args.timeout,
        )
    finally:
        Connection._open = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--stream-port", type=int, default=50001)
    parser.add_argument("--timeout", type=positive, default=10.0,
                        help="TCP/RPC and stream-check timeout, seconds (default: 10)")
    parser.add_argument("--output", type=Path, help="write the complete report as JSON")
    parser.add_argument("--control-test", action="store_true",
                        help="explicitly toggle and restore vessel lights, Flight only")
    args = parser.parse_args()
    if not all(1 <= p <= 65535 for p in (args.rpc_port, args.stream_port)):
        parser.error("ports must be between 1 and 65535")

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_READY",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "endpoint": {"address": args.address, "rpc_port": args.rpc_port,
                     "stream_port": args.stream_port},
        "control_test_requested": args.control_test,
        "checks": [],
        "flight_validation": {"status": "SKIP", "reason": "not checked yet"},
    }

    def check(name, status, detail=None):
        report["checks"].append({"name": name, "status": status, "detail": detail})

    conn = None
    krpc = None
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python >= 3.10 is required")
        import krpc as krpc_module
        from google.protobuf import __version__ as protobuf_version

        krpc = krpc_module
        version = importlib.metadata.version("krpc")
        report["packages"] = {"krpc": version, "protobuf": protobuf_version}
        if version != "0.7.0":
            raise RuntimeError(f"These scripts target krpc 0.7.0; found {version}")
        check("dependencies", "PASS", report["packages"])
    except Exception as exc:
        check("dependencies", "FAIL", f"{type(exc).__name__}: {exc}")

    if krpc is not None and not any(c["status"] == "FAIL" for c in report["checks"]):
        for name, port in (("rpc_port", args.rpc_port), ("stream_port", args.stream_port)):
            try:
                with socket.create_connection((args.address, port), timeout=args.timeout):
                    pass
                check(name, "PASS", f"TCP reachable at {args.address}:{port}")
            except OSError as exc:
                check(name, "NOT_READY", f"{exc}; load a save and start the kRPC server")

        if all(c["status"] == "PASS" for c in report["checks"]):
            try:
                conn = bounded_connect(krpc, args)
                status = conn.krpc.get_status()
                check("rpc_handshake", "PASS", {"server_version": status.version})
                if status.version != "0.7.0":
                    check("server_version", "FAIL", f"expected 0.7.0, got {status.version}")
                else:
                    check("server_version", "PASS", status.version)
                services = sorted(s.name for s in conn.krpc.get_services().services)
                check("services", "PASS" if {"KRPC", "SpaceCenter"} <= set(services)
                      else "FAIL", services)
                scene = conn.krpc.game_scene.name
                paused = conn.krpc.paused
                check("game_scene", "PASS", {"scene": scene, "paused": paused})
                bodies = sorted(conn.space_center.bodies)
                check("celestial_bodies", "PASS" if bodies else "FAIL", bodies)
                ut = conn.space_center.ut
                check("universal_time", "PASS" if math.isfinite(ut) else "FAIL",
                      ut if math.isfinite(ut) else repr(ut))

                # Callbacks provide a bounded wait for the first stream value;
                # Stream.__call__ itself can otherwise wait indefinitely.
                samples = []
                errors = []
                sample_lock = threading.Lock()
                ready = threading.Event()

                def on_sample(value):
                    with sample_lock:
                        samples.append(float(value))
                        if paused or len(samples) >= 3:
                            ready.set()

                def on_end(error):
                    with sample_lock:
                        errors.append(f"{type(error).__name__}: {error}")
                    ready.set()

                stream = conn.add_stream(getattr, conn.space_center, "ut")
                try:
                    stream.rate = 5
                    stream.add_callback(on_sample)
                    stream.add_end_callback(on_end)
                    ready.wait(args.timeout)
                    with sample_lock:
                        received, stream_errors = list(samples), list(errors)
                    if stream_errors:
                        check("ut_stream", "FAIL", {"errors": stream_errors})
                    elif not received:
                        check("ut_stream", "FAIL", "no stream value before timeout")
                    elif not all(math.isfinite(v) for v in received):
                        check("ut_stream", "FAIL", {"samples": [repr(v) for v in received]})
                    else:
                        check("ut_stream", "PASS", {"samples": received, "requested_hz": 5})
                        advanced = len(received) >= 2 and received[-1] > received[0]
                        check("simulation_advancement", "PASS" if advanced else "NOT_READY",
                              {"advanced": advanced, "paused": paused,
                               "note": "pause is never changed by this script"})
                finally:
                    stream.remove()

                if scene != "flight":
                    report["flight_validation"] = {
                        "status": "SKIP", "reason": f"scene is {scene}; Flight required"
                    }
                    if args.control_test:
                        check("lights_control", "NOT_READY", "enter a vessel on the launchpad/runway")
                elif not args.control_test:
                    report["flight_validation"] = {
                        "status": "SKIP", "reason": "control test requires --control-test"
                    }
                else:
                    vessel = conn.space_center.active_vessel
                    if vessel is None:
                        report["flight_validation"] = {"status": "NOT_READY", "reason": "no active vessel"}
                        check("lights_control", "NOT_READY", "no active vessel")
                    else:
                        report["flight_validation"] = {"status": "FAIL", "reason": "control test did not finish"}
                        control = vessel.control
                        original_lights = control.lights
                        detail = {"vessel": vessel.name, "before": original_lights}
                        try:
                            # Restore even when setting or reading the test state fails.
                            try:
                                control.lights = not original_lights
                                detail["after_toggle"] = control.lights
                                if detail["after_toggle"] != (not original_lights):
                                    raise RuntimeError("lights did not change to the requested state")
                            finally:
                                control.lights = original_lights
                                detail["restored"] = control.lights
                                if detail["restored"] != original_lights:
                                    raise RuntimeError("lights restoration could not be verified")
                        except Exception as exc:
                            detail["error"] = f"{type(exc).__name__}: {exc}"
                            report["flight_validation"] = {"status": "FAIL", **detail}
                            check("lights_control", "FAIL", detail)
                        else:
                            report["flight_validation"] = {"status": "PASS", **detail}
                            check("lights_control", "PASS", detail)
            except TimeoutError as exc:
                check("runtime", "NOT_READY", f"timed out: {exc}; ensure the game/server is responding")
            except Exception as exc:
                check("runtime", "FAIL", f"{type(exc).__name__}: {exc}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception as exc:
                        check("connection_close", "FAIL", f"{type(exc).__name__}: {exc}")

    statuses = {c["status"] for c in report["checks"]}
    report["status"] = "FAIL" if "FAIL" in statuses else "NOT_READY" if "NOT_READY" in statuses else "PASS"
    if report["flight_validation"].get("reason") == "not checked yet":
        report["flight_validation"]["reason"] = "connection checks did not reach a usable game scene"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    label = "未就绪 (NOT_READY)" if report["status"] == "NOT_READY" else report["status"]
    print(f"KSP environment: {label}")
    for result in report["checks"]:
        print(f"[{result['status']}] {result['name']}: {result['detail']}")
    print("Flight validation:", report["flight_validation"])
    if args.output:
        print("JSON report:", args.output.resolve())
    return {"PASS": 0, "FAIL": 1, "NOT_READY": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
