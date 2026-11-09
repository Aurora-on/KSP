#!/usr/bin/env python3
"""Read KSP flight telemetry at up to 5 Hz; never send flight controls.

Example: .venv/bin/python scripts/telemetry.py --seconds 10 --output reports/telemetry.csv
Enter a vessel on the launchpad/runway first. CSV goes to stdout if --output is
omitted. Fields use SI units and degrees; speed is relative to the rotating body.
Streams are sampled together locally, not guaranteed to share one physics tick.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def bounded_connect(krpc, args):
    # kRPC 0.7.0 clears its public connect timeout after TCP establishment.
    # Retain a per-socket I/O deadline for handshake and RPC calls, in this
    # process only. Do not modify installed library files.
    from krpc.connection import Connection

    original = Connection._open

    def open_socket(connection):
        sock = original(connection)
        sock.settimeout(args.timeout)
        return sock

    Connection._open = open_socket
    try:
        return krpc.connect(name="KSP read-only telemetry", address=args.address,
                            rpc_port=args.rpc_port, stream_port=args.stream_port,
                            timeout=args.timeout)
    finally:
        Connection._open = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--stream-port", type=int, default=50001)
    parser.add_argument("--timeout", type=positive, default=10.0)
    parser.add_argument("--seconds", type=positive, default=10.0)
    parser.add_argument("--output", type=Path, help="write CSV here (default: stdout)")
    args = parser.parse_args()
    if not all(1 <= p <= 65535 for p in (args.rpc_port, args.stream_port)):
        parser.error("ports must be between 1 and 65535")

    conn = None
    output = None
    stream_errors = {}
    values = {}
    updated = {}
    pending = set()
    streams = []
    lock = threading.Lock()
    condition = threading.Condition(lock)
    rows = 0
    try:
        import krpc

        version = importlib.metadata.version("krpc")
        if version != "0.7.0":
            raise RuntimeError(f"This script targets krpc 0.7.0; found {version}")
        conn = bounded_connect(krpc, args)
        scene = conn.krpc.game_scene.name
        if scene != "flight":
            print(f"NOT_READY (未就绪): scene={scene}; enter a vessel on the launchpad/runway.", file=sys.stderr)
            return 2
        vessel = conn.space_center.active_vessel
        if vessel is None:
            print("NOT_READY (未就绪): no active vessel.", file=sys.stderr)
            return 2
        body = vessel.orbit.body
        # The default flight frame moves with the vessel and gives speed=0.
        flight = vessel.flight(body.reference_frame)
        fields = {
            "ut_s": (conn.space_center, "ut"),
            "mass_kg": (vessel, "mass"),
            "mean_altitude_m": (flight, "mean_altitude"),
            "surface_altitude_m": (flight, "surface_altitude"),
            "surface_speed_m_s": (flight, "speed"),
            "latitude_deg": (flight, "latitude"),
            "longitude_deg": (flight, "longitude"),
            "vertical_speed_m_s": (flight, "vertical_speed"),
        }
        identity = {"vessel_name": vessel.name, "body_name": body.name}
        pending.update(fields)

        def receiver(key):
            def receive(value):
                with condition:
                    values[key] = value
                    updated[key] = time.monotonic()
                    pending.discard(key)
                    condition.notify_all()
            return receive

        def end_receiver(key):
            def ended(error):
                with condition:
                    stream_errors[key] = f"{type(error).__name__}: {error}"
                    pending.discard(key)
                    condition.notify_all()
            return ended

        for key, (obj, attribute) in fields.items():
            try:
                stream = conn.add_stream(getattr, obj, attribute)
                streams.append(stream)
                stream.rate = 5
                stream.add_callback(receiver(key))
                stream.add_end_callback(end_receiver(key))
            except Exception as exc:
                with condition:
                    stream_errors[key] = f"{type(exc).__name__}: {exc}"
                    pending.discard(key)

        deadline = time.monotonic() + args.timeout
        with condition:
            while pending and time.monotonic() < deadline:
                condition.wait(max(0, deadline - time.monotonic()))
            for key in pending:
                stream_errors[key] = "no initial value before timeout"
            if not values:
                raise RuntimeError(f"No telemetry values received: {stream_errors}")
        if stream_errors:
            print(f"Partial telemetry; unavailable fields are blank: {stream_errors}", file=sys.stderr)
        paused = conn.krpc.paused
        if paused:
            print("Game is paused; values may stay constant. Pause is left unchanged.", file=sys.stderr)

        names = ["recorded_at_utc", "elapsed_s", "vessel_name", "body_name", *fields,
                 "max_stream_age_s", "unavailable_fields"]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("w", encoding="utf-8", newline="")
        else:
            output = sys.stdout
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        start = time.monotonic()
        next_sample = start
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            with lock:
                row = {key: values.get(key, "") if key not in stream_errors else "" for key in fields}
                ages = [now - stamp for key, stamp in updated.items() if key not in stream_errors]
                missing = sorted(set(fields) - values.keys() | stream_errors.keys())
            row.update(identity)
            row.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_s=round(now - start, 3),
                       max_stream_age_s=round(max(ages), 3) if ages else "",
                       unavailable_fields=";".join(missing))
            writer.writerow(row)
            rows += 1
            output.flush()
            next_sample += 0.2
            time.sleep(max(0, min(next_sample, start + args.seconds) - time.monotonic()))
        print(f"Recorded {rows} rows; fields report their latest streamed values.", file=sys.stderr)
        if args.output:
            print(f"CSV: {args.output.resolve()}", file=sys.stderr)
        if stream_errors:
            print(f"Incomplete telemetry: {stream_errors}", file=sys.stderr)
            return 2
        return 0
    except KeyboardInterrupt:
        print(f"Stopped; kept {rows} recorded rows.", file=sys.stderr)
        return 130
    except (ConnectionRefusedError, TimeoutError) as exc:
        print(f"NOT_READY (未就绪): {exc}; load a save and start the kRPC server.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            # Closing removes all client streams. No flight controls are changed.
            try:
                conn.close()
            except Exception as exc:
                print(f"Connection cleanup: {exc}", file=sys.stderr)
        if output is not None and output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
