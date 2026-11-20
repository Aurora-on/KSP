"""Rebuild the committed acceptance summary, or verify the existing one.

Run from the simulator directory after `pip install -e '.[demo]'`:

    python examples/build_validation_summary.py
    python examples/build_validation_summary.py --check

Building reruns the same scripted acceptance as
`examples/reproduce_acceptance.py`: 100 nominal test starts, zero/random
failure baselines, the one-axis physical sweep, a demo episode and a
throughput measurement. Only the compact summary is written; raw trajectories
stay in the gitignored `artifacts/`.

`--check` writes nothing. It verifies the recorded source hashes and, when the
recorded NumPy version matches the running one, regenerates the start manifests
from their recorded seed/count/split and compares their hashes. This summary is
source-domain engineering evidence, not a KSP transfer or learning result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import gymnasium
import numpy as np

from mun_lander import SimConfig
from mun_lander.evaluation import (
    benchmark,
    evaluate,
    generate_initial_states,
    parameter_sweep,
    rollout,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "validation-summary.json"
NOMINAL = {"split": "test", "root_seed": 2026, "count": 100}
SMOKE = {"split": "test", "root_seed": 2026, "count": 10}
GENERALIZATION = {"split": "generalization", "root_seed": 2026, "count": 10}
EVALUATION_FIELDS = (
    "schema_version", "controller", "split", "starts_sha256", "n_episodes",
    "safe_landing_rate", "crash_rate", "safe_in_target_rate", "status_counts",
    "transitions", "mean_landing_error_safe", "mean_main_fuel_used_safe",
    "mean_rcs_fuel_used_safe", "mean_total_fuel_used_safe",
    "safe_landing_rate_wilson95",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted((ROOT / "src").rglob("*.py"))}


def _compact(evaluation):
    return {key: evaluation[key] for key in EVALUATION_FIELDS}


def _starts(spec, manifest):
    return {**spec, "sha256": manifest["sha256"]}


def build(config, nominal_count=None):
    nominal_spec = {**NOMINAL, "count": nominal_count or NOMINAL["count"]}
    nominal_starts = generate_initial_states(config, nominal_spec["count"],
                                             nominal_spec["root_seed"], nominal_spec["split"])
    nominal = evaluate(config, nominal_starts)
    smoke_starts = generate_initial_states(config, SMOKE["count"], SMOKE["root_seed"], SMOKE["split"])
    failure_checks = {}
    for controller in ("zero", "random"):
        failure = evaluate(config, smoke_starts, controller)
        failure_checks[controller] = {
            "n_episodes": failure["n_episodes"],
            "status_counts": failure["status_counts"],
            "starts_sha256": failure["starts_sha256"],
            "safe_landing_rate": failure["safe_landing_rate"],
        }
    generalization_starts = generate_initial_states(
        config, GENERALIZATION["count"], GENERALIZATION["root_seed"], GENERALIZATION["split"])
    sweep = parameter_sweep(config, generalization_starts, scales=(0.8, 1.0, 1.2))
    runs = [{"parameter": run["parameter"], "scale": run["scale"],
             "n_episodes": run["evaluation"]["n_episodes"],
             "safe_landing_rate": run["evaluation"]["safe_landing_rate"],
             "safe_in_target_rate": run["evaluation"]["safe_in_target_rate"],
             "status_counts": run["evaluation"]["status_counts"],
             "starts_sha256": run["evaluation"]["starts_sha256"],
             "mean_landing_error_safe": run["evaluation"]["mean_landing_error_safe"],
             "mean_total_fuel_used_safe": run["evaluation"]["mean_total_fuel_used_safe"]}
            for run in sweep["runs"]]
    measurement = benchmark(config, steps=2000, seed=7)
    demo, _ = rollout(config, seed=7)
    demo = {key: demo[key] for key in ("status", "is_success", "safe_in_target", "time",
                                       "landing_error", "horizontal_error", "main_fuel_used",
                                       "rcs_fuel_used", "total_fuel_used", "touchdown",
                                       "last_touchdown", "touchdown_count", "stable_time",
                                       "return", "transitions")}
    return {
        "schema_version": 2,
        "validated_on": date.today().isoformat(),
        "scope": "Source simulator pilot; no KSP alignment or learned policy results",
        "stability_window_s": config.landing.stability_time,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "gymnasium": gymnasium.__version__},
        "nominal_config": config.to_dict(),
        "starts": {"nominal": _starts(nominal_spec, nominal_starts),
                   "smoke": _starts(SMOKE, smoke_starts),
                   "generalization": _starts(GENERALIZATION, generalization_starts)},
        "nominal": _compact(nominal),
        "failure_checks": failure_checks,
        "generalization": {
            "unique_physical_conditions": sweep["unique_evaluation_runs"],
            "starts_per_condition": GENERALIZATION["count"],
            "scales": sweep["scales"],
            "fixed_nominal_controller": True,
            "runs": runs,
        },
        "benchmark": measurement,
        "demo": demo,
        "reproduce_command": "cd simulator && python examples/build_validation_summary.py",
        "source_sha256": _source_hashes(),
    }


def check(path):
    """Return a list of problems; regenerate manifests only if NumPy matches."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if summary.get("schema_version") != 2:
        problems.append(f"unexpected schema_version {summary.get('schema_version')}; rebuild the summary")
        return problems
    for relative, recorded in summary.get("source_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file():
            problems.append(f"missing source file {relative}")
        elif _sha256(source) != recorded:
            problems.append(f"source hash mismatch: {relative}")
    config = SimConfig.from_dict(summary["nominal_config"])
    if summary["stability_window_s"] != config.landing.stability_time:
        problems.append("stability_window_s disagrees with nominal_config")
    environment = summary.get("environment", {})
    if environment.get("numpy") != np.__version__:
        print(f"note: start-manifest hashes not checked; NumPy {np.__version__} "
              f"differs from recorded {environment.get('numpy')}", file=sys.stderr)
        return problems
    for key, spec in summary["starts"].items():
        manifest = generate_initial_states(config, spec["count"], spec["root_seed"], spec["split"])
        if manifest["sha256"] != spec["sha256"]:
            problems.append(f"start manifest hash mismatch: {key}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build or verify simulator acceptance evidence")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="verify the existing summary without running acceptance")
    parser.add_argument("--nominal-count", type=int, default=NOMINAL["count"])
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.output)
        for problem in problems:
            print(problem, file=sys.stderr)
        print("OK" if not problems else "FAILED")
        return 1 if problems else 0
    summary = build(SimConfig(), args.nominal_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "schema_version": summary["schema_version"],
                      "safe_landing_rate": summary["nominal"]["safe_landing_rate"],
                      "safe_in_target_rate": summary["nominal"]["safe_in_target_rate"],
                      "transitions_per_second": summary["benchmark"]["transitions_per_second"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
