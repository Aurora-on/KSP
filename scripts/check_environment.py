#!/usr/bin/env python3
"""Check local research dependencies without connecting to or controlling KSP.

Usage: python scripts/check_environment.py [--require-ksp] [--output reports/check.json]
KSP_PATH may name the game directory containing GameData. It is optional unless
--require-ksp is used. Exit codes: 0 = required checks passed, 1 = a check failed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


DEPENDENCIES = {
    "krpc": "krpc",
    "protobuf": "google.protobuf",
    "numpy": "numpy",
    "scipy": "scipy",
    "gymnasium": "gymnasium",
    "torch": "torch",
    "matplotlib": "matplotlib",
}


def check_python(version: tuple[int, ...]) -> dict:
    supported = (3, 12) <= version[:2] < (3, 15)
    return {"status": "PASS" if supported else "FAIL",
            "version": ".".join(map(str, version[:3])),
            "required": ">=3.12,<3.15"}


def overall_status(checks: dict) -> str:
    """Optional NOT_CHECKED/NOT_AVAILABLE/WARN results do not fail the CPU setup."""
    return "FAIL" if any(item["status"] == "FAIL" for item in checks.values()) else "PASS"


def error_summary(exc: Exception) -> str:
    # Avoid embedding personal filesystem paths in a report that might be shared.
    message = str(exc).replace(str(Path.home()), "<HOME>")
    ksp_path = os.environ.get("KSP_PATH")
    if ksp_path:
        message = message.replace(ksp_path, "<KSP_PATH>")
    return f"{type(exc).__name__}: {message}"


def check_ksp(path: str | None, required: bool = False) -> dict:
    """Inspect plugin metadata only; never load a save, start KSP, or open sockets."""
    unavailable = "FAIL" if required else "NOT_CHECKED"
    result = {"status": unavailable, "required": required, "path_configured": bool(path)}
    if not path:
        return {**result, "reason": "KSP_PATH is not set"}
    try:
        game_dir = Path(path).expanduser()
        if not game_dir.is_dir():
            return {**result, "reason": "KSP_PATH is not an existing directory"}
        metadata_file = game_dir / "GameData" / "kRPC" / "kRPC.version"
        if not metadata_file.is_file():
            return {**result, "reason": "GameData/kRPC/kRPC.version is missing"}
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))

        def version_string(value):
            if not isinstance(value, dict):
                raise ValueError("version metadata must be an object")
            numbers = [value[key] for key in ("MAJOR", "MINOR", "PATCH")]
            if not all(type(n) is int and n >= 0 for n in numbers):
                raise ValueError("version fields must be nonnegative integers")
            return ".".join(map(str, numbers))

        result.update(status="PASS", plugin_version=version_string(metadata["VERSION"]))
        if result["plugin_version"] != "0.7.0":
            return {**result, "status": "FAIL", "reason": "This project requires kRPC server 0.7.0"}
        for key in ("KSP_VERSION", "KSP_VERSION_MIN", "KSP_VERSION_MAX"):
            if key in metadata:
                result[key.lower()] = version_string(metadata[key])
        result["note"] = "Plugin metadata only; the installed game version and live connection are not verified"
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not include the path supplied by the user, even for malformed files.
        return {**result, "status": unavailable,
                "reason": f"Plugin metadata could not be read or validated ({type(exc).__name__})"}


def check_numeric_stack(modules: dict) -> dict:
    missing = [name for name in ("numpy", "scipy", "gymnasium") if name not in modules]
    if missing:
        return {"status": "NOT_CHECKED", "reason": f"imports failed: {', '.join(missing)}"}
    try:
        np = modules["numpy"]
        linalg = importlib.import_module("scipy.linalg")
        matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        target = np.array([9.0, 8.0], dtype=np.float64)
        solution = linalg.solve(matrix, target)
        residual = float(np.linalg.norm(matrix @ solution - target))
        if not np.all(np.isfinite(solution)) or not np.allclose(matrix @ solution, target):
            raise RuntimeError("NumPy/SciPy solve returned an invalid result")
        spaces = modules["gymnasium"].spaces
        box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32, seed=0)
        discrete = spaces.Discrete(3, seed=0)
        sample = box.sample()
        if not box.contains(sample) or not np.all(np.isfinite(sample)):
            raise RuntimeError("Gymnasium Box sample is incompatible with NumPy")
        if not discrete.contains(discrete.sample()):
            raise RuntimeError("Gymnasium Discrete sample failed its bounds check")
        return {"status": "PASS", "linear_solve_residual": residual,
                "box_sample_shape": list(sample.shape), "box_sample_dtype": str(sample.dtype)}
    except Exception as exc:
        return {"status": "FAIL", "reason": error_summary(exc)}


def check_torch_cpu(torch) -> dict:
    try:
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], device="cpu", dtype=torch.float32)
        weights = torch.tensor([[1.0, -2.0], [0.5, 3.0]], device="cpu",
                               dtype=torch.float32, requires_grad=True)
        prediction = x @ weights
        loss = prediction.square().mean()
        loss.backward()
        if (not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss))
                or weights.grad is None or not bool(torch.isfinite(weights.grad).all())):
            raise RuntimeError("CPU forward/backward produced nonfinite values or missing gradients")
        return {"status": "PASS", "device": "cpu", "loss": loss.item(),
                "gradient_norm": weights.grad.norm().item()}
    except Exception as exc:
        return {"status": "FAIL", "reason": error_summary(exc)}


def check_mps(torch) -> dict:
    try:
        backend = getattr(torch.backends, "mps", None)
        built = backend is not None and backend.is_built()
        available = backend is not None and backend.is_available()
        result = {"built": bool(built), "available": bool(available)}
        if not available:
            return {**result, "status": "NOT_AVAILABLE", "note": "CPU execution remains supported"}
        x = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float32)
        value = (x.square() + 1).sum().cpu().item()
        if value != 17.0:
            raise RuntimeError("MPS tensor calculation returned an unexpected result")
        return {**result, "status": "PASS", "tensor_result": value}
    except Exception as exc:
        # Accelerator support is optional; the separate CPU test remains decisive.
        return {"status": "WARN", "usable": False, "reason": error_summary(exc),
                "note": "Use CPU; MPS computation was not verified"}


def run_checks(require_ksp: bool = False) -> dict:
    checks = {"python": check_python(sys.version_info[:3])}
    modules = {}
    for distribution, module_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            modules[distribution] = module
            checks[distribution] = {"status": "PASS", "version": version}
        except Exception as exc:
            checks[distribution] = {"status": "FAIL", "reason": error_summary(exc)}
    checks["numeric_stack"] = check_numeric_stack(modules)
    if "torch" in modules:
        checks["torch_cpu"] = check_torch_cpu(modules["torch"])
        checks["mps"] = check_mps(modules["torch"])
    else:
        checks["torch_cpu"] = {"status": "NOT_CHECKED", "reason": "torch import failed"}
        checks["mps"] = {"status": "NOT_CHECKED", "reason": "torch import failed"}
    checks["ksp_files"] = check_ksp(os.environ.get("KSP_PATH"), required=require_ksp)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status(checks),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "python_implementation": platform.python_implementation()},
        "recommended_device": "mps" if checks["mps"]["status"] == "PASS" else "cpu",
        "live_ksp_connection": "NOT_CHECKED",
        "checks": checks,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ksp", action="store_true",
                        help="fail if KSP_PATH does not contain valid kRPC metadata; never connect")
    parser.add_argument("--output", type=Path, help="save JSON, e.g. reports/environment.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = run_checks(require_ksp=args.require_ksp)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
