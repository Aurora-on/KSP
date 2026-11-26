"""Evaluate frozen controllers on paired nominal-simulator validation starts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
from math import sqrt
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from mun_agent.dataset import file_hash
from mun_agent.nominal import load_manifest
from mun_agent.policy import load_policy
from mun_agent.transfer import rescale_fuel
from mun_lander import SimConfig
from mun_lander.controller import ScriptedController
from mun_lander.evaluation import rollout


A_PILOT = 9000 / 1800
A_KSP = 240000 / 11460
A_LIM = 60000 / 11460
ALPHA_EST = 0.07
SOURCE_FUEL_KG = 600.0
TARGET_FUEL_KG = 5000.0


@dataclass(frozen=True)
class Condition:
    name: str
    category: str = "physical"
    inertia: float | tuple[float, float, float] = 1.0
    thrust: float = 1.0
    torque: tuple[float, float, float] = (1.0, 1.0, 1.0)
    delay_steps: int = 0
    fuel_kg: bool = False
    normalize_fuel: bool = False
    z_offset_m: float = 0.0

    def config(self):
        cfg = SimConfig()
        if self.thrust != 1:
            cfg = cfg.perturbed("thrust", self.thrust)
        if isinstance(self.inertia, tuple):
            moments = tuple(x * k for x, k in zip(cfg.vehicle.inertia, self.inertia, strict=True))
            cfg = replace(cfg, vehicle=replace(cfg.vehicle, inertia=moments))
        elif self.inertia != 1:
            cfg = cfg.perturbed("inertia", self.inertia)
        if self.torque != (1., 1., 1.):
            torques = tuple(x * k for x, k in zip(cfg.vehicle.rcs_torque, self.torque, strict=True))
            cfg = replace(cfg, vehicle=replace(cfg.vehicle, rcs_torque=torques))
        return cfg


def conditions():
    full = A_KSP / A_PILOT
    limited = A_LIM / A_PILOT
    items = [Condition("nominal")]
    items.extend(Condition(f"inertia_{scale:g}", inertia=scale) for scale in (1.25, 1.5, 2, 2.5, 3, 4, 6))
    items.extend([
        Condition("inertia_alpha_est", inertia=0.75 / ALPHA_EST),
        Condition("inertia_pitch_yaw", inertia=(4., 4., 1.)),
        # Roll-only x4 exceeds the triangle bound (3200 > 1200+1200).
        # The maximal admissible inertia and the quarter-torque proxy are distinct plants.
        Condition("inertia_roll_3", inertia=(1., 1., 3.)),
        Condition("torque_roll_quarter", torque=(1., 1., 0.25)),
    ])
    items.extend(Condition(f"thrust_{scale:g}", thrust=scale) for scale in (0.9, 1.2, 2.))
    for label, scale in (("ksp_limited", limited), ("ksp_full", full)):
        items.append(Condition(f"thrust_{label}", thrust=scale))
        for inertia in (2, 4):
            items.append(Condition(f"thrust_{label}_inertia_{inertia}", thrust=scale, inertia=inertia))
    for label, scale in (("nominal", 1.), ("ksp_limited", limited), ("ksp_full", full)):
        items.append(Condition(f"delay1_{label}", "delay", thrust=scale, delay_steps=1))
    items.extend([Condition("fuel_kg", "observation", fuel_kg=True),
                  Condition("fuel_kg_rescaled", "observation", fuel_kg=True, normalize_fuel=True)])
    items.extend(Condition(f"z_offset_{z:+.1f}", "observation", z_offset_m=z)
                 for z in (-.5, -.3, -.1, .1, .3, .5))
    return {condition.name: condition for condition in items}


class OneStepDelay:
    """Delay native commands by one control interval, starting at zero throttle and torque."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = np.zeros(4, dtype=np.float32)

    def __call__(self, action):
        previous = self.previous
        self.previous = np.array(action, copy=True)
        return previous


class _PerturbedController:
    def __init__(self, controller, condition):
        self.controller = controller
        self.condition = condition
        self.delay = OneStepDelay() if condition.delay_steps else None

    def reset(self):
        self.controller.reset()
        if self.delay is not None:
            self.delay.reset()

    def __call__(self, obs):
        # rollout supplies one private observation copy per call; the simulator
        # still receives the original measured state when computing reward and events.
        if self.condition.fuel_kg:
            obs[13] *= TARGET_FUEL_KG / SOURCE_FUEL_KG
            if self.condition.normalize_fuel:
                obs = rescale_fuel(obs, source_capacity_kg=SOURCE_FUEL_KG, target_capacity_kg=TARGET_FUEL_KG)
        if self.condition.z_offset_m:
            obs[2] += self.condition.z_offset_m
        action = self.controller(obs)
        return self.delay(action) if self.delay is not None else action


def evaluate_pair(condition, controller_name, starts, online_path, bc_path):
    """Return one deterministic row per supplied validation entry in manifest order."""
    torch.set_num_threads(1)
    cfg = condition.config()
    if controller_name == "online":
        controller = load_policy(online_path, device="cpu")
    elif controller_name == "bc_only":
        controller = load_policy(bc_path, device="cpu")
    elif controller_name == "scripted_nominal":
        controller = ScriptedController(SimConfig())
    elif controller_name == "scripted_privileged" and condition.category != "observation":
        controller = ScriptedController(cfg)
    else:
        raise ValueError(f"unsupported controller {controller_name!r} for {condition.name}")
    if condition.category in ("observation", "delay"):
        controller = _PerturbedController(controller, condition)

    episodes = []
    for entry in starts:
        result, trace = rollout(cfg, entry["initial_state"], entry["seed"], controller, record=True)
        applied_rcs = trace["actions"][:, 1:]
        saturated = int((np.abs(applied_rcs) >= 0.99).any(axis=1).sum())
        touchdown = result["touchdown"]
        episodes.append({
            "start_id": entry["id"], "seed": entry["seed"],
            "status": result["status"], "is_success": result["is_success"], "safe_in_target": result["safe_in_target"],
            "touchdown_speed_m_s": touchdown["speed"] if touchdown is not None else None,
            "touchdown_tilt_deg": touchdown["tilt_deg"] if touchdown is not None else None,
            "landing_error_m": result["landing_error"], "return": result["return"],
            "main_fuel_used_kg": result["main_fuel_used"], "transitions": result["transitions"],
            "rcs_saturated_steps": saturated, "rcs_saturation_fraction": saturated / result["transitions"],
        })
    return episodes


def wilson95(safe, total):
    if total <= 0 or not 0 <= safe <= total:
        raise ValueError("a Wilson interval needs a positive total and 0 <= safe <= total")
    p, z = safe / total, 1.959963984540054
    denominator = 1 + z*z / total
    center = (p + z*z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z*z / (4 * total**2)) / denominator
    return [0. if safe == 0 else max(0., center - half), 1. if safe == total else min(1., center + half)]


def summarize(episodes):
    count = len(episodes)
    if not count:
        raise ValueError("at least one completed episode is required")
    statuses = Counter(ep["status"] for ep in episodes)
    safe = sum(ep["is_success"] for ep in episodes)
    target = sum(ep["safe_in_target"] for ep in episodes)
    speeds = [ep["touchdown_speed_m_s"] for ep in episodes if ep["touchdown_speed_m_s"] is not None]
    tilts = [ep["touchdown_tilt_deg"] for ep in episodes if ep["touchdown_tilt_deg"] is not None]
    steps = sum(ep["transitions"] for ep in episodes)
    saturated = sum(ep["rcs_saturated_steps"] for ep in episodes)
    return {
        "n_episodes": count, "safe_count": safe, "safe_landing_rate": safe / count,
        "safe_landing_rate_wilson95": wilson95(safe, count), "safe_in_target_count": target,
        "safe_in_target_rate": target / count, "status_counts": dict(statuses),
        "crash_count": statuses["crash"], "timeout_count": statuses["timeout"],
        "other_count": count - safe - statuses["crash"] - statuses["timeout"], "touchdown_count": len(speeds),
        "mean_touchdown_speed_m_s": float(np.mean(speeds)) if speeds else None,
        "p95_touchdown_speed_m_s": float(np.percentile(speeds, 95)) if speeds else None,
        "max_touchdown_speed_m_s": max(speeds, default=None),
        "mean_touchdown_tilt_deg": float(np.mean(tilts)) if tilts else None,
        "mean_return": float(np.mean([ep["return"] for ep in episodes])),
        "mean_main_fuel_used_kg": float(np.mean([ep["main_fuel_used_kg"] for ep in episodes])),
        "transitions": steps, "rcs_saturated_steps": saturated, "rcs_saturation_fraction": saturated / steps,
    }


# The full run has a hard gate between the nominal reproduction and the sweep.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "agent/runs/transfer-sensitivity"
POLICIES = ("online", "bc_only", "scripted_nominal", "scripted_privileged")
IDENTITIES = {
    "online": "4cb558d7e7fa66e0c9dca317b96c0d34dec58a60deba99802768178e1be57fee",
    "bc_only": "8fa4ccdf2ebf5bc684200e10ac03d6e7a9e99f97bd342f9b68df977e804778fb",
    "validation_content": "2bf7a0898b9ed4f38dd46f996765f675a97fe703f0c9dcc0fee83727d0e51321",
}


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def init_worker():
    torch.set_num_threads(1)


def run_pair(pool, pair_keys, catalog, starts, paths, out, metadata):
    """Save each finished pair independently, in frozen manifest order."""
    futures = {
        pool.submit(evaluate_pair, catalog[name], controller, starts, *paths): (name, controller)
        for name, controller in pair_keys
    }
    results = {}
    for future in as_completed(futures):
        name, controller = futures[future]
        episodes = future.result()
        row = summarize(episodes)
        results.setdefault(name, {})[controller] = row
        write_json(out / f"{name}--{controller}.json", {
            "condition": asdict(catalog[name]), "controller": controller,
            "inputs": metadata, "summary": row, "episodes": episodes,
        })
        print(f"{name} {controller}: safe {row['safe_count']}/{len(episodes)}", flush=True)
    return results


def write_table(path, results):
    lines = [
        "| Condition | Controller | N | Safe [Wilson 95%] | Safe in target | Crash / timeout / other | Contacts | Touch speed mean / p95 / max (m/s) | Mean return | Saturated RCS steps |",
        "|---|---|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for name, rows in results.items():
        for controller in POLICIES:
            if controller not in rows:
                continue
            row = rows[controller]
            lo, hi = row["safe_landing_rate_wilson95"]
            failures = " / ".join(str(row[key]) for key in ("crash_count", "timeout_count", "other_count"))
            speeds = [row[f"{stat}_touchdown_speed_m_s"] for stat in ("mean", "p95", "max")]
            speed_text = " / ".join("—" if value is None else f"{value:.3f}" for value in speeds)
            lines.append(f"| {name} | {controller} | {row['n_episodes']} | {row['safe_count']} "
                         f"[{lo:.1%}, {hi:.1%}] | {row['safe_in_target_count']} | {failures} | "
                         f"{row['touchdown_count']} | {speed_text} | {row['mean_return']:.4f} | "
                         f"{row['rcs_saturation_fraction']:.2%} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the frozen S3 validation transfer sweep on CPU.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT / "bundle")
    parser.add_argument("--bc", type=Path, default=ROOT / "agent/runs/s3-stable/seed-33/checkpoints/bc-only.pt")
    parser.add_argument("--selection", type=Path, default=ROOT / "agent/runs/s3-stable/report/selection.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest_path = args.bundle / "s3-validation.json"
    manifest = load_manifest(manifest_path, "validation")
    paths = (args.bundle / "policy.pt", args.bc)
    identity = {"online": file_hash(paths[0]), "bc_only": file_hash(paths[1]),
                "validation_content": manifest["sha256"]}
    selection = json.loads(args.selection.read_text())
    if (identity != IDENTITIES or len(manifest["starts"]) != 100 or
            selection["checkpoint_sha256"]["bc_only"] != identity["bc_only"] or
            selection["validation_sha256"] != identity["validation_content"]):
        raise ValueError("input files differ from the registered policies or validation starts")
    metadata = {
        "sha256": {**identity, "validation_file": file_hash(manifest_path),
                   "runner": file_hash(Path(__file__).resolve()),
                   "transfer_helper": file_hash(ROOT / "agent/src/mun_agent/transfer.py")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "device": "cpu", "split": "validation",
    }
    catalog = conditions()
    pairs = [(name, policy) for name, condition in catalog.items()
             for policy in (POLICIES[:3] if condition.category == "observation" else POLICIES)]
    pair_dir = args.out / "pairs"
    repeat_dir = args.out / "repeat"
    pair_dir.mkdir(parents=True, exist_ok=True)
    repeat_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        def collect(keys, destination):
            return run_pair(pool, keys, catalog, manifest["starts"], paths, destination, metadata)

        results.update(collect([key for key in pairs if key[0] == "nominal"], pair_dir))
        references = {"online": "93.3988", "bc_only": "92.0332", "scripted_nominal": "92.1085"}
        for policy, expected in references.items():
            row = results["nominal"][policy]
            if (row["safe_count"], row["safe_in_target_count"], f"{row['mean_return']:.4f}") != (100, 100, expected):
                raise RuntimeError(f"G1 failed for {policy}: {row}")
        print("G1 passed for all three references.", flush=True)
        results.update(collect([key for key in pairs if key[0] != "nominal"], pair_dir))
        repeated = [key for key in pairs if key[0] in {"nominal", "inertia_4", "fuel_kg"}]
        collect(repeated, repeat_dir)

    for name, policy in repeated:
        filename = f"{name}--{policy}.json"
        original = json.loads((pair_dir / filename).read_text())["episodes"]
        rerun = json.loads((repeat_dir / filename).read_text())["episodes"]
        if original != rerun:
            raise RuntimeError(f"G2 per-episode mismatch: {name}, {policy}")
    deltas = {}
    for policy in POLICIES[:3]:
        original = results["nominal"][policy]
        row = results["fuel_kg_rescaled"][policy]
        delta = abs(row["mean_return"] - original["mean_return"])
        if (row["status_counts"] != original["status_counts"] or
                row["safe_in_target_count"] != original["safe_in_target_count"] or delta >= 1e-3):
            raise RuntimeError(f"G3 rescaling mismatch: {policy}, {delta}")
        deltas[policy] = delta
    ordered = {name: {p: results[name][p] for p in POLICIES if p in results[name]} for name in catalog}
    write_json(args.out / "summary.json", {
        "inputs": metadata, "results": ordered,
        "unsupported": {"inertia_roll_4": "1200 + 1200 < 3200 kg m^2; invalid rigid body"},
        "gates": {"G1_reproduced": references, "G2_identical_pairs": repeated,
                  "G3_absolute_return_deltas": deltas},
    })
    write_table(args.out / "summary.md", ordered)
    print("G2/G3 passed. Wrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
