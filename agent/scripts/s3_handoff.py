"""Export and verify the accepted nominal simulator policy and its fixed start lists."""
import argparse
import json
import os
import platform
import zipfile
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mun_agent.dataset import file_hash
from mun_agent.nominal import load_manifest
from mun_agent.policy import load_policy, policy_checkpoint
from mun_lander import LanderEnv

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "a4451b97f733712c4d7832572d787ff24b816d214bb9f55fc03cdfc0837eae65"
REFERENCE_REVISION = "0e4ff9daef78bf3ceba0e467288b62a4ffaccae0"
SPLITS = ("validation", "test")
FILES = ("policy.pt", "s3-validation.json", "s3-test.json")


def reference_actions(policy, start):
    """Record eight successive real observation/action pairs from the source policy."""
    env = LanderEnv()
    try:
        obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
        policy.reset()
        frames = []
        for _ in range(8):
            action = policy(obs)
            frames.append({"observation": obs.tolist(), "action": action.tolist()})
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return frames
    finally:
        env.close()


def verify(directory):
    info = json.loads((directory / "handoff.json").read_text())
    if (info["handoff_version"], info["seed"], info["iteration"], info["source_checkpoint_sha256"]) != (1, 33, 22, SOURCE_SHA):
        raise ValueError("unexpected S3 baseline identity")
    for name in FILES:
        if file_hash(directory / name) != info["file_sha256"][name]:
            raise ValueError(f"{name}: file checksum mismatch")
    probe = info["validation_probe"]
    for split in SPLITS:
        manifest = load_manifest(directory / f"s3-{split}.json", split)
        stats = info["evaluations"][split]
        if (manifest["sha256"] != stats["starts_sha256"] or len(manifest["starts"]) != 100
                or stats["n_episodes"] != 100):
            raise ValueError(f"{split}: manifest differs from the recorded evaluation")
        if split == "validation" and (probe["start_id"] != manifest["starts"][0]["id"] or len(probe["frames"]) != 8):
            raise ValueError("expected eight reference frames from the first validation start")
    policy = load_policy(directory / "policy.pt", "cpu")
    for frame in probe["frames"]:
        actual = policy(np.asarray(frame["observation"], dtype=np.float32))
        np.testing.assert_allclose(actual, frame["action"], atol=2e-6, rtol=0,
                                   err_msg="CPU output differs from the full source checkpoint")
    return info


def export(source, report_path, starts_dir, destination):
    report = json.loads(report_path.read_text())
    selected = report["selected"]
    if (selected["seed"], selected["iteration"], selected["checkpoint_sha256"]["online"]) != (33, 22, SOURCE_SHA):
        raise ValueError("report does not describe the accepted seed 33 iteration 22 baseline")
    if file_hash(source) != SOURCE_SHA:
        raise ValueError("source checkpoint differs from the report")
    policy = load_policy(source, "cpu")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp)
        torch.save(policy_checkpoint(policy.actor, policy.spec), stage / "policy.pt")
        metrics = {}
        for split in SPLITS:
            name = f"s3-{split}.json"
            (stage / name).write_bytes((starts_dir / name).read_bytes())
            manifest = load_manifest(stage / name, split)
            row = report["evaluations"][split]["online"]
            if (row["checkpoint_sha256"] != SOURCE_SHA or row["starts_sha256"] != manifest["sha256"]
                    or row["split"] != split or row["n_episodes"] != len(manifest["starts"]) or row["n_episodes"] != 100):
                raise ValueError(f"{split}: inputs differ from the formal report")
            metrics[split] = {k: row[k] for k in ("starts_sha256", "n_episodes", "safe_landing_rate",
                                                  "safe_in_target_rate", "mean_return", "mean_landing_error_m")}
            if split == "validation":
                start = manifest["starts"][0]
                probe = {"start_id": start["id"], "frames": reference_actions(policy, start)}
        info = {"handoff_version": 1, "seed": 33, "iteration": 22, "domain": "nominal MunLander-v0 simulator",
                "policy_schema_version": 1, "reference_code_revision": REFERENCE_REVISION,
                "source_checkpoint_sha256": SOURCE_SHA,
                "source_report_sha256": file_hash(report_path),
                "runtime": {"python": platform.python_version(), **{pkg: version(pkg) for pkg in
                            ("mun-agent", "mun-lander-sim", "torch", "numpy", "gymnasium")}},
                "file_sha256": {name: file_hash(stage / name) for name in FILES},
                "evaluations": metrics, "validation_probe": probe}
        (stage / "handoff.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        verify(stage)

        # Fixed ZIP headers keep filesystem timestamps and permissions out of the artifact.
        archive = stage / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name in (*FILES, "handoff.json"):
                member = zipfile.ZipInfo(name)
                member.external_attr = 0o100644 << 16
                output.writestr(member, (stage / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        # Publish the completed archive atomically and refuse to overwrite an existing release.
        os.link(archive, destination)
    return file_hash(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("export")
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--checkpoint", type=Path, default=ROOT / "agent/runs/s3-stable/seed-33/checkpoints/best.pt")
    build.add_argument("--report", type=Path, default=ROOT / "agent/runs/s3-stable/report/report.json")
    build.add_argument("--starts-dir", type=Path, default=ROOT / "agent/data")
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        print(f"{export(args.checkpoint, args.report, args.starts_dir, args.out)}  {args.out}")
    else:
        info = verify(args.directory)
        print(f"Verified source seed {info['seed']}, round {info['iteration']}, "
              f"both official manifests and {len(info['validation_probe']['frames'])} reference actions.")


if __name__ == "__main__":
    main()
