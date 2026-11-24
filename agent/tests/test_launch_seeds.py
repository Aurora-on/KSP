import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "launch_seeds.sh"


@pytest.fixture
def launch(tmp_path):
    trainer = tmp_path / "mun-agent"
    trainer.write_text(f"#!{sys.executable}\n" + """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
seed = args[args.index('--seed') + 1]
out = Path(args[args.index('--out') + 1])
if '--resume' not in args:
    out.mkdir()
assert out.is_dir() and args[0] == 'train-policy'
t = time.monotonic()
if os.environ.get('CHECK_CONCURRENCY') == '1' and int(seed) <= 34:
    (out.parent / ('ready-' + seed)).touch()
    while len(list(out.parent.glob('ready-*'))) < 4:
        if time.monotonic() - t > 10:
            sys.exit(24)
        time.sleep(.01)
time.sleep(.05)
print(json.dumps(dict(args=args, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), times=[t,time.monotonic()])))
sys.exit(23 if os.environ.get('FAILING_SEED') == seed else 0)
""")
    trainer.chmod(0o755)
    for name in ("config", "model", "validation", "collection.json", "replay.npz"):
        (tmp_path / name).touch()
    def run(*args, **extra_env):
        cmd = [str(LAUNCHER), "--config", str(tmp_path / "config"), "--data", str(tmp_path),
               "--world-model", str(tmp_path / "model"), "--validation", str(tmp_path / "validation"),
               "--out-root", str(tmp_path / "runs"), *args]
        env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], **extra_env}
        return subprocess.run(cmd, env=env, text=True, capture_output=True)
    return run


def test_launcher_respects_gpu_mask_capacity_and_independent_seed_dirs(launch, tmp_path):
    result = launch("--seeds", "31,32,33,34,35", "--jobs-per-device", "2", CUDA_VISIBLE_DEVICES="3,7", CHECK_CONCURRENCY="1")
    assert result.returncode == 0, result.stderr
    logs = [json.loads((tmp_path / "runs" / f"seed-{s}.log").read_text()) for s in range(31, 36)]
    assert [x["gpu"] for x in logs] == ["3", "7", "3", "7", "3"]
    assert max(x["times"][0] for x in logs[:4]) < min(x["times"][1] for x in logs[:4])
    assert logs[4]["times"][0] >= max(x["times"][1] for x in logs[:4])


def test_launcher_resumes_existing_and_starts_missing_and_propagates_failure(launch, tmp_path):
    first = launch("--seeds", "31,32,33", "--jobs-per-device", "2", "--device", "cpu", FAILING_SEED="32")
    assert first.returncode == 23 and "Seed 32 failed" in first.stderr
    assert not (tmp_path / "runs" / "seed-33").exists()
    refused = launch("--device", "cpu")
    assert refused.returncode != 0
    second = launch("--device", "cpu", "--resume")
    assert second.returncode == 0, second.stderr
    for seed in range(31, 34):
        log = (tmp_path / "runs" / f"seed-{seed}.log").read_text().splitlines()
        action = json.loads(log[-1])["args"]
        assert action[action.index("--device") + 1] == "cpu"
        assert ("--resume" in action) == (seed < 33)
