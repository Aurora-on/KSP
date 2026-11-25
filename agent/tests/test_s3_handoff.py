import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import torch
from mun_agent.dataset import file_hash
from mun_agent.imagination import Actor
from mun_agent.interface import deterministic_spec
from mun_agent.nominal import prepare_manifest
from mun_agent.policy import policy_checkpoint
from mun_lander import SimConfig

script = Path(__file__).resolve().parents[1] / "scripts/s3_handoff.py"
spec = importlib.util.spec_from_file_location("s3_handoff", script)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_export_uses_report_inputs_and_strips_training_state(tmp_path, monkeypatch):
    source = tmp_path / "best.pt"
    checkpoint = policy_checkpoint(Actor(width=8, vertical_only=False), deterministic_spec(SimConfig()))
    torch.save(dict(checkpoint, fifo=["private-training-record"], adam={"step": 420}), source)
    sha = file_hash(source)
    monkeypatch.setattr(handoff, "SOURCE_SHA", sha)
    selected = {"seed": 33, "iteration": 22, "checkpoint_sha256": {"online": sha}}
    evaluations = {}
    for split in handoff.SPLITS:
        path = tmp_path / f"s3-{split}.json"
        prepare_manifest(path, split)
        digest = json.loads(path.read_text())["sha256"]
        evaluations[split] = {"online": dict(split=split, starts_sha256=digest, n_episodes=100,
                                              checkpoint_sha256=sha, safe_landing_rate=1.,
                                              safe_in_target_rate=1., mean_return=93., mean_landing_error_m=1.)}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(selected=selected, evaluations=evaluations)))
    archive = tmp_path / "v1.zip"
    assert handoff.export(source, report, tmp_path, archive) == file_hash(archive)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"handoff.json", "s3-validation.json", "s3-test.json", "policy.pt"}
        bundle.extractall(unpacked)
    handoff.verify(unpacked)
    actual = torch.load(unpacked / "policy.pt", weights_only=True)
    assert actual.keys() == checkpoint.keys()
    for key, tensor in checkpoint["actor"].items():
        assert torch.equal(tensor, actual["actor"][key])
    for split in handoff.SPLITS:
        name = f"s3-{split}.json"
        assert (tmp_path / name).read_bytes() == (unpacked / name).read_bytes()
    with (unpacked / "s3-test.json").open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        handoff.verify(unpacked)
    evaluations["test"]["online"]["starts_sha256"] = "wrong reported start list"
    report.write_text(json.dumps(dict(selected=selected, evaluations=evaluations)))
    new_archive = tmp_path / "wrong.zip"
    with pytest.raises(ValueError, match="differ from the formal report"):
        handoff.export(source, report, tmp_path, new_archive)
    assert not new_archive.exists()
