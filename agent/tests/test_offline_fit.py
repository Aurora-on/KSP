import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mun_lander import SimConfig

from mun_agent.cli import main
from mun_agent.fit import load_world_model
from mun_agent.dataset import load_dataset
from mun_agent.wm_report import open_loop_report


def test_offline_fit_checkpoint_and_heldout_report(tmp_path):
    config = str(Path(__file__).resolve().parents[1] / "configs/debug.json")
    common = ["--config", config, "--device", "cpu"]
    train_data, heldout = tmp_path / "train", tmp_path / "heldout"
    for seed, path in ((17, train_data), (18, heldout)):
        main(["collect", *common, "--episodes", "2", "--seed", str(seed), "--out", str(path)])
    run = tmp_path / "model"
    main(["train", *common, "--data", str(train_data), "--seed", "31", "--out", str(run),
          "--set", "fit.updates=300"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["probe_after"]["reconstruction"] < 0.5 * summary["probe_before"]["reconstruction"]
    provenance = json.loads((run / "config.json").read_text())
    assert provenance["train_episodes"] == 2
    assert provenance["collection_seed"] == 17
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 300
    checkpoint = run / "world_model.pt"
    model, saved = load_world_model(checkpoint, torch.device("cpu"), SimConfig())
    train_replay, meta = load_dataset(train_data)
    metrics = open_loop_report(model, train_replay, saved["interface"], train_mean=saved["train_mean"],
                               provenance=meta["episode_provenance"], horizons=(1,))
    recon = metrics["posterior"]
    assert recon["position_error_m"] < 0.8 * recon["baselines"]["train_mean"]["position_error_m"]
    with pytest.raises(ValueError, match="interface mismatch"):
        load_world_model(checkpoint, torch.device("cpu"), replace(SimConfig(), control_dt=0.2))
    report = tmp_path / "report.json"
    options = ["wm-report", *common, "--checkpoint", str(checkpoint), "--out", str(report)]
    with pytest.raises(SystemExit):
        main([*options, "--data", str(train_data)])
    main([*options, "--data", str(heldout)])
    results = json.loads(report.read_text())
    assert results["fit_seed"] == 31
    assert results["heldout_manifest_sha256"] != results["train_manifest_sha256"]
    for horizon in ("1", "5", "15", "50"):
        events = results["horizons"][horizon]["terminal"]
        assert events["tp"] + events["fn"] == 2
