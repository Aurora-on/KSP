import json
from pathlib import Path

import pytest

from mun_agent.cli import main
from mun_agent.dataset import file_hash


def test_deterministic_fit_report_and_split_roles(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    rssm_config = str(configs / "debug.json")
    paths = {role: tmp_path / role for role in ("train", "dev", "heldout")}
    for seed, path in enumerate(paths.values(), 51):
        main(["collect", "--config", rssm_config, "--seed", str(seed),
              "--episodes", "1", "--out", str(path)])

    rssm_dir = tmp_path / "rssm"
    main(["train", "--config", rssm_config, "--data", str(paths["train"]), "--out", str(rssm_dir),
          "--device", "cpu", "--set", "fit.updates=1"])
    settings = json.loads((configs / "s1.json").read_text())
    settings.update(updates=1, batch_size=2, checkpoint_every=1, cpu_threads=1, width=16)
    config = tmp_path / "s1.json"
    config.write_text(json.dumps(settings))
    fit_dir = tmp_path / "deterministic"
    common = ["--config", str(config), "--device", "cpu"]
    main(["fit-deterministic", *common, "--data", str(paths["train"]), "--dev-data", str(paths["dev"]),
          "--out", str(fit_dir)])
    rssm_checkpoint = rssm_dir / "world_model.pt"
    heldout = json.loads((paths["heldout"] / "collection.json").read_text())
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"name": "smoke collection", "manifest_sha256": heldout["train_manifest_sha256"],
                                     "replay_sha256": [heldout["replay_sha256"]],
                                     "rssm_checkpoint_sha256": file_hash(rssm_checkpoint)}))
    report = ["wm-report", *common, "--checkpoint", str(fit_dir / "best.pt"),
              "--rssm-checkpoint", str(rssm_checkpoint), "--benchmark", str(benchmark)]
    with pytest.raises(SystemExit):
        main([*report, "--data", str(paths["train"]), "--out", str(tmp_path / "reject.json")])
    for role, path in paths.items():
        output = tmp_path / f"{role}.json"
        options = ["--training-diagnostic"] if role == "train" else []
        main([*report, *options, "--data", str(path), "--out", str(output)])
        result = json.loads(output.read_text())
        assert result["dataset_role"] == {"train": "training", "dev": "development", "heldout": "heldout"}[role]
        assert result["acceptance"]["pass"] is False
        assert result["acceptance"]["matches_benchmark"] is (role == "heldout")
        assert set(result["horizons"]["32"]["all"]) == {"deterministic", "rssm", "train_mean", "persistence"}
        assert result["reward_parity"]["terminal_total_abs_error"]["count"] == 1
