import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mun_agent import nominal_report
from mun_agent.nominal import prepare_manifest


def test_report_selects_by_validation_and_reuses_finished_test_episodes(tmp_path, monkeypatch):
    """Unequal test rankings cannot influence the chosen seed or checkpoint."""
    validation, test = (tmp_path / f"{name}.json" for name in ("validation", "test"))
    manifest = prepare_manifest(validation, "validation", episodes=2)
    prepare_manifest(test, "test", episodes=2)
    runs = []
    for seed, best_return, last_rate in [(31, 2., .5), (32, 2.5, 1.), (33, 2.5, .5)]:
        run = tmp_path / str(seed)
        runs.append(run)
        folder = run / "checkpoints"
        folder.mkdir(parents=True)
        config = dict(config={"task": "nominal", "iterations": 2}, seed=seed, validation_sha256=manifest["sha256"],
                      prefill_replay_sha256="replay", pretrained_sha256="model")
        (run / "config.json").write_text(json.dumps(config))
        rows = []
        for phase, iteration, rate, ret in [("bc_only", 0, 1., 1.), ("online", 1, 1., best_return),
                                            ("online", 2, last_rate, best_return)]:
            rows.append(dict(phase=phase, iteration=iteration, safe_landing_rate=rate, mean_return=ret,
                             starts_sha256=manifest["sha256"], status_counts={"safe_landing": 2}))
        (run / "validation.jsonl").write_text("\n".join(map(json.dumps, rows)))
        best = max(rows[1:], key=lambda r: nominal_report.validation_rank(r, nominal_report.LEGACY_SELECTION_ORDER))["iteration"]
        (run / "summary.json").write_text(json.dumps(dict(elapsed_s=5, iteration=2, best_iteration=best)))
        for name, iteration in (("best", best), ("bc-only", 0)):
            torch.save({"provenance": config, "progress": {"iteration": iteration}, "actor_spec": {"vertical_only": False}},
                       folder / f"{name}.pt")

    calls = []
    def evaluate(policy, starts):
        calls.append((policy, starts["split"]))
        return dict(starts_sha256=starts["sha256"], status_counts={"safe_landing": 2}, safe_landing_rate=1.,
                    safe_in_target_rate=1., mean_return=float(policy == "best") + 1.,
                    mean_landing_error_safe=1., mean_total_fuel_used_kg=20., mean_flight_time_s=20.,
                    mean_touchdown_speed_m_s=.6)
    monkeypatch.setattr(nominal_report, "load_policy", lambda path, device: Path(path).stem)
    monkeypatch.setattr(nominal_report, "evaluate_nominal", evaluate)
    args = SimpleNamespace(run=runs, checkpoint=None, out=tmp_path / "report", validation=validation,
                           test=test, resume=False, device="cpu")
    result = nominal_report.make_report(args)
    assert (result["selected"]["seed"], result["selected"]["iteration"]) == (32, 2)
    assert [split for _, split in calls] == ["validation"] * 3 + ["test"] * 3
    assert not result["gates"]["qualified"]  # two-start smoke manifests
    assert result["gates"]["by_split"]["test"]["mean_return_above_clone"]
    markdown = (args.out / "report.md").read_text()
    assert "Last in target | Min online in target" in markdown
    assert all(run["min_online_safe_in_target_rate"] is None for run in
               json.loads((args.out / "report.json").read_text())["runs"])
    args.resume = True
    nominal_report.make_report(args)
    assert len(calls) == 6
    args.checkpoint = runs[0] / "checkpoints/best.pt"
    with pytest.raises(ValueError, match="validation-selected"):
        nominal_report.make_report(args)


def test_contribution_gate_applies_independently_on_both_splits():
    high = {"safe_landing_rate": .96, "safe_in_target_rate": .91, "mean_return": 90.}
    low = {**high, "safe_landing_rate": .94, "mean_return": 89.}
    comparisons = {split: {"online": high, "bc_only": low} for split in ("validation", "test")}
    assert nominal_report.gates(comparisons, 3, True)["qualified"]
    comparisons["test"]["bc_only"] = {**low, "safe_landing_rate": .97}
    failed = nominal_report.gates(comparisons, 3, True)
    assert failed["test_target"] and not failed["qualified"]


def test_versioned_cross_seed_and_intra_seed_rank():
    legacy = list(nominal_report.LEGACY_SELECTION_ORDER)
    current = list(nominal_report.SELECTION_ORDER)
    def candidate(seed, target, reward, iteration, order=current, safe=1.):
        return dict(seed=seed, selection_order=order,
                    best=dict(safe_landing_rate=safe, safe_in_target_rate=target,
                              mean_return=reward, iteration=iteration))
    seeds = [candidate(1, .95, 12., 2), candidate(2, .9, 30., 3),
             candidate(3, .95, 12., 4), candidate(4, .95, 10., 8),
             candidate(5, .95, 90., 9, safe=.96)]
    winner, priority = nominal_report.select_candidate(seeds)
    assert winner["seed"] == 3 and priority == current
    for row in seeds:
        row["selection_order"] = legacy
    assert nominal_report.select_candidate(seeds) == (seeds[1], legacy)
    seeds[0]["selection_order"] = current
    assert nominal_report.select_candidate(seeds) == (seeds[2], current)
    with pytest.raises(ValueError, match="unsupported nominal"):
        nominal_report.validation_rank(seeds[0]["best"], ["mean_return"])
