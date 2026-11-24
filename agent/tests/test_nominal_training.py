import json
from pathlib import Path

import numpy as np
import torch
from mun_lander import SimConfig

from mun_agent import nominal, train
from mun_agent.collect import collect
from mun_agent.dataset import save_dataset
from mun_agent.deterministic import DeterministicModel
from mun_agent.interface import deterministic_spec
from mun_agent.policy import load_policy


def test_nominal_fixed_budget_selection_and_safety_drop_artifacts(tmp_path, monkeypatch):
    torch.manual_seed(7)
    spec = deterministic_spec(SimConfig())
    dynamics = DeterministicModel(spec, width=16)
    dynamics.delta_std.fill_(.001)
    source = tmp_path / "wm.pt"
    torch.save({"model_type": "deterministic", "schema_version": 1, "config": {"width": 16, "event_weight": 40.},
                "interface_spec": spec, "model": dynamics.state_dict()}, source)
    replay, metadata = collect(seed=92, episodes=2, noise_levels=(0.,), random_fraction=0., hard_fraction=0.)
    data = tmp_path / "prefill"
    save_dataset(data, replay, metadata)
    validation = tmp_path / "validation.json"
    nominal.prepare_manifest(validation, "validation", episodes=1)
    config = json.loads((Path(__file__).parents[1] / "configs/s3_nominal.json").read_text())
    config.update(iterations=4, validate_every=1, cpu_threads=1, bc_updates=2, bc_batch=8, actor_width=8,
                  critic_width=8, wm_updates=1, wm_batch=4, actor_updates=1, imagined_batch=4,
                  horizon=2, collect_episodes=1, fifo_episodes=1, critic_epochs=1)

    # Keep the real simulator rollout and its per-episode diagnostics, but inject
    # a controlled validation ranking, including two ties and a mid-run dip.
    evaluate = nominal.evaluate_nominal
    ranks = iter([(0., 0.), (1., 10.), (1., 20.), (1., 20.), (.5, 200.), (1., 19.)])
    def ranked(policy, manifest):
        result = evaluate(policy, manifest)
        result["safe_landing_rate"], result["mean_return"] = next(ranks)
        return result
    monkeypatch.setattr(nominal, "evaluate_nominal", ranked)
    out = tmp_path / "nominal"
    result = train.train_policy(config, data, source, validation, out)
    assert result["iteration"] == 4 and result["completed_fixed_iterations"]
    assert result["best_iteration"] == 2 and result["best_mean_return"] == 20.
    assert result["validation_passed_vs_bc"] and result["passed"] is None
    rows = [json.loads(line) for line in (out / "validation.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["iteration"]) for r in rows] == [
        ("untrained", 0), ("bc_only", 0), ("online", 1), ("online", 2), ("online", 3), ("online", 4)]
    for row in rows:
        assert {"mean_return", "safe_in_target_rate", "mean_flight_time_s", "mean_landing_error_m"} <= row.keys()
        assert all(f"{stat}_touchdown_{quantity}" in row for stat in ("mean", "max")
                   for quantity in ("speed_m_s", "angular_speed_rad_s", "tilt_deg"))
    best = torch.load(out / "checkpoints/best.pt", weights_only=True)
    for iteration in (1, 2, 3, 4):
        label = f"online-{iteration:06d}"
        evidence = json.loads((out / "validation" / (label + ".json")).read_text())
        assert len(evidence["episodes"]) == 1 and evidence["iteration"] == iteration
        snapshot = out / "checkpoints" / (label + ".pt")
        frozen = torch.load(snapshot, weights_only=True)
        assert "fifo" not in frozen and frozen["progress"]["iteration"] == iteration
        policy = load_policy(snapshot)
        assert not policy.actor.vertical_only and policy(replay.episodes[0].obs[0]).shape == (4,)
        assert train.load_training_world_model(snapshot).spec == spec
        if iteration == result["best_iteration"]:
            for name in ("actor", "world_model"):
                assert all(torch.equal(value, frozen[name][key]) for key, value in best[name].items())
    last = torch.load(out / "checkpoints/latest.pt", weights_only=True)
    assert np.all(np.any(last["fifo"][0]["action"].numpy()[:, 1:] != 0, axis=0))
    assert last["progress"]["metrics"][-1]["collection_env_steps_per_second"] > 0
