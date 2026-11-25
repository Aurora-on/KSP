import hashlib
import json
from pathlib import Path

import pytest
import torch
from mun_lander import SimConfig

import mun_agent.train as training
from mun_agent.dataset import save_dataset
from mun_agent.deterministic import DeterministicModel
from mun_agent.interface import deterministic_spec
from mun_agent.policy import load_policy
from mun_agent.vertical import collect_prefill, load_validation, prepare_validation, vertical_manifest


@pytest.mark.parametrize(("estimator", "warm_start"), [("pathwise", False), ("sampled", False), ("pathwise", True)])
def test_real_loop_resumes_from_last_complete_iteration(tmp_path, monkeypatch, estimator, warm_start):
    spec = deterministic_spec(SimConfig())
    torch.manual_seed(123)
    wm = DeterministicModel(spec, width=16)
    wm.delta_std.fill_(.001)
    source = tmp_path / "pretrained.pt"
    torch.save(dict(model_type="deterministic", schema_version=1, config={"width":16, "event_weight":40.},
                    interface_spec=spec, model=wm.state_dict()), source)
    replay, metadata = collect_prefill(episodes=2)
    data = tmp_path / "data"
    save_dataset(data, replay, metadata)
    validation = tmp_path / "validation.json"
    prepare_validation(validation, episodes=1)
    config = json.loads((Path(__file__).parents[1] / "configs/diag_vertical.json").read_text())
    config.update(estimator=estimator, action_std=.3, iterations=2, collect_episodes=1, fifo_episodes=1, actor_updates=1, wm_updates=1,
                  actor_width=8, critic_width=8, imagined_batch=4, wm_batch=4, horizon=2,
                  cpu_threads=1, validate_every=1, critic_epochs=1)
    config["bc_updates"] = 2 if warm_start else 0
    if warm_start:
        config.update(bc_lr=.0005, bc_batch=8)
        # Force the raw and BC-only baselines above threshold in each run.
        # Both online evaluations must still run and become the eligible best.
        evaluation = training.evaluate_vertical
        rates = iter([1., 1., 0., 0.] * 2)
        def test_rate(*args):
            result = evaluation(*args)
            result["safe_landing_rate"] = next(rates)
            return result
        monkeypatch.setattr(training, "evaluate_vertical", test_rate)
    args = config, data, source, validation
    uninterrupted, interrupted = tmp_path / "control", tmp_path / "interrupted"
    result = training.train_policy(*args, uninterrupted)
    assert not result["official_validation"] and not result["passed"]
    collect = training.collect_policy
    def fail_on_second(policy, *, seed, **kwargs):
        if seed % 1_000_000 == 2:
            raise InterruptedError("simulated job interruption")
        return collect(policy, seed=seed, **kwargs)
    monkeypatch.setattr(training, "collect_policy", fail_on_second)
    with pytest.raises(InterruptedError):
        training.train_policy(*args, interrupted)
    # A partial or stale text log is reconstructed from the authoritative checkpoint.
    (interrupted / "validation.jsonl").write_text('{"broken"')
    monkeypatch.setattr(training, "collect_policy", collect)
    training.train_policy(*args, interrupted, resume=True)
    ref, saved = [torch.load(d / "checkpoints/latest.pt", weights_only=True) for d in (uninterrupted,interrupted)]
    assert saved["progress"]["iteration"] == 2
    assert saved["progress"]["env_steps"] == ref["progress"]["env_steps"] > metadata["env_steps"]
    assert saved["progress"]["actor_updates"] == saved["progress"]["model_updates"] == 2
    for name in ("actor", "critic", "world_model"):
        assert all(torch.equal(v, saved[name][k]) for k,v in ref[name].items())
    assert len(saved["fifo"]) == 1
    assert torch.equal(saved["cpu_rng"], ref["cpu_rng"]) and saved["numpy_rng"] == ref["numpy_rng"]
    assert torch.equal(saved["fifo"][0]["action"], ref["fifo"][0]["action"])
    rows = [json.loads(line) for line in (interrupted / "validation.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == ([0, 0, 1, 2] if warm_start else [0, 1, 2])
    if warm_start:
        assert [r["phase"] for r in rows] == ["untrained", "bc_only", "online", "online"]
        clone = torch.load(interrupted / "checkpoints/bc-only.pt", weights_only=True)
        assert clone["progress"]["bc_only_rate"] == 1 and clone["progress"]["actor_updates"] == 0
        assert saved["progress"]["best_rate"] == 0
        assert (interrupted / "bc-only-validation.json").is_file()
    assert all(r["n_episodes"] == 1 and r["starts_sha256"] == rows[0]["starts_sha256"] for r in rows)
    best = interrupted / "checkpoints/best.pt"
    assert torch.load(best, weights_only=True)["progress"]["actor_updates"] > 0
    policy = load_policy(best)
    assert policy(replay.episodes[0].obs[0])[1:].tolist() == [0.,0.,0.]
    fitted = training.load_training_world_model(interrupted / "checkpoints/latest.pt")
    assert all(torch.equal(v, fitted.state_dict()[k]) for k,v in saved["world_model"].items())
    with pytest.raises(ValueError, match="identical"):
        training.train_policy(config | {"horizon":3}, data, source, validation, interrupted, resume=True)
    # A missing committed checkpoint must never silently overwrite a published run.
    (interrupted / "checkpoints/latest.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint is missing"):
        training.train_policy(*args, interrupted, resume=True)
    expired = tmp_path / "budget"
    result = training.train_policy(config | {"max_seconds":1e-9}, data, source, validation, expired)
    assert result["time_budget_exhausted"] and result["actor_updates"] == result["model_updates"] == 0
    assert (expired / "checkpoints/latest.pt").exists() and not (expired / "checkpoints/best.pt").exists()

def test_official_gate_checks_all_fixed_starts(tmp_path):
    canonical = vertical_manifest(split="validation", seed=2026, episodes=100)
    assert training._official_validation(canonical)
    canonical["starts"][0]["initial_state"]["position"][2] += 1.
    canonical.pop("sha256")
    canonical["sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "alternate-validation.json"
    path.write_text(json.dumps(canonical))
    assert not training._official_validation(load_validation(path))
