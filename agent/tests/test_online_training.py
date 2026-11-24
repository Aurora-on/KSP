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
from mun_agent.vertical import collect_prefill, prepare_validation


def test_real_loop_resumes_from_last_complete_iteration(tmp_path, monkeypatch):
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
    config.update(iterations=2, collect_episodes=1, fifo_episodes=1, actor_updates=1, wm_updates=1,
                  actor_width=8, critic_width=8, imagined_batch=4, wm_batch=4, horizon=2,
                  cpu_threads=1, validate_every=1, critic_epochs=1)
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
    assert [r["iteration"] for r in rows] == [0,1,2]
    assert all(r["n_episodes"] == 1 and r["starts_sha256"] == rows[0]["starts_sha256"] for r in rows)
    best = interrupted / "checkpoints/best.pt"
    policy = load_policy(best)
    assert policy(replay.episodes[0].obs[0])[1:].tolist() == [0.,0.,0.]
    fitted = training.load_training_world_model(interrupted / "checkpoints/latest.pt")
    assert all(torch.equal(v, fitted.state_dict()[k]) for k,v in saved["world_model"].items())
    with pytest.raises(ValueError, match="identical"):
        training.train_policy(config | {"horizon":3}, data, source, validation, interrupted, resume=True)
    expired = tmp_path / "budget"
    result = training.train_policy(config | {"max_seconds":1e-9}, data, source, validation, expired)
    assert result["time_budget_exhausted"] and result["actor_updates"] == result["model_updates"] == 0
    assert (expired / "checkpoints/latest.pt").exists() and not (expired / "checkpoints/best.pt").exists()
