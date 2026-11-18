"""Protect common evaluation starts, conditional metrics and replay semantics."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from mun_lander import SimConfig
from mun_lander import evaluation


def test_starts_are_reproducible_and_split_seed_namespaces_are_disjoint():
    config = SimConfig()
    seen_seeds, seen_states = set(), set()
    for root_seed in (2026, 2027):
        for split in ("train", "adaptation", "validation", "test", "generalization"):
            manifest = evaluation.generate_initial_states(config, count=3, seed=root_seed, split=split)
            assert manifest == evaluation.generate_initial_states(config, count=3, seed=root_seed, split=split)
            evaluation.validate_manifest(manifest)
            seeds = {entry["seed"] for entry in manifest["starts"]}
            states = {json.dumps(entry["initial_state"], sort_keys=True) for entry in manifest["starts"]}
            assert len(seeds) == len(states) == 3
            assert seen_seeds.isdisjoint(seeds)
            assert seen_states.isdisjoint(states)
            seen_seeds.update(seeds)
            seen_states.update(states)


def test_manifest_rejects_changed_initial_state_and_duplicate_ids():
    manifest = evaluation.generate_initial_states(SimConfig(), count=2)
    altered = copy.deepcopy(manifest)
    altered["starts"][0]["initial_state"]["position"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        evaluation.validate_manifest(altered)
    duplicate = copy.deepcopy(manifest)
    duplicate["starts"][1]["id"] = duplicate["starts"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.validate_manifest(duplicate)


def test_no_safe_landings_produce_undefined_conditional_means():
    config = replace(SimConfig(), max_time=0.1)
    manifest = evaluation.generate_initial_states(config, count=2, split="test")
    summary = evaluation.evaluate(config, manifest, controller="zero")
    assert summary["n_episodes"] == 2
    assert summary["safe_landing_rate"] == summary["safe_in_target_rate"] == 0
    assert summary["status_counts"] == {"timeout": 2}
    assert summary["crash_rate"] == 0  # A timeout is a failure, not a collision.
    for field in ("horizontal_error", "main_fuel_used", "rcs_fuel_used", "total_fuel_used"):
        assert summary[f"mean_{field}_safe"] is None
    lo, hi = summary["safe_landing_rate_wilson95"]
    assert 0 <= lo < hi <= 1
    json.dumps(summary, allow_nan=False)


def test_sweep_hides_perturbed_parameters_from_scripted_controller(monkeypatch):
    config = replace(SimConfig(), max_time=0.1)
    manifest = evaluation.generate_initial_states(config, count=1, split="generalization")
    seen_controller_configs = []

    class ObservedController:
        def __init__(self, controller_config):
            seen_controller_configs.append(controller_config)

        def __call__(self, observation):
            assert observation.shape == (14,)
            return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(evaluation, "ScriptedController", ObservedController)
    sweep = evaluation.parameter_sweep(config, manifest, scales=(0.8, 1.2))
    assert len(seen_controller_configs) == 9  # Shared nominal plus 2 changes x 4 parameters.
    assert all(value == config for value in seen_controller_configs)
    assert sweep["scales"] == [0.8, 1.0, 1.2]
    assert len(sweep["runs"]) == 12
    for run in sweep["runs"]:
        result = run["evaluation"]
        assert result["controller_config"] == config.to_dict()
        assert result["config"] == config.perturbed(run["parameter"], run["scale"]).to_dict()
        assert result["starts_sha256"] == manifest["sha256"]
        assert result["episodes"][0]["start_id"] == manifest["starts"][0]["id"]


@pytest.mark.parametrize("ending", ["timeout", "crash"])
def test_recording_arrays_align_and_keep_final_observation(tmp_path, ending):
    config = replace(SimConfig(), max_time=0.25)
    initial = None
    if ending == "crash":
        initial = {"position": [0, 0, 0.001], "velocity": [0, 0, -8],
                   "quaternion": [1, 0, 0, 0], "angular_velocity": [0, 0, 0], "fuel": 600}
    policy = lambda obs: np.array([0.3, 0, 0, 0], dtype=np.float32)
    result, record = evaluation.rollout(config, initial_state=initial, controller=policy, record=True)
    assert result["status"] == ending
    path = tmp_path / "trajectory.npz"
    np.savez_compressed(path, **record)
    with np.load(path) as saved:
        count = result["transitions"]
        assert saved["observations"].shape == (count + 1, 14)
        assert saved["actions"].shape == (count, 4)
        for field in ("rewards", "terminated", "truncated", "dt"):
            assert saved[field].shape == (count,)
        for field in ("times", "main_fuel_used", "rcs_fuel_used"):
            assert saved[field].shape == (count + 1,)
        np.testing.assert_allclose(np.diff(saved["times"]), saved["dt"], atol=1e-12)
        assert not (saved["terminated"][:-1] | saved["truncated"][:-1]).any()
        assert bool(saved["terminated"][-1]) == (ending == "crash")
        assert bool(saved["truncated"][-1]) == (ending == "timeout")
        assert saved["rewards"].sum() == pytest.approx(result["return"])
        assert saved["main_fuel_used"][-1] == pytest.approx(result["main_fuel_used"])
        assert saved["times"][-1] == pytest.approx(result["time"])
        if ending == "crash":
            assert saved["observations"][-1, 2] == 0
            assert saved["observations"][-1, 5] < -7.9
        else:
            assert saved["dt"][-1] == pytest.approx(0.05)
            assert saved["observations"][-1, 2] < saved["observations"][0, 2]
