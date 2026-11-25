import json

import numpy as np
import pytest
import torch
from mun_lander import SimConfig
from mun_lander.evaluation import generate_initial_states, validate_manifest

from mun_agent.deterministic import DeterministicModel
from mun_agent.imagination import Actor
from mun_agent.interface import deterministic_spec
from mun_agent.policy import Policy
from mun_agent.vertical import (collect_policy, collect_prefill, evaluate_vertical, load_validation,
                                paired_returns, prepare_validation, vertical_manifest)


def test_vertical_projection_and_frozen_validation_file(tmp_path):
    path = tmp_path / "vertical.json"
    projected = prepare_validation(path, episodes=2)
    nominal = generate_initial_states(SimConfig(), 2, 2026, "validation")
    assert projected["source_manifest_sha256"] == nominal["sha256"]
    assert projected["sha256"] != nominal["sha256"]
    train = vertical_manifest(split="train", seed=2026, episodes=2)
    test = generate_initial_states(SimConfig(), 2, 2026, "test")
    for other in (test, train):
        assert {s["seed"] for s in other["starts"]}.isdisjoint(s["seed"] for s in projected["starts"])
    validate_manifest(projected)
    for original, start in zip(nominal["starts"], projected["starts"], strict=True):
        assert start["seed"] == original["seed"] and start["id"] == original["id"]
        state = start["initial_state"]
        for key in ("position", "velocity"):
            assert state[key][:2] == [0., 0.]
            assert state[key][2] == original["initial_state"][key][2]
        assert state["angular_velocity"] == [0., 0., 0.]
        assert state["quaternion"][1:3] == [0., 0.]
        assert state["fuel"] == original["initial_state"]["fuel"]
    assert prepare_validation(path, episodes=2) == projected
    with pytest.raises(ValueError, match="different seed or size"):
        prepare_validation(path, episodes=1)
    projected["starts"][0]["initial_state"]["position"][0] = 1
    path.write_text(json.dumps(projected))
    with pytest.raises(ValueError, match="checksum"):
        load_validation(path)


def test_online_noise_applied_before_rcs_projection_and_resets_per_episode():
    class Constant:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

        def __call__(self, obs):
            return np.array([0., 1., 1., 1.], dtype=np.float32)

    policy = Constant()
    replay, meta = collect_policy(policy, seed=31, episodes=2, noise_std=.1)
    assert policy.resets == 2
    for episode, row in zip(replay.episodes, meta["episode_provenance"], strict=True):
        assert not np.any(episode.action[:, 1:])
        assert row["action_rng_seed"] == [row["reset_seed"], 6]
        noise = np.random.default_rng(row["action_rng_seed"]).normal(0., .1, size=4)[0]
        assert episode.action[0, 0] == pytest.approx((np.clip(-1. + noise, -1., 1.) + 1.) / 2.)
        assert not np.any(episode.obs[:, [0,1,3,4,7,8,10,11,12]])
    validation = vertical_manifest(split="validation", seed=2026, episodes=2)
    result = evaluate_vertical(policy, validation)
    assert policy.resets == 4 and result["starts_sha256"] == validation["sha256"]
    assert result["status_counts"] == {"crash": 2}


def test_paired_closed_loop_starts_and_terminal_rewards():
    torch.manual_seed(8)
    spec = deterministic_spec(SimConfig())
    actor = Actor(width=8)
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()
        actor.network[-1].bias[0] = -10.
    model = DeterministicModel(spec, width=8)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        # Constant state and no predicted termination create a clean optimism check.
        model.event[-1].bias[0] = 25.
    manifest = vertical_manifest(split="validation", seed=2026, episodes=2)
    result = paired_returns(model, actor, Policy(actor, spec), manifest)
    assert result["episodes"] == 2 and result["horizon"] == 32
    assert result["failed_status_counts"] == {"crash": 2}
    assert result["by_phase"]["terminal"]["mean_gap"] > 90.
    assert result["by_phase"]["reset"]["count"] == 2
    assert len(result["windows"]) == 8
    assert all(r["real_steps"] == 32 for r in result["windows"])
    assert all(r["real_terminal"] for r in result["windows"] if r["phase"] == "terminal")
    other = Actor(width=8)
    with pytest.raises(ValueError, match="disagree"):
        paired_returns(model, other, Policy(actor, spec), manifest)


def test_permanent_prefill_actions_obey_constraint():
    replay, metadata = collect_prefill(seed=230, episodes=10)
    assert {r["controller"] for r in metadata["episode_provenance"]} == {"scripted", "random", "hard_landing"}
    assert all(not np.any(ep.action[:, 1:]) for ep in replay.episodes)
