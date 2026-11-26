"""Regression checks for simulator-to-simulator adaptation without KSP dependencies."""
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch
from mun_lander import SimConfig
from mun_lander.evaluation import generate_initial_states, validate_manifest

from mun_agent import adapt as online
from mun_agent.adapt import check_seed_partition, initialize_models
from mun_agent.deterministic import DeterministicModel
from mun_agent.imagination import (Actor, Critic, TouchdownMargin, first_touchdown_cost,
                                   imagined_touchdown_margin, update_actor_critic)
from mun_agent.interface import deterministic_spec
from mun_agent.nominal import cheap_validation_manifest, collect_policy, evaluate_nominal
from mun_agent.policy import policy_checkpoint
from mun_agent.task_reward import task_reward
from mun_agent.transfer import target_config


AGENT = Path(__file__).resolve().parents[1]
BEST = AGENT / "runs/s3-stable/seed-33/checkpoints/best.pt"


def example_checkpoint():
    torch.manual_seed(42)
    spec = deterministic_spec(SimConfig())
    actor = Actor(8, vertical_only=False, yaw_invariant=True)
    critic_cfg = {"critic_width": 8, "value_scale": 100., "critic_upper_bound": 100.,
                  "yaw_invariant": True}
    critic = Critic(8, 100., upper_bound=100., yaw_invariant=True)
    arch = {"width": 8, "kinematic": True, "contact_margin": True}
    wm = DeterministicModel(spec, **arch)
    return dict(policy_checkpoint(actor, spec), training_schema=1, wm_arch=arch,
                world_model=wm.state_dict(), critic=critic.state_dict(), provenance={"config": critic_cfg})


def assert_loaded(saved):
    policy, critic, model = initialize_models(saved)
    for module, key in [(policy.actor, "actor"), (critic, "critic"), (model, "world_model")]:
        assert module.state_dict().keys() == saved[key].keys()
        for name, value in module.state_dict().items():
            assert torch.equal(value, saved[key][name])
    assert policy.spec == model.spec == saved["interface_spec"]


def test_init_roundtrip():
    assert_loaded(example_checkpoint())


@pytest.mark.skipif(not BEST.exists(), reason="local archived S3 training checkpoint")
def test_actual_s3_checkpoint_initialization():
    assert_loaded(torch.load(BEST, map_location="cpu", weights_only=True))


def test_all_five_seed_partitions():
    groups = {name: [i] for i, name in enumerate(("target_train", "target_heldout", "source_prefill", "validation", "test"))}
    check_seed_partition(**groups)
    for left, right in combinations(groups, 2):
        with pytest.raises(ValueError, match="reset seed"):
            check_seed_partition(**{**groups, right: groups[left]})
    with pytest.raises(ValueError, match="duplicate"):
        check_seed_partition(**{**groups, "target_train": [1, 1]})


def test_cheap_validation_bank():
    full = generate_initial_states(SimConfig(), 100, 2026, "validation")
    cheap = cheap_validation_manifest(full)
    validate_manifest(cheap)
    assert cheap["starts"] == full["starts"][:20]
    assert [s["id"] for s in cheap["starts"]] == [f"validation-2026-{i:06d}" for i in range(20)]
    with pytest.raises(ValueError):
        cheap_validation_manifest(generate_initial_states(SimConfig(), 100, 2026, "test"))


def test_target_dynamics_and_source_policy_interface():
    cfg = SimConfig()
    policy, _, _ = initialize_models(example_checkpoint())
    spec = deepcopy(policy.spec)
    manifest = generate_initial_states(cfg, 2, 88, "validation")
    source = evaluate_nominal(policy, manifest)
    target = evaluate_nominal(policy, manifest, config=target_config("P1"))
    assert source["mean_return"] != target["mean_return"] and target["config"] == target_config("P1").to_dict()
    nominal, _ = collect_policy(policy, seed=88, episodes=2, noise_std=.15)
    changed, _ = collect_policy(policy, seed=88, episodes=2, noise_std=.15, config=target_config("P1"))
    for old, new in zip(nominal.episodes, changed.episodes, strict=True):
        np.testing.assert_array_equal(old.obs[0], new.obs[0])
        np.testing.assert_array_equal(old.action[0], new.action[0])
        assert not np.array_equal(old.obs[1], new.obs[1])
    assert policy.spec == spec


def test_default_imagined_return_matches_s3_equation_bitwise():
    policy, critic, wm = initialize_models(example_checkpoint())
    starts = torch.zeros(2, 18)
    starts[:, 2] = torch.tensor([0., .02])
    starts[:, 5] = -.04
    starts[:, 6] = starts[:, 10] = starts[:, 15] = 1.
    with torch.no_grad():
        state = starts
        survival = torch.ones(2)
        total = torch.zeros(2)
        for t in range(4):
            action = policy.actor(state)
            following, probs, _ = wm.step(state, action)
            reward, continuation = task_reward(state, action, following, probs, wm.spec)
            distance = torch.linalg.vector_norm(following[..., :2], dim=-1) * wm.spec["scales"]["position"]
            extra = 4 * wm.spec["source_config"]["reward"]["horizontal_cost"]
            total = total + .997**t * survival * (reward - extra * distance * wm.spec["control_dt"])
            state = torch.where((survival > 0).unsqueeze(-1), following, state)
            survival = survival * continuation
    stats = update_actor_critic(wm, policy.actor, critic, starts,
                                torch.optim.Adam(policy.actor.parameters()), torch.optim.Adam(critic.parameters()),
                                horizon=4, actor_terminal_value=False, imagination_horizontal_cost_multiplier=5.)
    assert stats["imagined_return"] == total.mean().item()


def test_cheap_selector_and_budget_categories():
    cheap = [{"iteration": i, "safe_landing_rate": safe, "safe_in_target_rate": target, "mean_return": reward}
             for i, safe, target, reward in [(0, .85, .8, 100.), (5, .9, .7, 95.),
                                              (10, .9, .8, 50.), (15, .9, .8, 51.), (20, .9, .8, 51.)]]
    assert online.select_checkpoint(cheap)["iteration"] == 20
    budget = online.InteractionBudget()
    for kind, episodes, steps in [("train", 6, 1010), ("train", 6, 1250), ("heldout", 50, 0),
                                  ("cheap", 20, 0), ("full", 100, 0)]:
        budget.record(kind, episodes, steps)
    assert asdict(budget) == dict(train_episodes=12, train_transitions=2260, heldout_episodes=50,
                                  cheap_episodes=20, full_episodes=100)


def test_margin_uses_incoming_vector_speed_and_only_first_predicted_contact():
    margin = TouchdownMargin(speed_m_s=1., cost_per_m_s=10.)
    # All three rows start supported; lift off, land, lift off and land again.
    seen = torch.zeros(3, dtype=torch.bool)
    velocity = torch.tensor([[.03, .04, 0.], [0., 0., -.02], [0., -.01, 0.]], requires_grad=True)
    for i, (ground, next_ground) in enumerate([(1, 1), (1, 0), (0, 0), (0, 1), (1, 0), (0, 1)]):
        state = torch.zeros(3, 18)
        state[:, 16] = ground
        state[:, 3:6] = velocity
        following = torch.zeros_like(state)
        following[:, 16] = next_ground
        cost, seen = first_touchdown_cost(state, following, seen, 50., margin)
        expected = [15., 0., 0.] if i == 3 else [0., 0., 0.]
        torch.testing.assert_close(cost, torch.tensor(expected))
        if i == 3:
            cost[0].backward()
            torch.testing.assert_close(velocity.grad[0], torch.tensor([300., 400., 0.]))


def test_margin_context_first_landing_after_liftoff_and_default_restoration(monkeypatch):
    # Exercise update_actor_critic itself with controlled contact times and
    # zero base rewards, so its imagined return isolates the first impact.
    def rollout():
        policy, critic, model = initialize_models(example_checkpoint())
        starts = torch.zeros(1, 18)
        starts[:, 16] = 1.
        starts[:, 3] = 2. / model.spec["scales"]["velocity"]
        ground = iter([1, 0, 1, 0, 1])
        def step(state, action):
            following = state.clone()
            following[:, 16] = next(ground)
            # A zero-valued edge lets backward run also in the no-penalty arm.
            return following, None, action[:, 0]
        monkeypatch.setattr(model, "step", step)
        import mun_agent.imagination as imagination
        monkeypatch.setattr(imagination, "task_reward", lambda s, a, nxt, p, spec:
                            (0 * a[:, 0], torch.ones_like(a[:, 0])))
        return update_actor_critic(model, policy.actor, critic, starts,
                                   torch.optim.Adam(policy.actor.parameters()), torch.optim.Adam(critic.parameters()),
                                   horizon=5, gamma=.5, actor_terminal_value=False)["imagined_return"]
    base = rollout()
    with imagined_touchdown_margin(TouchdownMargin(1., 10.)):
        assert rollout() == base - .5**2 * 10.
    assert rollout() == base == 0.


@pytest.mark.parametrize("frozen", [True, False])
def test_two_start_one_round_smoke(tmp_path, monkeypatch, frozen):
    from mun_agent import adapt as core
    saved = example_checkpoint()
    policy, _, _ = initialize_models(saved)
    replay, meta = collect_policy(policy, seed=27, episodes=2)
    validation = generate_initial_states(SimConfig(), 2, 2026, "validation")
    test = generate_initial_states(SimConfig(), 2, 2026, "test")
    inputs = online.OnlineInputs(saved, replay, meta, validation, test,
                                 {"train_reset_seeds": [998], "heldout_reset_seeds": [999]})
    c = json.loads((AGENT / "configs/adapt_online.json").read_text())
    c.update(iterations=1, collect_episodes=2, validate_every=1, wm_updates=1, actor_updates=1,
             imagined_batch=4, wm_batch=4, horizon=2, cpu_threads=1)
    models = initialize_models(saved)
    monkeypatch.setattr(core, "initialize_models", lambda checkpoint: models)
    out = tmp_path / "online"
    run = online.run_online(inputs, c, out, frozen=frozen, smoke=True)
    assert run["budget"]["train_episodes"] == 2 and run["budget"]["train_transitions"] > 0
    assert [run["budget"][k] for k in ("cheap_episodes", "full_episodes", "heldout_episodes")] == [4, 4, 0]
    assert run["model_updates"] == int(not frozen) and run["actor_updates"] == 1
    assert all(torch.equal(w, saved["world_model"][k]) for k, w in models[2].state_dict().items()) == frozen
    assert any(not torch.equal(w, saved["actor"][k]) for k, w in models[0].actor.state_dict().items())
    chosen = torch.load(out / "best.pt", weights_only=True)
    assert chosen["iteration"] == run["selected_cheap"]["iteration"]
    assert run["smoke"] and run["g1"] is None
    assert not (out / "test.json").exists()


def test_failed_a1_retains_provenance_checks_and_fixed_world_recipe():
    report = dict(domain="P1", passed=False, selected_recipe=None,
                  source_checkpoint_sha256=online.CHECKPOINT_SHA, source_replay_sha256=online.PREFILL_SHA,
                  train_reset_seeds=list(range(30)), heldout_reset_seeds=list(range(100, 150)))
    inputs = online.OnlineInputs(None, None, {"replay_sha256": online.PREFILL_SHA}, {}, {}, report)
    online._verify_a1_provenance(inputs, "P1")
    config = json.loads((AGENT / "configs/adapt_online.json").read_text())
    for name in ("main", "fallback"):
        settings = online._run_recipe(config, name)
        assert (settings["wm_lr"], settings["source_fraction"], settings["wm_updates"]) == (1e-4, .5, 200)
    with pytest.raises(ValueError, match="target domain"):
        online._verify_a1_provenance(inputs, "P2")
    for key, value in (("source_checkpoint_sha256", ""), ("source_replay_sha256", ""),
                       ("train_reset_seeds", []), ("heldout_reset_seeds", [])):
        invalid = online.OnlineInputs(None, None, inputs.source_meta, {}, {}, dict(report, **{key: value}))
        with pytest.raises(ValueError, match="A1 provenance"):
            online._verify_a1_provenance(invalid, "P1")


def test_a2_cli_forwards_failed_a1_to_runner(tmp_path, monkeypatch):
    from mun_agent import dataset

    report = tmp_path / "summary.json"
    report.write_text(json.dumps({"domain": "P1", "passed": False, "selected_recipe": None}))
    calls = []
    monkeypatch.setattr(online, "run_online", lambda inputs, config, out, **kw: calls.append((inputs, config, kw)))
    monkeypatch.setattr(dataset, "load_dataset", lambda path: (None, {}))
    monkeypatch.setattr(online.nominal, "load_manifest", lambda path, split: {})
    script = AGENT / "scripts/adapt.py"
    monkeypatch.setattr(sys, "argv", [str(script), "a2", "--domain", "P1", "--seed", "31", "--a1", str(report)])
    runpy.run_path(str(script), run_name="__main__")
    [(inputs, settings, options)] = calls
    assert inputs.a1["passed"] is False
    assert settings["wm_lr"] == 1e-4 and settings["source_fraction"] == .5
    assert options["domain"] == "P1" and options["recipe"] == "main" and not options["frozen"]


def test_p2_requires_matching_three_seed_p1_gate(tmp_path):
    paths = []
    for seed in (31, 32, 33):
        path = tmp_path / f"seed-{seed}.json"
        row = dict(seed=seed, domain="P1", frozen=False, smoke=False, recipe="main", completed=True,
                   g1=True, budget={"train_episodes": 150}, selected_full={"n_episodes": 100, "safe_count": 80})
        path.write_text(json.dumps(row))
        paths.append(path)
    online._authorize_online("P2", "main", False, paths, None)
    row["selected_full"]["safe_count"] = 79
    paths[-1].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="passed G-A2"):
        online._authorize_online("P2", "main", False, paths, None)
