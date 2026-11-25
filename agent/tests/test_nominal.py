import json

import numpy as np
import pytest
import torch
from mun_lander import SimConfig
from mun_lander.evaluation import evaluate, generate_initial_states

from mun_agent.deterministic import DeterministicModel
from mun_agent.imagination import Actor
from mun_agent.interface import deterministic_spec
from mun_agent.nominal import collect_policy, evaluate_nominal, load_manifest, prepare_manifest, risk_report
from mun_agent.policy import Policy
from mun_agent.vertical import vertical_manifest


def test_fixed_nominal_starts_round_trip_and_separation(tmp_path):
    saved = prepare_manifest(tmp_path / "val.json", "validation", episodes=2)
    assert saved == generate_initial_states(SimConfig(), 2, 2026, "validation")
    assert prepare_manifest(tmp_path / "val.json", "validation", episodes=2) == saved
    test = prepare_manifest(tmp_path / "test.json", "test", episodes=2)
    assert {s["seed"] for s in test["starts"]}.isdisjoint(s["seed"] for s in saved["starts"])
    with pytest.raises(ValueError, match="requested split"):
        load_manifest(tmp_path / "val.json", "test")
    with pytest.raises(ValueError, match="size"):
        prepare_manifest(tmp_path / "val.json", "validation", episodes=4)
    path = tmp_path / "vertical.json"
    path.write_text(json.dumps(vertical_manifest(split="validation", episodes=2, seed=2026)))
    with pytest.raises(ValueError, match="nominal"):
        load_manifest(path, "validation")


def test_full_actions_and_official_first_impact_statistics(tmp_path):
    native = np.array([0., .2, -.3, .1], dtype=np.float32)
    policy = lambda obs: native.copy()
    replay, meta = collect_policy(policy, seed=2026, episodes=2, noise_std=0.)
    for ep in replay.episodes:
        np.testing.assert_array_equal(ep.action, np.broadcast_to(native, ep.action.shape))
    m = prepare_manifest(tmp_path / "val.json", "validation", episodes=2)
    assert set(meta["reset_seeds"]).isdisjoint(s["seed"] for s in m["starts"])
    got = evaluate_nominal(policy, m)
    official = evaluate(SimConfig(), m, policy)
    assert got["safe_landing_rate"] == official["safe_landing_rate"]
    contacts = [e["touchdown"] for e in official["episodes"]]
    assert got["touchdown_count"] == 2 and got["status_counts"] == {"crash": 2}
    assert got["max_touchdown_tilt_deg"] == max(t["tilt_deg"] for t in contacts)
    assert got["mean_touchdown_speed_m_s"] == np.mean([t["speed"] for t in contacts])
    assert got["mean_total_fuel_used_kg"] == np.mean([e["total_fuel_used"] for e in official["episodes"]])
    assert got["mean_landing_error_m"] == np.mean([t["horizontal_error"] for t in contacts])
    no_contact = evaluate_nominal(lambda obs: np.array([1.,0.,0.,0.]), m)
    assert no_contact["touchdown_count"] == 0 and no_contact["mean_landing_error_m"] is None


def test_deterministic_visits_vs_prefill_and_closed_loop_pairs(tmp_path):
    torch.set_num_threads(1)
    spec = deterministic_spec(SimConfig())
    actor = Actor(8, vertical_only=False)
    model = DeterministicModel(spec, width=8)
    with torch.no_grad():
        for module in (actor, model):
            for par in module.parameters():
                par.zero_()
        actor.network[-1].bias.copy_(torch.tensor([-8., .2, -.3, .1]))
    policy = Policy(actor, spec)
    data, provenance = collect_policy(policy, seed=2026, episodes=2, noise_std=.1)
    m = prepare_manifest(tmp_path / "val.json", "validation", episodes=1)
    result = risk_report(model, actor, policy, m, data, provenance["episode_provenance"])
    assert result["policy_visited"]["episodes"] == 1 and result["permanent_prefill"]["episodes"] == 2
    assert result["paired_returns"]["summary"]["count"] == 4
    for key in ("policy_visited", "permanent_prefill"):
        source = result[key]
        assert source["open_loop"]["1"]["all"]["count"] == source["transitions"]
        for h in (1,5,15,32):
            row = source["open_loop"][str(h)]
            assert row["all"]["count"] == source["transitions"] - (h-1)*source["episodes"]
            assert 0 < row["near_ground"]["count"] < row["all"]["count"]
            assert row["all"]["horizontal_position_error_m"] > 0
    assert result["policy_visited"]["touchdown_distributions"]["crash"]["speed_m_s"]["count"] == 1
    test = prepare_manifest(tmp_path / "test.json", "test", episodes=1)
    with pytest.raises(ValueError, match="requested split"):
        risk_report(model, actor, policy, test, data, provenance["episode_provenance"])
