import json

import pytest

from mun_agent import nominal_risk


def test_decline_pairs_cover_clone_and_adjacent_online_and_cache_by_model(tmp_path, monkeypatch):
    rates = [("untrained", 0, .1), ("bc_only", 0, 1.), ("online", 1, .9),
             ("online", 2, .8), ("online", 4, .85), ("online", 6, .85)]
    rows = [{"phase": phase, "iteration": it, "safe_landing_rate": rate, "starts_sha256": "valid"}
            for phase, it, rate in rates]
    pairs = nominal_risk.declines(rows)
    assert [(a["phase"], a["iteration"], b["iteration"]) for a,b in pairs] == [
        ("bc_only", 0, 1), ("online", 1, 2)]
    with pytest.raises(ValueError, match="increase"):
        nominal_risk.declines(rows[:3]+rows[2:])

    m = {"sha256": "valid", "starts": [{"id": "start"}]}
    run = tmp_path / "run"
    (run/"validation").mkdir(parents=True)
    (run/"checkpoints").mkdir()
    (run/"validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for row in rows[1:4]:
        name = nominal_risk._point_name(row)
        (run/"checkpoints"/(name+".pt")).write_text(name)
        (run/"validation"/(name+".json")).write_text(json.dumps(
            dict(starts_sha256="valid", safe_landing_rate=row["safe_landing_rate"],
                 episodes=[{"start_id": "start", "status": "crash"}])))
    visits = {"first_touchdowns": [{"start_id": "start", "status": "crash"}]}
    calls = []
    def measure(model, policy, manifest):
        calls.append(policy.name)
        return visits
    monkeypatch.setattr(nominal_risk, "load_training_world_model", lambda path,device: None)
    monkeypatch.setattr(nominal_risk, "load_policy", lambda path,device: path)
    monkeypatch.setattr(nominal_risk, "policy_risk", measure)
    result = nominal_risk._drop_records(run, 31, m, tmp_path/"risk", "cpu")
    assert len(result) == 2 and result[0]["after"] == result[1]["before"]
    assert calls == ["bc-only.pt","online-000001.pt","online-000002.pt"]
    assert nominal_risk._drop_records(run,31,m,tmp_path/"risk","cpu") == result
    assert len(calls) == 3
    (run/"checkpoints/online-000001.pt").write_text("different weights")
    nominal_risk._drop_records(run,31,m,tmp_path/"risk","cpu")
    assert calls[-1] == "online-000001.pt" and len(calls) == 4


def test_matching_archived_outcome_identifies_wrong_reroll():
    v = {"first_touchdowns":[{"start_id":"a","status":"timeout"}]}
    official = {"starts_sha256":"v","episodes":[{"start_id":"a","status":"crash"}]}
    with pytest.raises(ValueError, match="outcomes"):
        nominal_risk._verify_visits(v,official,{"sha256":"v","starts":[{"id":"a"}]})
