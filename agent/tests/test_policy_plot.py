import json
from types import SimpleNamespace

import pytest

from mun_agent.policy_commands import plot_curves


def test_plot_records_separate_stages_at_identical_transition_counts(tmp_path):
    run = tmp_path / "warm-start"
    run.mkdir()
    rows = [{"phase": phase, "env_steps": steps, "safe_landing_rate": rate, "n_episodes": 100,
             "actor_updates": updates} for phase, steps, rate, updates in [
                 ("untrained", 67962, 0., 0), ("bc_only", 67962, 1., 0), ("online", 69941, 1., 30)]]
    (run / "validation.jsonl").write_text("\n".join(map(json.dumps, rows)))
    args = SimpleNamespace(run=[run], out=tmp_path / "result.svg")
    plot_curves(args)
    svg = args.out.read_text()
    assert svg.count("67,962 steps") == 2
    for label in ("Behavior cloning", "Untrained", "30 actor updates", "69,941 steps", "0/100", "100/100"):
        assert label in svg
    with pytest.raises(FileExistsError):
        plot_curves(args)


def test_nominal_plot_shows_target_accuracy_separately_from_safety(tmp_path):
    run = tmp_path / "seed31"
    run.mkdir()
    rows = [dict(phase=phase, env_steps=steps, starts_sha256="same", safe_landing_rate=1.,
                 safe_in_target_rate=target, mean_return=reward, mean_touchdown_speed_m_s=impact)
            for phase, steps, target, reward, impact in
            [("bc_only", 1100, 1., 90., .7), ("online", 1400, 0., 82., 1.1)]]
    (run / "validation.jsonl").write_text("\n".join(map(json.dumps, rows)))
    args = SimpleNamespace(run=[run], out=tmp_path / "nominal.svg")
    plot_curves(args)
    contents = args.out.read_text()
    for label in ("Nominal 6DOF", "Safe landing in target", "Mean episode return", "first-touchdown speed"):
        assert label in contents
