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
