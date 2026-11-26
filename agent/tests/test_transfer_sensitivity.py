import importlib
from pathlib import Path

import numpy as np
import pytest

from mun_agent.transfer import rescale_fuel
from mun_lander import SimConfig
from mun_lander.evaluation import generate_initial_states, rollout


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("transfer_sensitivity")


def test_rescale_preserves_tank_fraction_and_input():
    obs = np.arange(4 * 14, dtype=np.float32).reshape(2, 2, 14)
    obs[..., 13] = [[0, 1250], [2500, 5000]]
    obs.setflags(write=False)
    result = rescale_fuel(obs, source_capacity_kg=600, target_capacity_kg=5000)
    assert result.dtype == obs.dtype and not np.shares_memory(result, obs)
    np.testing.assert_array_equal(result[..., :13], obs[..., :13])
    np.testing.assert_array_equal(result[..., 13], [[0, 150], [300, 600]])
    np.testing.assert_array_equal(obs[..., 13], [[0, 1250], [2500, 5000]])
    obs_int = np.ones(14, dtype=int)
    assert rescale_fuel(obs_int, source_capacity_kg=1, target_capacity_kg=2)[13] == .5


@pytest.mark.parametrize("index,value", [(0, np.nan), (13, np.inf), (8, -np.inf)])
def test_rescale_rejects_nonfinite_observations(index, value):
    obs = np.zeros(14)
    obs[index] = value
    with pytest.raises(ValueError):
        rescale_fuel(obs, source_capacity_kg=600, target_capacity_kg=5000)


@pytest.mark.parametrize("source,target", [(0, 1), (-1, 1), (np.inf, 1), (1, 0), (1, np.nan)])
def test_rescale_rejects_invalid_capacities(source, target):
    with pytest.raises(ValueError):
        rescale_fuel(np.zeros(14), source_capacity_kg=source, target_capacity_kg=target)


def test_rescale_rejects_wrong_observation_width():
    with pytest.raises(ValueError):
        rescale_fuel(np.zeros((3, 13)), source_capacity_kg=1, target_capacity_kg=1)


def test_condition_parameters(sweep):
    cases = sweep.conditions()
    base = SimConfig().vehicle
    assert cases["nominal"].config().vehicle == base
    for name, thrust in [("thrust_ksp_limited", 60_000), ("thrust_ksp_full", 240_000)]:
        cfg = cases[name].config().vehicle
        assert cfg.max_thrust / base.max_thrust == pytest.approx((thrust / 11_460) / (9_000 / 1_800), rel=1e-12)
        assert cfg.inertia == base.inertia
    inertia = np.asarray(base.inertia)
    np.testing.assert_allclose(cases["inertia_alpha_est"].config().vehicle.inertia, inertia * (.75 / .07), rtol=1e-12)
    np.testing.assert_array_equal(cases["inertia_pitch_yaw"].config().vehicle.inertia, inertia * [4, 4, 1])
    np.testing.assert_array_equal(cases["inertia_roll_3"].config().vehicle.inertia, inertia * [1, 1, 3])
    roll_proxy = cases["torque_roll_quarter"].config().vehicle
    assert roll_proxy.inertia == base.inertia
    np.testing.assert_array_equal(roll_proxy.rcs_torque, np.asarray(base.rcs_torque) * [1, 1, .25])


def test_one_step_delay_resets_and_keeps_queued_action(sweep):
    delay = sweep.OneStepDelay()
    first = np.array([.4, -.3, .2, 1.], dtype=np.float32)
    second = np.array([.8, 1., -.5, .1], dtype=np.float32)
    original = first.copy()
    np.testing.assert_array_equal(delay(first), np.zeros(4))
    first[:] = 0  # A controller can reuse its output buffer on the next call.
    np.testing.assert_array_equal(delay(second), original)
    np.testing.assert_array_equal(delay(first), second)
    delay.reset()
    np.testing.assert_array_equal(delay(second), np.zeros(4))


def test_wilson_interval(sweep):
    assert sweep.wilson95(0, 100) == pytest.approx([0., .036994], abs=1e-5)
    assert sweep.wilson95(100, 100) == pytest.approx([.963006, 1.], abs=1e-5)
    assert sweep.wilson95(5, 10) == pytest.approx([.236593, .763407], abs=1e-5)


def test_two_validation_starts_match_simulator(sweep):
    cfg = SimConfig()
    starts = generate_initial_states(cfg, 2, 48, "validation")["starts"]
    rows = sweep.evaluate_pair(sweep.conditions()["nominal"], "scripted_nominal", starts, None, None)
    assert [r["start_id"] for r in rows] == [s["id"] for s in starts]
    for row, start in zip(rows, starts, strict=True):
        reference, trajectory = rollout(cfg, start["initial_state"], start["seed"], "scripted", record=True)
        assert row["status"] == reference["status"]
        assert row["return"] == reference["return"]
        assert row["touchdown_speed_m_s"] == reference["touchdown"]["speed"]
        saturated = np.any(np.abs(trajectory["actions"][:, 1:]) >= .99, axis=1)
        assert row["rcs_saturated_steps"] == saturated.sum()
        assert row["rcs_saturation_fraction"] == pytest.approx(saturated.mean())
    summary = sweep.summarize(rows)
    assert summary["n_episodes"] == 2 and summary["safe_count"] == sum(r["is_success"] for r in rows)
    assert summary["mean_return"] == pytest.approx(np.mean([r["return"] for r in rows]))
    assert summary["rcs_saturation_fraction"] == pytest.approx(
        sum(r["rcs_saturated_steps"] for r in rows) / sum(r["transitions"] for r in rows))
