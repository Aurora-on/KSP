from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.error import ResetNeeded

from mun_lander import LanderEnv, SimConfig
from mun_lander.state import LanderState
from mun_lander.math3d import quaternion_from_euler


def initial_state(**updates):
    values = dict(
        position=[0.0, 0.0, 100.0], velocity=[0.0, 0.0, -1.0],
        quaternion=[1.0, 0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0], fuel=600.0,
    )
    values.update(updates)
    return values


def reset_at(env, **updates):
    return env.reset(seed=42, options={"initial_state": initial_state(**updates)})


def test_gymnasium_contract():
    check_env(LanderEnv(), skip_render_check=True)


def test_seed_and_checkpoint_reproduce_trajectory():
    env = LanderEnv()
    first, _ = env.reset(seed=125)
    second, _ = env.reset(seed=125)
    np.testing.assert_array_equal(first, second)
    env.step(np.array([0.4, 0.1, -0.1, 0.02], dtype=np.float32))
    checkpoint = env.get_state()
    actions = [np.array([0.4, -0.04, 0.03, 0.01], dtype=np.float32)] * 4
    expected = [env.step(action) for action in actions]
    env.set_state(checkpoint)
    actual = [env.step(action) for action in actions]
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        for key in ["time", "stable_time", "main_fuel_used", "rcs_fuel_used", "status"]:
            assert a[4][key] == b[4][key]


def test_action_bounds_are_clipped_and_nonfinite_actions_rejected():
    env = LanderEnv()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([3.0, -4.0, 5.0, 0.0]))
    np.testing.assert_array_equal(info["applied_action"], [1.0, -1.0, 1.0, 0.0])
    assert info["action_clipped"]
    for action in [np.array([np.nan, 0, 0, 0]), np.array([0, np.inf, 0, 0]), np.zeros(3)]:
        with pytest.raises(ValueError):
            env.step(action)


def test_high_speed_touchdown_is_crash_and_reports_preimpact_speed():
    env = LanderEnv()
    reset_at(env, position=[0, 0, 0.001], velocity=[0, 0, -8])
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["status"] == "crash"
    assert not info["is_success"]
    assert info["touchdown"]["speed"] >= 8.0
    assert info["touchdown"]["time"] < env.config.control_dt
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(4))


def test_safe_landing_requires_window_and_precision_is_separate():
    env = LanderEnv()
    reset_at(env, position=[10, 0, 0.001], velocity=[0, 0, -0.1])
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert not terminated and not truncated
    assert info["contact"]
    assert info["status"] == "stabilizing"
    assert not info["is_success"]
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["status"] == "safe_landing"
    assert info["is_success"] and not info["safe_in_target"]
    assert info["stable_time"] >= env.config.landing.stability_time - 1e-9
    assert info["time"] - info["touchdown"]["time"] >= env.config.landing.stability_time - 1e-9
    assert info["horizontal_error"] == pytest.approx(10.0)
    assert info["total_fuel_used"] == pytest.approx(info["main_fuel_used"] + info["rcs_fuel_used"])
    assert "episode" in info


def test_contact_window_keeps_dynamics_active():
    env = LanderEnv()
    reset_at(env, position=[0, 0, 0.001], velocity=[0, 0, -0.1])
    obs, _, _, _, info = env.step(np.zeros(4))
    assert info["contact"]
    obs2, _, terminated, truncated, info2 = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert obs2[2] > obs[2] or obs2[5] > obs[5] or info2["status"] == "crash"
    assert not info2["is_success"]
    assert info2["main_fuel_used"] > info["main_fuel_used"]


def test_touchdown_detected_when_thrust_reverses_descent_inside_substep():
    env = LanderEnv()
    reset_at(env, position=[0, 0, 1e-5], velocity=[0, 0, -0.01])
    # The trajectory crosses z=0 and then returns above it within one physics
    # step. An endpoint-only collision check would miss this valid touchdown.
    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert info["touchdown"] is not None
    assert 0 < info["touchdown"]["time"] < env.config.physics_dt
    assert not info["is_success"]


def test_recontact_preserves_first_touchdown_and_restarts_window():
    env = LanderEnv()
    reset_at(env, position=[0, 0, 0.001], velocity=[0, 0, -0.1])
    _, _, _, _, first_info = env.step(np.zeros(4))
    first_touchdown = first_info["touchdown"]
    _, _, _, _, lifted = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert not lifted["contact"] and lifted["stable_time"] == 0
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        assert not terminated and not truncated
        if info["touchdown_count"] == 2:
            break
    assert info["touchdown_count"] == 2
    assert info["touchdown"] == first_touchdown
    assert info["last_touchdown"]["time"] > first_touchdown["time"]
    assert info["stable_time"] < env.config.control_dt
    checkpoint = env.get_state()
    env.step(np.zeros(4))
    env.set_state(checkpoint)
    restored = env.get_state()
    for key in ["touchdown", "last_touchdown", "touchdown_count", "stable_time", "contact"]:
        assert restored[key] == checkpoint[key]


def test_fuel_exhaustion_and_timeout_are_failures_with_distinct_flags():
    env = LanderEnv()
    reset_at(env, fuel=0.001)
    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert terminated and not truncated
    assert info["status"] == "fuel_exhausted" and not info["is_success"]
    assert info["main_fuel_used"] == pytest.approx(0.001)
    assert env.state.fuel >= 0
    timeout_env = LanderEnv(replace(SimConfig(), max_time=0.1))
    reset_at(timeout_env)
    _, _, terminated, truncated, info = timeout_env.step(np.zeros(4))
    assert not terminated and truncated
    assert info["status"] == "timeout" and not info["is_success"]


def test_initial_state_serialization_and_observation_layout():
    state = LanderState.from_dict(initial_state(position=[1, 2, 3], velocity=[4, 5, 6], fuel=123))
    restored = LanderState.from_dict(state.to_dict())
    np.testing.assert_array_equal(state.observation(), restored.observation())
    np.testing.assert_array_equal(state.observation(), [1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 0, 0, 0, 123])
    assert state.observation().dtype == np.float32


@pytest.mark.parametrize("changes", [
    {"quaternion": quaternion_from_euler(np.deg2rad(11), 0, 0)},
    {"angular_velocity": [0, 0, 0.3]},
])
def test_soft_impact_still_fails_for_unsafe_attitude_or_rate(changes):
    env = LanderEnv()
    reset_at(env, position=[0, 0, 0.001], velocity=[0, 0, -0.1], **changes)
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated and info["status"] == "crash"
    assert info["touchdown"]["speed"] < env.config.landing.max_touchdown_speed


def test_continued_ground_torque_can_fail_stability_window():
    env = LanderEnv()
    reset_at(env, position=[0, 0, 0], velocity=[0, 0, 0],
             quaternion=quaternion_from_euler(np.deg2rad(9), 0, 0))
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(np.array([0, 1, 0, 0]))
        if terminated or truncated:
            break
    assert terminated and info["status"] == "crash"
    assert info["tilt_deg"] > env.config.landing.max_tilt_deg
    assert info["stable_time"] < env.config.landing.stability_time
