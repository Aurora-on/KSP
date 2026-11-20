"""Reject corrupt checkpoints without changing the current episode or RNGs."""
import copy

import numpy as np
import pytest

from mun_lander import LanderEnv


@pytest.mark.parametrize("field,value", [
    ("episode_return", float("nan")),
    ("episode_return", float("inf")),
    ("steps", -1),
    ("steps", 1.5),
    ("initial_fuel", 601.0),
    ("initial_fuel", 0.0),
    ("contact", True),
    ("stable_time", 0.1),
    ("touchdown_count", -1),
    ("touchdown_count", 0.5),
])
def test_corrupt_checkpoint_rejected_without_mutating_episode(field, value):
    env = LanderEnv()
    env.reset(seed=37)
    env.step(np.array([0.3, 0.02, -0.01, 0.0]))
    before = env.get_state()
    corrupted = copy.deepcopy(before)
    corrupted[field] = value
    # A different but valid position makes partial restoration observable.
    corrupted["state"]["position"][0] += 1.0
    with pytest.raises((ValueError, TypeError)):
        env.set_state(corrupted)
    assert env.get_state() == before


@pytest.mark.parametrize("rng_field", ["rng", "action_rng"])
def test_bad_rng_restoration_is_atomic(rng_field):
    env = LanderEnv()
    env.reset(seed=101)
    before = env.get_state()
    corrupted = copy.deepcopy(before)
    corrupted["state"]["position"][0] += 1.0
    # For action_rng failure, first install a different valid environment RNG
    # in the candidate; rejection must not leave that partial change behind.
    corrupted["rng"] = np.random.default_rng(999).bit_generator.state
    corrupted[rng_field] = {"bit_generator": "NotARealGenerator"}
    with pytest.raises((ValueError, TypeError)):
        env.set_state(corrupted)
    assert env.get_state() == before
