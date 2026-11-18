from dataclasses import replace

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np

from mun_lander import LanderEnv, SimConfig
from mun_lander.controller import ScriptedController
from mun_lander.wrappers import NormalizedObservation, SymmetricAction


def test_wrappers_preserve_raw_transition_and_explicit_scales():
    raw = LanderEnv()
    wrapped = SymmetricAction(NormalizedObservation(LanderEnv()))
    raw_obs, _ = raw.reset(seed=71)
    scaled_obs, _ = wrapped.reset(seed=71)
    scale = np.array([100] * 3 + [10] * 3 + [1] * 4 + [0.5] * 3 + [600], dtype=np.float32)
    np.testing.assert_allclose(scaled_obs, raw_obs / scale)
    raw_result = raw.step(np.array([0.25, 0.1, -0.2, 0.3], dtype=np.float32))
    wrapped_result = wrapped.step(np.array([-0.5, 0.1, -0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(wrapped_result[0], raw_result[0] / scale)
    assert wrapped_result[1:4] == raw_result[1:4]
    np.testing.assert_array_equal(wrapped_result[4]["applied_action"], raw_result[4]["applied_action"])


def test_vector_autoreset_retains_terminal_observation():
    config = replace(SimConfig(), max_time=0.1)
    vector = gym.vector.SyncVectorEnv(
        [lambda: LanderEnv(config), lambda: LanderEnv(config)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    try:
        obs, _ = vector.reset(seed=21)
        reset_obs, reward, terminated, truncated, infos = vector.step(np.zeros((2, 4), dtype=np.float32))
        assert not terminated.any() and truncated.all()
        assert infos["_final_obs"].all()
        for index in range(2):
            # Final observation is post-step telemetry; SAME_STEP's return is a new reset.
            assert infos["final_obs"][index][2] < obs[index][2]
            assert not np.array_equal(infos["final_obs"][index], reset_obs[index])
    finally:
        vector.close()


def test_vector_mixed_flight_contact_and_autoreset_info_schemas():
    vector = gym.vector.SyncVectorEnv(
        [lambda: LanderEnv() for _ in range(4)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    controllers = [ScriptedController() for _ in range(4)]
    mixed_contact_seen = False
    ended_count = 0
    try:
        observations, _ = vector.reset(seed=[7, 11, 17, 29])
        for _ in range(800):
            actions = np.stack([controller(obs) for controller, obs in zip(controllers, observations)])
            observations, _, terminated, truncated, infos = vector.step(actions)
            ended = terminated | truncated
            ended_count += int(ended.sum())
            if ended.any():
                assert infos["_final_obs"][ended].all()
            if "_touchdown" in infos:
                mask = infos["_touchdown"]
                mixed_contact_seen |= bool(mask.any() and not mask.all())
        assert ended_count > 0
        assert mixed_contact_seen
    finally:
        vector.close()
