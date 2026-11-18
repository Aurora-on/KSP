"""Vector aggregation must work across mixed flying/contact/reset phases."""

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np
import pytest

import mun_lander
from mun_lander.controller import ScriptedController


@pytest.mark.parametrize("mode", [AutoresetMode.SAME_STEP, AutoresetMode.NEXT_STEP])
def test_vector_env_crosses_touchdown_and_autoreset(mode):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("MunLander-v0") for _ in range(4)], autoreset_mode=mode)
    policy = ScriptedController()
    observations, _ = envs.reset(seed=7)
    completions = 0
    saw_mixed_contact = False
    try:
        for _ in range(800):
            actions = np.stack([policy(observation) for observation in observations])
            observations, _, terminated, truncated, infos = envs.step(actions)
            assert observations.shape == (4, 14)
            assert np.all(np.isfinite(observations))
            if "_touchdown" in infos:
                mask = infos["_touchdown"]
                saw_mixed_contact |= bool(np.any(mask) and not np.all(mask))
            ended = terminated | truncated
            if np.any(ended):
                completions += int(np.count_nonzero(ended))
                final_infos = infos["final_info"] if mode == AutoresetMode.SAME_STEP else infos
                assert np.all(final_infos["is_success"][ended])
                if mode == AutoresetMode.SAME_STEP:
                    for index in np.flatnonzero(ended):
                        assert infos["final_obs"][index][2] == 0.0
                        assert observations[index][2] > 0.0
        assert completions >= 8
        assert saw_mixed_contact
    finally:
        envs.close()
