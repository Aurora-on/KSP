"""Minimal vector environment integration for the model/policy team.

This produces transitions only; it does not implement SAC or Dreamer training.
The last observation and terminal flag must be stored before resetting an
episode. SAME_STEP vector autoreset supplies these in final_obs/final_info.
"""

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import mun_lander  # registers MunLander-v0

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("MunLander-v0") for _ in range(4)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
observations, infos = envs.reset(seed=7)
for _ in range(400):
    actions = envs.action_space.sample()
    next_observations, rewards, terminated, truncated, infos = envs.step(actions)
    for index in range(envs.num_envs):
        ended = terminated[index] or truncated[index]
        transition_next = infos["final_obs"][index] if ended else next_observations[index]
        # Replay row: observations[index], actions[index], rewards[index],
        # transition_next, terminated[index], truncated[index]. For a time
        # limit, an algorithm may bootstrap from transition_next; never use
        # the next episode's reset observation as the preceding final state.
    observations = next_observations
envs.close()
print("Collected 1600 vectorized transitions, including episode endings.")
