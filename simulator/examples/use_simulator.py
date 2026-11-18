"""Run an engineering reference landing, without a KSP or ML dependency.

Run from the project directory after `pip install -e .`:
    python examples/use_simulator.py
"""

from mun_lander import LanderEnv, SimConfig
from mun_lander.controller import ScriptedController

config = SimConfig()
env = LanderEnv(config)
controller = ScriptedController(config)
observation, info = env.reset(seed=7)
while True:
    action = controller(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info["episode"])
        break
env.close()
