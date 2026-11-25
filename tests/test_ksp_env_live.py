"""Opt-in live smoke test for the KSP adapter.

Run with a game and kRPC server available:

    KSP_LIVE=1 .venv/bin/python -m unittest discover -s tests -p 'test_ksp_env_live.py' -v

The smoke case is read-only and skipped by default so CI and offline checks do
not need a running game. The reset/step case loads a save and writes controls;
the current Mun test save is in orbit, so live touchdown detection is not covered.
"""

import os
import unittest
from pathlib import Path

from ksp_env import Action, KSPEnv

TEST_CONFIG = os.environ.get(
    "KSP_TEST_CONFIG",
    str(Path(__file__).resolve().parents[1] / "configs" / "ksp_adapter_test.toml"),
)


@unittest.skipUnless(os.environ.get("KSP_LIVE") == "1", "set KSP_LIVE=1 with KSP and kRPC running")
class LiveAdapterTests(unittest.TestCase):
    def test_connect_and_read_diagnostic_state(self):
        env = KSPEnv.from_config()
        try:
            env.connect()
            scene = env.connection.krpc.game_scene.name
            self.assertIn(scene, {"flight", "space_center"})
            if scene != "flight":
                self.skipTest("enter a vessel in the Flight scene for diagnostic state")
            state = env.read_diagnostic_state()
            self.assertEqual(state["scene"], "flight")
            self.assertIsInstance(state["ut_s"], float)
            self.assertGreaterEqual(state["ut_s"], 0.0)
            self.assertTrue(state["vessel_name"])
            self.assertTrue(state["body_name"])
        finally:
            env.close()


@unittest.skipUnless(
    os.environ.get("KSP_LIVE_RESET_STEP") == "1",
    "set KSP_LIVE_RESET_STEP=1 to load a save and advance one control interval",
)
class LiveResetStepTests(unittest.TestCase):
    def test_reset_and_one_zero_action_step(self):
        env = KSPEnv.from_config(TEST_CONFIG)
        try:
            env.connect()
            observation, reset_info = env.reset()
            self.assertEqual(reset_info.scene, "flight")
            self.assertGreater(reset_info.game_state_loads, 0)
            self.assertGreaterEqual(observation.main_engine_fuel, 0.0)
            result = env.step(Action())
            control_hz = float(env.config.raw["scenario"]["control_hz"])
            self.assertTrue(result.info["episode_evaluated"])
            self.assertIsNotNone(result.info["ut_delta_s"])
            self.assertGreater(result.info["ut_delta_s"], 0.0)
            self.assertAlmostEqual(result.info["control_hz"], control_hz)
            self.assertAlmostEqual(result.info["control_interval_s"], 1.0 / control_hz)
            self.assertAlmostEqual(result.info["dt_s"], result.info["ut_delta_s"])
            self.assertGreaterEqual(result.info["physics_ticks"], 1)
            self.assertIn(result.info["status"], {"flying", "stabilizing", "safe_landing"})
            self.assertFalse(result.info["vessel_destroyed"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
