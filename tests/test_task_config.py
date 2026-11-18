"""Validate the declarative task contract without KSP or scientific packages."""

import tomllib
import unittest
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mun_landing.toml"


class TaskConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("rb") as handle:
            cls.config = tomllib.load(handle)

    def test_readiness_flags_stay_false_while_numbers_are_open(self):
        for key in ("initial_state_ready", "success_thresholds_ready",
                    "reset_step_validated"):
            with self.subTest(key=key):
                self.assertFalse(self.config[key])

    def test_qualitative_task_decisions_are_recorded(self):
        self.assertEqual(self.config["target_body"], "Mun")
        self.assertEqual(self.config["scenario"]["randomize_initial"],
                         ["position", "velocity", "attitude"])
        self.assertIn("attitude_quaternion", self.config["observation"]["fields"])
        self.assertIn("main_engine_throttle", self.config["action"]["fields"])
        self.assertEqual(self.config["safety"]["stable_window_s"], 5.0)
        self.assertEqual(self.config["experiment"]["design"],
                         "2x2_pretraining_by_imagination")
        self.assertEqual(self.config["experiment"]["validation_success_threshold"], 0.80)
        self.assertEqual(self.config["experiment"]["min_training_seeds"], 3)

    def test_unresolved_numbers_remain_explicitly_tbd(self):
        self.assertEqual(self.config["scenario"]["control_hz"], "TBD")
        self.assertEqual(self.config["scenario"]["save_name"], "TBD")
        self.assertEqual(self.config["safety"]["max_touchdown_speed_m_s"], "TBD")
        self.assertEqual(self.config["evaluation"]["ood_ranges"], "TBD")
        self.assertEqual(self.config["observation"]["normalization"], "TBD")
        self.assertEqual(self.config["observation"]["main_engine_fuel_resource"], "TBD")

    def test_step_api_is_gymnasium_compatible_core(self):
        self.assertEqual(self.config["implementation"]["step_api"], "gymnasium_compatible_core")


if __name__ == "__main__":
    unittest.main()
