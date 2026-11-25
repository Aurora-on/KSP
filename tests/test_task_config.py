"""Validate the declarative task contract without KSP or scientific packages."""

import math
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
                         ["position", "velocity", "attitude", "angular_velocity"])
        self.assertIn("attitude_quaternion", self.config["observation"]["fields"])
        self.assertIn("main_engine_throttle", self.config["action"]["fields"])
        self.assertEqual(self.config["safety"]["stable_window_s"], 5.0)
        self.assertEqual(self.config["experiment"]["design"],
                         "2x2_pretraining_by_imagination")
        self.assertEqual(self.config["experiment"]["validation_success_threshold"], 0.80)
        self.assertEqual(self.config["experiment"]["min_training_seeds"], 3)

    def test_approved_nominal_task_values(self):
        scenario, safety = self.config["scenario"], self.config["safety"]
        self.assertEqual((scenario["control_hz"], scenario["episode_horizon_s"]), (10.0, 90.0))
        self.assertEqual(scenario["descent_altitude_m"], [80.0, 120.0])
        for axis in ("x", "y"):
            self.assertEqual(scenario["initial_position_range"][axis + "_m"], [-30., 30.])
            self.assertEqual(scenario["initial_velocity_range"][axis + "_m_s"], [-2., 2.])
        self.assertEqual(scenario["initial_velocity_range"]["z_m_s"], [-8., -3.])
        self.assertEqual(scenario["initial_attitude_range"]["max_tilt_deg"], 12.)
        self.assertEqual(scenario["initial_angular_velocity_range"]["max_norm_rad_s"], 0.05)
        self.assertEqual((safety["max_touchdown_speed_m_s"], safety["max_touchdown_tilt_deg"],
                          safety["target_radius_m"]), (2., 10., 5.))
        self.assertAlmostEqual(math.radians(safety["max_touchdown_angular_velocity_deg_s"]), 0.2)
        observation = self.config["observation"]
        self.assertEqual(set(observation["diagnostic_only_fields"]), {"mass", "inertia_tensor"})
        self.assertTrue(set(observation["fields"]).isdisjoint(observation["diagnostic_only_fields"]))

    def test_unresolved_numbers_remain_explicitly_tbd(self):
        self.assertEqual(self.config["scenario"]["save_name"], "TBD")
        self.assertEqual(self.config["scenario"]["target_position"], "TBD")
        self.assertEqual(self.config["evaluation"]["ood_ranges"], "TBD")
        self.assertEqual(self.config["observation"]["normalization"], "TBD")
        self.assertEqual(self.config["observation"]["main_engine_fuel_resource"], "TBD")
        self.assertEqual(self.config["observation"]["body_to_task_rotation_xyzw"], "TBD")

    def test_step_api_is_gymnasium_compatible_core(self):
        self.assertEqual(self.config["implementation"]["step_api"], "gymnasium_compatible_core")


if __name__ == "__main__":
    unittest.main()
