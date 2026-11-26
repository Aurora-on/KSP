"""Offline tests for the frozen initial-state sampler."""

from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from ksp_env.contracts import load_config
from ksp_env.initial_states import InitialState, sample_bank, sample_initial_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InitialStateSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_config(PROJECT_ROOT / "configs" / "mun_landing.toml").raw[
            "scenario"
        ]

    def test_samples_respect_frozen_ranges(self):
        rng = random.Random(7)
        for _ in range(200):
            state = sample_initial_state(rng, self.scenario)
            self.assertGreaterEqual(state.clearance_m, 80.0)
            self.assertLessEqual(state.clearance_m, 120.0)
            self.assertGreaterEqual(state.position_m[0], -30.0)
            self.assertLessEqual(state.position_m[0], 30.0)
            self.assertGreaterEqual(state.position_m[1], -30.0)
            self.assertLessEqual(state.position_m[1], 30.0)
            self.assertAlmostEqual(state.position_m[2], state.clearance_m)
            self.assertGreaterEqual(state.velocity_m_s[0], -2.0)
            self.assertLessEqual(state.velocity_m_s[0], 2.0)
            self.assertGreaterEqual(state.velocity_m_s[1], -2.0)
            self.assertLessEqual(state.velocity_m_s[1], 2.0)
            self.assertGreaterEqual(state.velocity_m_s[2], -8.0)
            self.assertLessEqual(state.velocity_m_s[2], -3.0)
            self.assertGreaterEqual(state.roll_deg, 0.0)
            self.assertLessEqual(state.roll_deg, 360.0)

            body_z_norm = math.sqrt(sum(value * value for value in state.body_z_direction))
            self.assertAlmostEqual(body_z_norm, 1.0)
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, state.body_z_direction[2]))))
            self.assertLessEqual(tilt_deg, 12.0 + 1e-12)
            angular_norm = math.sqrt(
                sum(value * value for value in state.angular_velocity_rad_s)
            )
            self.assertLessEqual(angular_norm, 0.05 + 1e-12)

    def test_bank_is_reproducible_for_a_seed(self):
        first = sample_bank(self.scenario, seed=33, count=5)
        second = sample_bank(self.scenario, seed=33, count=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_bank_rejects_nonpositive_count(self):
        with self.assertRaises(ValueError):
            sample_bank(self.scenario, seed=1, count=0)

    def test_state_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            InitialState(
                clearance_m=100.0,
                position_m=(math.nan, 0.0, 100.0),
                velocity_m_s=(0.0, 0.0, -5.0),
                body_z_direction=(0.0, 0.0, 1.0),
                roll_deg=0.0,
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
