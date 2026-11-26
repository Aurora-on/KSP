"""Offline tests for the candidate-scene measurement helpers."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "measure_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "measure_candidate", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load measure_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEASURE = _load_script()


class PlaneFitTests(unittest.TestCase):
    def test_recovers_a_known_plane(self):
        rows = [
            {
                "north_m": float(north),
                "east_m": float(east),
                "height_m": 0.5 * north + 0.2 * east + 3.0,
                "valid": True,
            }
            for north in (-2, 0, 2)
            for east in (-2, 0, 2)
        ]
        result = MEASURE.fit_plane(rows)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["gradient_north"], 0.5)
        self.assertAlmostEqual(result["gradient_east"], 0.2)
        self.assertAlmostEqual(result["height_at_target_m"], 3.0)
        self.assertAlmostEqual(
            result["slope_deg"],
            math.degrees(math.atan(math.hypot(0.5, 0.2))),
        )
        self.assertEqual(result["samples_used"], 9)

    def test_ignores_invalid_samples(self):
        rows = [
            {
                "north_m": 0.0,
                "east_m": 0.0,
                "height_m": 999.0,
                "valid": False,
            },
            {
                "north_m": -1.0,
                "east_m": 0.0,
                "height_m": 1.0,
                "valid": True,
            },
            {
                "north_m": 1.0,
                "east_m": 0.0,
                "height_m": 1.0,
                "valid": True,
            },
            {
                "north_m": 0.0,
                "east_m": -1.0,
                "height_m": 1.0,
                "valid": True,
            },
            {
                "north_m": 0.0,
                "east_m": 1.0,
                "height_m": 1.0,
                "valid": True,
            },
        ]
        result = MEASURE.fit_plane(rows)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["height_at_target_m"], 1.0)
        self.assertAlmostEqual(result["slope_deg"], 0.0)
        self.assertEqual(result["samples_used"], 4)


class ResourceSummaryTests(unittest.TestCase):
    def test_aggregates_duplicate_resource_parts(self):
        vessel = SimpleNamespace(
            resources=SimpleNamespace(
                all=[
                    SimpleNamespace(name="LiquidFuel", amount=10.0, max=20.0, density=5.0),
                    SimpleNamespace(name="LiquidFuel", amount=5.0, max=10.0, density=5.0),
                    SimpleNamespace(name="Oxidizer", amount=4.0, max=8.0, density=5.0),
                ]
            )
        )
        rows = {row["name"]: row for row in MEASURE.resource_summary(vessel)}
        self.assertEqual(rows["LiquidFuel"]["amount"], 15.0)
        self.assertEqual(rows["LiquidFuel"]["max"], 30.0)
        self.assertEqual(rows["LiquidFuel"]["mass_kg"], 75.0)
        self.assertEqual(rows["Oxidizer"]["mass_kg"], 20.0)


if __name__ == "__main__":
    unittest.main()
