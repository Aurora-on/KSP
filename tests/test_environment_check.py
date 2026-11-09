"""Offline checks for readiness rules; no game or scientific packages required."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_environment import check_ksp, check_python, overall_status, parse_args


class EnvironmentCheckTests(unittest.TestCase):
    def test_supported_python_range(self):
        for version in ((3, 12, 0), (3, 13, 9), (3, 14, 4)):
            with self.subTest(version=version):
                self.assertEqual(check_python(version)["status"], "PASS")
        for version in ((3, 11, 9), (3, 15, 0), (4, 0, 0)):
            with self.subTest(version=version):
                self.assertEqual(check_python(version)["status"], "FAIL")

    def test_ci_does_not_require_a_game_or_accelerator(self):
        checks = {"cpu": {"status": "PASS"}, "mps": {"status": "NOT_AVAILABLE"},
                  "ksp": check_ksp(None)}
        self.assertEqual(checks["ksp"]["status"], "NOT_CHECKED")
        self.assertEqual(overall_status(checks), "PASS")
        checks["ksp"] = check_ksp(None, required=True)
        self.assertEqual(overall_status(checks), "FAIL")

    def test_missing_game_path_is_optional_unless_required(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "absent-game")
            self.assertEqual(check_ksp(missing)["status"], "NOT_CHECKED")
            required = check_ksp(missing, required=True)
            self.assertEqual(required["status"], "FAIL")
            self.assertNotIn(directory, json.dumps(required))

    def test_plugin_metadata_is_checked_without_exposing_its_path(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "GameData" / "kRPC" / "kRPC.version"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(json.dumps({
                "VERSION": {"MAJOR": 0, "MINOR": 7, "PATCH": 0},
                "KSP_VERSION": {"MAJOR": 1, "MINOR": 12, "PATCH": 5},
            }), encoding="utf-8")
            result = check_ksp(directory, required=True)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["plugin_version"], "0.7.0")
            self.assertNotIn(directory, json.dumps(result))
            metadata_path.write_text(json.dumps({
                "VERSION": {"MAJOR": 0, "MINOR": 6, "PATCH": 0},
            }), encoding="utf-8")
            self.assertEqual(check_ksp(directory, required=True)["status"], "FAIL")
            metadata_path.write_text("not-json", encoding="utf-8")
            self.assertEqual(check_ksp(directory, required=True)["status"], "FAIL")

    def test_cli_is_offline_and_optional_by_default(self):
        defaults = parse_args([])
        self.assertFalse(defaults.require_ksp)
        self.assertIsNone(defaults.output)
        explicit = parse_args(["--require-ksp", "--output", "reports/check.json"])
        self.assertTrue(explicit.require_ksp)
        self.assertEqual(explicit.output, Path("reports/check.json"))


if __name__ == "__main__":
    unittest.main()
