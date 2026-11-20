"""Keep the committed acceptance summary consistent with the code it describes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mun_lander import SimConfig
from mun_lander.evaluation import generate_initial_states

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs" / "validation-summary.json"


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_summary_schema_and_sources_are_current(summary):
    assert summary["schema_version"] == 2
    assert summary["source_sha256"], "summary must record source hashes"
    for relative, recorded in summary["source_sha256"].items():
        source = ROOT / relative
        assert source.is_file(), f"summary references missing source {relative}"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == recorded, (
            f"{relative} changed; rebuild docs/validation-summary.json with "
            "python examples/build_validation_summary.py")


def test_summary_describes_the_default_configuration(summary):
    assert summary["nominal_config"] == SimConfig().to_dict()
    assert summary["stability_window_s"] == SimConfig().landing.stability_time


def test_recorded_start_manifests_are_reproducible(summary):
    # The hash depends on NumPy's generator, so only check in the recorded stack.
    if summary["environment"]["numpy"] != np.__version__:
        pytest.skip(f"recorded NumPy {summary['environment']['numpy']}, running {np.__version__}")
    config = SimConfig.from_dict(summary["nominal_config"])
    for key, spec in summary["starts"].items():
        manifest = generate_initial_states(config, spec["count"], spec["root_seed"], spec["split"])
        assert manifest["sha256"] == spec["sha256"], key
