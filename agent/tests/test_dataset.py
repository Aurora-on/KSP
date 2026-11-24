import json
from pathlib import Path

import pytest

from mun_agent.cli import main
from mun_agent.dataset import load_dataset


def test_collect_command_persists_validated_dataset(tmp_path):
    config = Path(__file__).resolve().parents[1] / "configs/debug.json"
    output = tmp_path / "data"
    main(["collect", "--config", str(config), "--out", str(output),
          "--episodes", "2", "--seed", "7", "--device", "cpu"])
    replay, metadata = load_dataset(output)
    assert sum(map(len, replay.episodes)) == metadata["env_steps"]
    assert len(replay.episodes) == 2
    record = output / "collection.json"
    metadata["manifest"]["starts"][0]["seed"] += 1
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(output)
