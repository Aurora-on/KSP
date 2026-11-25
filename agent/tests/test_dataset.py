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


def test_collection_checks_manifest_replay_and_event_alignment(tmp_path):
    from copy import deepcopy

    from mun_agent.collect import collect
    from mun_agent.dataset import save_dataset
    from mun_agent.replay import Replay

    replay, metadata = collect(seed=17, episodes=2, noise_levels=(0.0,), random_fraction=0.0)
    output = tmp_path / "dataset"
    save_dataset(output, replay, metadata)
    record_path = output / "collection.json"
    original = json.loads(record_path.read_text())

    def reject(update, error):
        record = deepcopy(original)
        update(record)
        record_path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match=error):
            load_dataset(output)

    reject(lambda r: r["reset_seeds"].reverse(), "reset_seeds")
    reject(lambda r: r["episode_provenance"].reverse(), "start_id")
    reject(lambda r: r["episode_provenance"][0].update(steps=0), "steps")
    reject(lambda r: r["episode_provenance"][0].update(reset_seed=0), "reset_seed")
    reject(lambda r: r["episode_provenance"][0].update(status="timeout"), "terminal flags")
    reject(lambda r: r.update(status_counts={}), "status_counts")

    legacy_record = deepcopy(original)
    del legacy_record["status_counts"]
    for row in legacy_record["episode_provenance"]:
        del row["status"]
    record_path.write_text(json.dumps(legacy_record))
    assert len(load_dataset(output)[0].episodes) == 2

    swapped = tmp_path / "swapped"
    save_dataset(swapped, Replay(reversed(replay.episodes)), metadata)
    with pytest.raises(ValueError, match="initial observation"):
        load_dataset(swapped)
