"""Store a replay together with the simulator manifest and collection provenance."""
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from mun_lander.env import TERMINAL
from mun_lander.evaluation import validate_manifest
from mun_lander.state import LanderState

from .replay import Replay


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_dataset(path, replay, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        replay_path = stage / "replay.npz"
        replay.save(replay_path)
        record = dict(metadata, replay_sha256=file_hash(replay_path))
        (stage / "collection.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        stage.rename(path)


def _validate_provenance(replay, metadata):
    manifest = metadata["manifest"]
    starts = manifest["starts"]
    if len(replay.episodes) != len(starts):
        raise ValueError("episode count does not match collection manifest")
    expected = {"episodes": len(starts), "split": manifest["split"], "seed": manifest["seed"],
                "reset_seeds": [start["seed"] for start in starts],
                "env_steps": sum(map(len, replay.episodes))}
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"collection {key} does not match manifest or replay")

    rows = metadata.get("episode_provenance")
    if rows is not None and len(rows) != len(starts):
        raise ValueError("episode provenance count does not match manifest")
    counts = Counter()
    for i, (ep, start) in enumerate(zip(replay.episodes, starts, strict=True)):
        if not np.array_equal(ep.obs[0], LanderState.from_dict(start["initial_state"]).observation()):
            raise ValueError(f"episode {i} initial observation does not match manifest")
        if rows is None:
            continue
        row = rows[i]
        for key, value in (("start_id", start["id"]), ("reset_seed", start["seed"]), ("steps", len(ep))):
            if row.get(key) != value:
                raise ValueError(f"episode {i} provenance {key} does not match manifest or replay")
        if "status" in row:
            status = row["status"]
            # Legacy RSSM collections predate the per-episode terminal labels.
            truncated = status in ("timeout", "out_of_bounds")
            if status not in TERMINAL or ep.truncated[-1] != truncated or ep.terminated[-1] == truncated:
                raise ValueError(f"episode {i} status does not match terminal flags")
            counts[status] += 1
    if "status_counts" in metadata and (sum(counts.values()) != len(starts) or dict(counts) != metadata["status_counts"]):
        raise ValueError("collection status_counts do not match episode labels")


def load_dataset(path):
    path = Path(path)
    metadata = json.loads((path / "collection.json").read_text(encoding="utf-8"))
    manifest = metadata["manifest"]
    validate_manifest(manifest)
    if manifest["split"] != "train" or manifest["sha256"] != metadata["train_manifest_sha256"]:
        raise ValueError("collection manifest mismatch")
    replay_path = path / "replay.npz"
    if file_hash(replay_path) != metadata["replay_sha256"]:
        raise ValueError("collection replay checksum mismatch")
    replay = Replay.load(replay_path)
    _validate_provenance(replay, metadata)
    return replay, metadata
