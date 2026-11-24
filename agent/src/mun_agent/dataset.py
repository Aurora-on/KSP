"""Store a replay together with the simulator manifest and collection provenance."""
import hashlib
import json
import tempfile
from pathlib import Path

from mun_lander.evaluation import validate_manifest

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
    if len(replay.episodes) != len(manifest["starts"]):
        raise ValueError("episode count does not match collection manifest")
    return replay, metadata
