"""Complete simulator episodes and uniformly covered transition sequences."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def _float_array(value, name):
    if np.asarray(value).dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _flags(value, length, name):
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector of length {length}")
    return array.copy()


@dataclass(frozen=True, eq=False)
class Episode:
    """L applied native actions and rewards, bracketed by L+1 raw observations."""

    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __post_init__(self):
        action = _float_array(self.action, "action")
        if action.ndim != 2 or action.shape[1] != 4 or len(action) == 0:
            raise ValueError("action must have shape [L, 4] with L > 0")
        if (action[:, 0] < 0).any() or (np.abs(action) > 1).any():
            raise ValueError("action outside simulator bounds")

        length = len(action)
        obs = _float_array(self.obs, "obs")
        reward = _float_array(self.reward, "reward")
        if obs.shape != (length + 1, 14) or reward.shape != (length,):
            raise ValueError("expected obs[L+1, 14] and reward[L]")
        terminated = _flags(self.terminated, length, "terminated")
        truncated = _flags(self.truncated, length, "truncated")
        done = terminated | truncated
        if done[:-1].any() or not done[-1]:
            raise ValueError("only the last transition of a complete episode may end it")

        for name, array in (
            ("obs", obs), ("action", action), ("reward", reward),
            ("terminated", terminated), ("truncated", truncated),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def __len__(self):
        return len(self.action)


class Replay:
    """Sample padded windows from complete episodes without crossing resets."""

    def __init__(self, episodes: Iterable[Episode]):
        self.episodes = tuple(episodes)
        if not all(isinstance(ep, Episode) for ep in self.episodes):
            raise TypeError("replay expects Episode instances")
        self._lengths = np.asarray([len(ep) for ep in self.episodes], dtype=np.int64)

    def sample(self, batch_size: int, length: int, rng: np.random.Generator, *, frame_features=None):
        """Return padded windows; flags and mask describe transitions, not frames."""
        if batch_size < 1 or length < 1 or not self.episodes:
            raise ValueError("sampling needs episodes and positive batch_size and length")

        # An L-step episode has L+T-1 possible starts, from 1-T through L-1.
        # Every transition belongs to exactly T of these equally likely windows.
        limits = np.cumsum(self._lengths + length - 1)
        draws = rng.integers(limits[-1], size=batch_size)
        episode_ids = np.searchsorted(limits, draws, side="right")
        offsets = np.r_[0, limits[:-1]]
        starts = draws - offsets[episode_ids] - (length - 1)

        shape = (batch_size, length)
        batch = {
            "obs": np.empty((batch_size, length + 1, 14), np.float32),
            "action": np.zeros((*shape, 4), np.float32),
            "reward": np.zeros(shape, np.float32),
        }
        if frame_features is not None:
            batch["features"] = np.empty((batch_size, length + 1, frame_features[0].shape[-1]), np.float32)
        for key in ("mask", "is_first", "is_last", "terminated", "truncated"):
            batch[key] = np.zeros(shape, bool)

        # Group rows by episode so a large batch can gather frames in NumPy.
        for ep_id in np.unique(episode_ids):
            ep = self.episodes[ep_id]
            rows = np.flatnonzero(episode_ids == ep_id)
            frames = starts[rows, None] + np.arange(length + 1)
            frame_indices = np.clip(frames, 0, len(ep))
            batch["obs"][rows] = ep.obs[frame_indices]
            if frame_features is not None:
                batch["features"][rows] = frame_features[ep_id][frame_indices]
            steps = frames[:, :-1]
            valid = (steps >= 0) & (steps < len(ep))
            local_rows, cols = np.nonzero(valid)
            source = steps[local_rows, cols]
            dest = (rows[local_rows], cols)
            for key in ("action", "reward", "terminated", "truncated"):
                batch[key][dest] = getattr(ep, key)[source]
            batch["mask"][dest] = True
            batch["is_first"][dest] = source == 0
            batch["is_last"][dest] = source == len(ep) - 1

        batch["is_terminal"] = batch["terminated"] | batch["truncated"]
        return batch

    def save(self, path):
        """Write a pickle-free NPZ, replacing the destination after a complete write."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shapes = {"obs": (0, 14), "action": (0, 4), "reward": (0,),
                  "terminated": (0,), "truncated": (0,)}
        payload = {"lengths": self._lengths}
        for key, shape in shapes.items():
            if self.episodes:
                payload[key] = np.concatenate([getattr(ep, key) for ep in self.episodes])
            else:
                dtype = bool if key in ("terminated", "truncated") else np.float32
                payload[key] = np.empty(shape, dtype)
        out = NamedTemporaryFile(dir=path.parent, delete=False)
        temp = Path(out.name)
        try:
            with out:
                np.savez_compressed(out, **payload)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            lengths = archive["lengths"]
            if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or (lengths < 1).any():
                raise ValueError("invalid replay episode lengths")
            lengths = lengths.tolist()
            steps = sum(lengths)
            shapes = {"obs": (steps + len(lengths), 14), "action": (steps, 4),
                      "reward": (steps,), "terminated": (steps,), "truncated": (steps,)}
            fields = {key: archive[key] for key in shapes}
            if any(fields[key].shape != shape for key, shape in shapes.items()):
                raise ValueError("replay arrays do not match recorded lengths")
        episodes = []
        start = 0
        for i, length in enumerate(lengths):
            stop = start + length
            episodes.append(Episode(
                obs=fields["obs"][start + i:stop + i + 1],
                **{key: fields[key][start:stop] for key in
                   ("action", "reward", "terminated", "truncated")},
            ))
            start = stop
        return cls(episodes)
