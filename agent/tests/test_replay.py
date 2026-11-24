import numpy as np
import pytest

from mun_agent.replay import Episode, Replay


def episode(start, length, truncated=False):
    # The first observation component uniquely identifies a transition across episodes.
    obs = np.zeros((length + 1, 14), np.float32)
    obs[:, 0] = np.arange(start, start + length + 1)
    obs[:, 6] = 1
    action = np.zeros((length, 4), np.float32)
    action[:, 0] = np.linspace(0, 1, length)
    ended = np.arange(length) == length - 1
    return Episode(obs, action, obs[:-1, 0], ended & (not truncated), ended & truncated)


def test_sampling_covers_every_transition_equally():
    lengths = [1, 3, 9]
    replay = Replay(episode(start, length) for start, length in zip([0, 1, 4], lengths))
    draws, width = 16_000, 5
    batch = replay.sample(draws, width, np.random.default_rng(2026))
    counts = np.bincount(batch["reward"][batch["mask"]].astype(int), minlength=sum(lengths))
    p = width / sum(length + width - 1 for length in lengths)
    # Each window contains any particular transition at most once (binomial count).
    tolerance = 5 * np.sqrt(draws * p * (1 - p))
    assert np.all(np.abs(counts - draws * p) < tolerance), counts


@pytest.mark.parametrize("width", [1, 3, 8])
def test_padded_transitions_and_true_successors(width):
    episodes = (episode(10, 1), episode(30, 4, truncated=True))
    batch = Replay(episodes).sample(200, width, np.random.default_rng(83))
    assert batch["obs"].shape == (200, width + 1, 14)
    assert batch["action"].shape == (200, width, 4)
    for i, valid in enumerate(batch["mask"]):
        ep = episodes[int(batch["obs"][i, 0, 0] >= 30)]
        cols = np.flatnonzero(valid)
        np.testing.assert_array_equal(cols, np.arange(cols[0], cols[-1] + 1))
        frames = batch["obs"][i, :, 0]
        src = (frames[:-1][valid] - ep.obs[0, 0]).astype(int)
        np.testing.assert_array_equal(frames[1:][valid], ep.obs[src + 1, 0])
        np.testing.assert_array_equal(batch["action"][i, valid], ep.action[src])
        assert np.all(frames[:cols[0]] == ep.obs[0, 0])
        assert np.all(frames[cols[-1] + 2:] == ep.obs[-1, 0])
        for key, expected in (
            ("reward", ep.reward[src]), ("is_first", src == 0),
            ("is_last", src == len(ep) - 1), ("terminated", ep.terminated[src]),
            ("truncated", ep.truncated[src]), ("is_terminal", src == len(ep) - 1),
        ):
            assert not batch[key][i, ~valid].any()
            np.testing.assert_array_equal(batch[key][i, valid], expected)


def test_npz_roundtrip_and_failed_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "data" / "samples.npz"
    episodes = (episode(10, 2), episode(30, 3, truncated=True))
    replay = Replay(episodes)
    replay.save(path)

    def fail(out, **arrays):
        out.write(b"incomplete archive")
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError):
        replay.save(path)
    restored = Replay.load(path)
    assert len(restored.episodes) == len(episodes)
    assert list(path.parent.iterdir()) == [path]
    for actual, expected in zip(restored.episodes, episodes):
        for key in ("obs", "action", "reward", "terminated", "truncated"):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))


def test_invalid_episode_and_archive(tmp_path):
    ep = episode(0, 2)
    with pytest.raises(ValueError):
        Episode(ep.obs[:-1], ep.action, ep.reward, ep.terminated, ep.truncated)
    with pytest.raises(ValueError):
        Episode(ep.obs, ep.action, ep.reward, [True, True], ep.truncated)
    with pytest.raises(ValueError):
        Episode(ep.obs, ep.action + 2, ep.reward, ep.terminated, ep.truncated)
    fields = {key: getattr(ep, key) for key in ("obs", "action", "reward", "terminated", "truncated")}
    path = tmp_path / "corrupt.npz"
    np.savez(path, lengths=[3], **fields)
    with pytest.raises(ValueError):
        Replay.load(path)
