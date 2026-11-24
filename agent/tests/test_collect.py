import json

import numpy as np
import pytest
from mun_lander import LanderEnv, SimConfig
from mun_lander.controller import ScriptedController
from mun_lander.evaluation import generate_initial_states, validate_manifest

from mun_agent import collect as collection


def test_collection_on_nominal_simulator():
    config = SimConfig()
    replay, metadata = collection.collect(seed=2026, episodes=3, noise_levels=(0, 0.1),
                                          random_fraction=1 / 3)
    manifest = json.loads(json.dumps(metadata))["manifest"]
    validate_manifest(manifest)
    assert manifest["split"] == "train"
    assert manifest["sha256"] == metadata["train_manifest_sha256"]
    seeds = [start["seed"] for start in manifest["starts"]]
    assert seeds == metadata["reset_seeds"]
    for split in ("validation", "test"):
        fixed = generate_initial_states(config, 100, 2026, split)
        assert set(seeds).isdisjoint(entry["seed"] for entry in fixed["starts"])

    records = metadata["episode_provenance"]
    assert {(r["controller"], r["noise_std"]) for r in records} == {
        ("scripted", 0), ("scripted", 0.1), ("random", None)}
    assert metadata["env_steps"] == sum(len(ep.action) for ep in replay.episodes)
    env = LanderEnv(config)
    controller = ScriptedController(config)
    for ep, start, details in zip(replay.episodes, manifest["starts"], records, strict=True):
        assert details["reset_seed"] == start["seed"] and details["start_id"] == start["id"]
        obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
        np.testing.assert_array_equal(obs, ep.obs[0])
        assert len(ep.action) == details["steps"]
        if details["noise_std"] == 0:
            np.testing.assert_array_equal(ep.action[0], controller(obs))
        else:
            assert not np.array_equal(ep.action[0], controller(obs))
        for t, action in enumerate(ep.action):
            obs, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(ep.obs[t + 1], obs)
            np.testing.assert_array_equal(action, info["applied_action"])
            assert ep.reward[t] == np.float32(reward)
            assert ep.terminated[t] == terminated and ep.truncated[t] == truncated
        assert terminated or truncated
    env.close()


def test_records_environment_applied_action_and_only_reads_that_info_key(monkeypatch):
    from gymnasium.spaces import Box

    applied = np.array([0.9375, -1.0, 0.25, -0.75], dtype=np.float32)
    initial = np.zeros(14, dtype=np.float32)
    initial[6] = 1
    final = initial.copy()
    final[13] = 7

    class OneStepEnv:
        def __init__(self, config):
            self.action_space = Box(np.array([0, -1, -1, -1], np.float32), np.ones(4, np.float32))

        def reset(self, *, seed, options):
            return initial.copy(), {}

        def step(self, action):
            assert not np.array_equal(action, applied)
            return final.copy(), -100.0, False, True, {"applied_action": applied.copy()}

        def close(self):
            pass

    monkeypatch.setattr(collection, "LanderEnv", OneStepEnv)
    replay, _ = collection.collect(seed=7, episodes=1, noise_levels=(0,), random_fraction=1.0)
    ep = replay.episodes[0]
    np.testing.assert_array_equal(ep.action[0], applied)
    np.testing.assert_array_equal(ep.obs[-1], final)
    assert ep.truncated[-1] and not ep.terminated[-1]


@pytest.mark.parametrize("override", [
    {"noise_levels": ()}, {"noise_levels": (-0.1,)}, {"noise_levels": (float("inf"),)},
    {"random_fraction": float("nan")}, {"random_fraction": 2}, {"episodes": 0}, {"seed": -1},
])
def test_rejects_invalid_collection_parameters(override):
    params = {"seed": 0, "episodes": 1, "noise_levels": (0,), "random_fraction": 0}
    with pytest.raises(ValueError):
        collection.collect(**(params | override))
