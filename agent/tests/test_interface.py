import json
from dataclasses import replace

import numpy as np
import pytest
from mun_lander import LanderEnv, SimConfig
from mun_lander.math3d import rotation_matrix
from mun_lander.wrappers import NormalizedObservation, SymmetricAction

from mun_agent.interface import (
    interface_spec,
    preprocess,
    to_internal_action,
    to_native_action,
    validate_interface_spec,
)


def test_features_follow_simulator_rotation_and_scales():
    config = SimConfig()
    config = replace(config, vehicle=replace(config.vehicle, fuel_capacity=300.0))
    env = NormalizedObservation(LanderEnv(config))
    obs, _ = env.unwrapped.reset(seed=41)
    obs[6:10] = [0.2, -0.4, 0.8, 0.4]
    negated = obs.copy()
    negated[6:10] *= -1

    features = preprocess(np.stack([obs, negated])[None], interface_spec(config))
    scaled = env.observation(obs)
    columns = rotation_matrix(obs[6:10])[:, :2].T.ravel()
    expected = np.concatenate((scaled[:6], columns, scaled[10:]))
    assert features.shape == (1, 2, 16)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[0, 0], expected, atol=1e-7)
    np.testing.assert_array_equal(features[0, 0], features[0, 1])


def test_actions_match_simulator_wrapper_and_round_trip():
    wrapper = SymmetricAction(LanderEnv())
    actions = np.random.default_rng(91).uniform(-1, 1, size=(4, 7, 4)).astype(np.float32)
    actions[0, :3, 0] = [-1, 0, 1]
    native = to_native_action(actions)
    expected = np.stack([wrapper.action(a) for a in actions.reshape(-1, 4)])
    np.testing.assert_allclose(native.reshape(-1, 4), expected, atol=4e-8)
    np.testing.assert_allclose(to_internal_action(native), actions, atol=1e-7)
    assert native.dtype == np.float32


def test_checkpoint_spec_survives_json_and_rejects_incompatible_config():
    config = SimConfig()
    saved = json.loads(json.dumps(interface_spec(config)))
    validate_interface_spec(saved, config)
    with pytest.raises(ValueError, match="control_dt"):
        validate_interface_spec(saved, replace(config, control_dt=0.2))
    with pytest.raises(ValueError, match="source_config"):
        validate_interface_spec(saved, replace(config, max_time=60.0))
    saved["scales"]["position"] = 120.0
    with pytest.raises(ValueError, match="scales"):
        validate_interface_spec(saved, config)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="quaternion"):
        preprocess(np.zeros(14), interface_spec(SimConfig()))
    with pytest.raises(ValueError, match="action"):
        to_native_action([np.nan, 0, 0, 0])
    with pytest.raises(ValueError, match="dimension 4"):
        to_internal_action([0.5])
