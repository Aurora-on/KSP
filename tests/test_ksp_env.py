"""Offline tests for the KSP environment adapter.

No game, kRPC server, NumPy or Gymnasium is required.
"""

import dataclasses
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from ksp_env import (
    Action,
    ConfigurationError,
    EnvConfig,
    InfrastructureError,
    KSPConnectionError,
    KSPEnv,
    Observation,
    ResetError,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ksp_adapter_test.toml"


class ConfigurationTests(unittest.TestCase):
    def test_adapter_test_profile_is_complete_but_not_task_ready(self):
        config = load_config(CONFIG_PATH)
        env = KSPEnv(config)
        env._require_observation_config()
        env._require_action_config()
        self.assertEqual(config.raw["status"], "adapter_test_only")
        self.assertEqual(config.raw["target_body"], "Kerbin")
        self.assertEqual(config.raw["scenario"]["save_name"], "persistent")
        self.assertEqual(config.raw["observation"]["frame"], "target_surface")
        self.assertEqual(config.raw["action"]["rcs_semantics"], "normalized_input")
        for key in ("initial_state_ready", "success_thresholds_ready", "reset_step_validated"):
            self.assertFalse(config.raw[key])

    def test_repository_config_loads(self):
        config = load_config()
        self.assertEqual(config.address, "127.0.0.1")
        self.assertEqual(config.rpc_port, 50000)
        self.assertEqual(config.stream_port, 50001)
        self.assertEqual(config.server_version, "0.7.0")
        self.assertEqual(config.raw["target_body"], "Mun")
        self.assertTrue(is_tbd(config.raw["scenario"]["target_position"]))

    def test_tbd_helpers_are_explicit(self):
        self.assertTrue(is_tbd("TBD"))
        self.assertTrue(is_tbd(" tbd "))
        self.assertFalse(is_tbd("0.1"))
        with self.assertRaises(ConfigurationError):
            require_defined("TBD", "value")
        with self.assertRaises(ConfigurationError):
            require_number("TBD", "value")
        self.assertEqual(require_number(2.5, "value"), 2.5)

    def test_invalid_connection_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text(
                "\n".join(
                    [
                        "[connection]",
                        'address = "127.0.0.1"',
                        "rpc_port = 0",
                        "stream_port = 50001",
                        'client_version = "0.7.0"',
                        'server_version = "0.7.0"',
                        'game_version = "1.12.5"',
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_missing_config_is_reported(self):
        with self.assertRaises(ConfigurationError):
            load_config("/definitely/not/a/config.toml")


class ContractTests(unittest.TestCase):
    def test_valid_observation(self):
        observation = Observation(
            position_m=(1.0, 2.0, 3.0),
            velocity_m_s=(0.0, 0.0, -1.0),
            attitude_quaternion=(0.0, 0.0, 0.0, 1.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
            main_engine_fuel=12.5,
        )
        self.assertEqual(observation.position_m, (1.0, 2.0, 3.0))
        self.assertEqual(observation.main_engine_fuel, 12.5)

    def test_observation_rejects_nonfinite_and_zero_quaternion(self):
        with self.assertRaises(ValueError):
            Observation(
                position_m=(math.nan, 0.0, 0.0),
                velocity_m_s=(0.0, 0.0, 0.0),
                attitude_quaternion=(0.0, 0.0, 0.0, 1.0),
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                main_engine_fuel=0.0,
            )
        with self.assertRaises(ValueError):
            Observation(
                position_m=(0.0, 0.0, 0.0),
                velocity_m_s=(0.0, 0.0, 0.0),
                attitude_quaternion=(0.0, 0.0, 0.0, 0.0),
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
                main_engine_fuel=0.0,
            )

    def test_action_ranges(self):
        self.assertEqual(Action().main_engine_throttle, 0.0)
        with self.assertRaises(ValueError):
            Action(rcs_pitch=1.01)
        with self.assertRaises(ValueError):
            Action(main_engine_throttle=-0.01)
        with self.assertRaises(ValueError):
            Action(rcs_roll=math.nan)

    def test_step_result_keeps_reward_undefined(self):
        observation = Observation(
            position_m=(0.0, 0.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            attitude_quaternion=(0.0, 0.0, 0.0, 1.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
            main_engine_fuel=1.0,
        )
        result = StepResult(observation=observation, info={"episode_evaluated": False})
        self.assertIsNone(result.reward)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertFalse(result.info["episode_evaluated"])


class AdapterTests(unittest.TestCase):
    def test_construction_does_not_connect(self):
        env = KSPEnv.from_config()
        self.assertFalse(env.connected)
        with self.assertRaises(KSPConnectionError):
            env.read_diagnostic_state()
        env.close()
        env.close()

    def test_reset_and_step_refuse_unresolved_configuration(self):
        env = KSPEnv(load_config(), connection=object())
        with self.assertRaises(ConfigurationError):
            env.reset()
        with self.assertRaises(ConfigurationError):
            env.step(Action())
        with self.assertRaises(ConfigurationError):
            env.observe()

    def test_infrastructure_errors_are_separate(self):
        for error in (KSPConnectionError, ResetError, TickError, VesselError):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, InfrastructureError))

    def test_config_dataclass_is_frozen(self):
        config = EnvConfig(
            address="127.0.0.1",
            rpc_port=50000,
            stream_port=50001,
            client_version="0.7.0",
            server_version="0.7.0",
            game_version="1.12.5",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.address = "0.0.0.0"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()


class _EnumValue:
    def __init__(self, name):
        self.name = name


class _FakeControl:
    def __init__(self):
        self.input_mode = 0
        self.sas = True
        self.reaction_wheels = True
        self.rcs = False
        self.forward = 0.0
        self.right = 0.0
        self.up = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.throttle = 0.0


class _FakeResources:
    def has_resource(self, name):
        return name == "LiquidFuel"

    def amount(self, name):
        if name != "LiquidFuel":
            raise KeyError(name)
        return 12.0


class _FakeBody:
    def surface_reference_frame(self, latitude, longitude, altitude):
        return (latitude, longitude, altitude)


class _FakeVessel:
    def __init__(self):
        self.name = "FakeVessel"
        self.control = _FakeControl()
        self.resources = _FakeResources()
        self.position = lambda frame: (0.0, 0.0, 1.0)
        self.velocity = lambda frame: (0.0, 0.0, -1.0)
        self.rotation = lambda frame: (0.0, 0.0, 0.0, 1.0)
        self.angular_velocity = lambda frame: (0.0, 0.0, 0.0)


class _FakeSpaceCenter:
    def __init__(self):
        self._ut = 100.0
        self.active_vessel = _FakeVessel()
        self.bodies = {"Kerbin": _FakeBody()}
        self.can_save = True

    @property
    def ut(self):
        return self._ut


class _FakeKRPC:
    def __init__(self, space_center):
        self.space_center = space_center
        self.game_scene = _EnumValue("flight")
        self.paused = False
        self.game_state_loads = 1
        self.calls = []

    def hold_tick(self):
        self.calls.append("hold")

    def next_tick(self):
        self.calls.append("next")
        self.space_center._ut += 0.02

    def release_tick(self):
        self.calls.append("release")


class _FakeConnection:
    def __init__(self):
        self.space_center = _FakeSpaceCenter()
        self.krpc = _FakeKRPC(self.space_center)
        self.closed = False

    def close(self):
        self.closed = True


@unittest.skipUnless(importlib.util.find_spec("krpc") is not None, "krpc package not installed")
class StepSequencingTests(unittest.TestCase):
    def make_env(self):
        raw = {
            "target_body": "Kerbin",
            "scenario": {"target_position": [0.0, 0.0, 0.0], "save_name": "TBD"},
            "observation": {
                "frame": "target_surface",
                "main_engine_fuel_resource": "LiquidFuel",
            },
            "action": {"rcs_semantics": "normalized_input", "autopilot_inner_loop": False},
        }
        config = EnvConfig(
            address="127.0.0.1",
            rpc_port=50000,
            stream_port=50001,
            client_version="0.7.0",
            server_version="0.7.0",
            game_version="1.12.5",
            raw=raw,
        )
        connection = _FakeConnection()
        return KSPEnv(config, connection=connection), connection

    def test_step_holds_advances_and_releases(self):
        env, connection = self.make_env()
        result = env.step(Action(rcs_pitch=0.1, main_engine_throttle=0.5))
        self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
        self.assertAlmostEqual(result.info["ut_delta_s"], 0.02)
        self.assertFalse(result.info["episode_evaluated"])
        self.assertEqual(connection.space_center.active_vessel.control.throttle, 0.5)
        self.assertTrue(connection.space_center.active_vessel.control.rcs)
        env.close()
        self.assertEqual(connection.space_center.active_vessel.control.throttle, 0.0)
        self.assertFalse(connection.space_center.active_vessel.control.rcs)
        self.assertTrue(connection.closed)

    def test_step_releases_tick_when_observation_fails(self):
        env, connection = self.make_env()

        def fail(frame):
            raise RuntimeError("telemetry failure")

        connection.space_center.active_vessel.position = fail
        with self.assertRaises(TickError):
            env.step(Action())
        self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
        env.close()

    def test_reset_loads_named_save_and_observes(self):
        env, connection = self.make_env()

        def load(name):
            self.assertEqual(name, "test-save")
            connection.krpc.game_state_loads = 2

        connection.space_center.load = load
        observation, info = env.reset(save_name="test-save")
        self.assertEqual(info.save_name, "test-save")
        self.assertEqual(info.scene, "flight")
        self.assertEqual(info.game_state_loads, 2)
        self.assertEqual(observation.main_engine_fuel, 12.0)
        env.close()
