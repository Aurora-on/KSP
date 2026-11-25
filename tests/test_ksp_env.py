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
    EpisodeDoneError,
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
MUN_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ksp_mun_test.toml"


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

    def test_mun_adapter_test_profile_is_complete_but_not_task_ready(self):
        config = load_config(MUN_CONFIG_PATH)
        env = KSPEnv(config)
        env._require_observation_config()
        env._require_action_config()
        self.assertEqual(config.raw["status"], "adapter_test_only")
        self.assertEqual(config.raw["target_body"], "Mun")
        self.assertEqual(config.raw["scenario"]["save_name"], "Mun landing 1")
        self.assertEqual(config.raw["scenario"]["target_position"], [0.0, 139.70616, 0.0])
        self.assertEqual(config.raw["observation"]["main_engine_fuel_resource"], "LiquidFuel")
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


class _FakeFlight:
    def __init__(self, vessel):
        self.vessel = vessel

    @property
    def surface_altitude(self):
        return 0.0 if self.vessel.situation.name in {"landed", "splashed"} else 100.0


class _FakeVessel:
    def __init__(self):
        self.name = "FakeVessel"
        self.control = _FakeControl()
        self.resources = _FakeResources()
        self.situation = _EnumValue("flying")
        self.position = lambda frame: (0.0, 0.0, 1.0)
        self.velocity = lambda frame: (0.0, 0.0, -1.0)
        self.rotation = lambda frame: (0.0, 0.0, 0.0, 1.0)
        self.angular_velocity = lambda frame: (0.0, 0.0, 0.0)

    def flight(self, frame):
        return _FakeFlight(self)


class _FakeReferenceFrame:
    @staticmethod
    def create_relative(parent, rotation=None, **kwargs):
        return ("relative", parent, rotation, kwargs)


class _FakeSpaceCenter:
    def __init__(self):
        self._ut = 100.0
        self.active_vessel = _FakeVessel()
        self.bodies = {"Kerbin": _FakeBody()}
        self.can_save = True
        self.ReferenceFrame = _FakeReferenceFrame

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
        self.on_next = None

    def hold_tick(self):
        self.calls.append("hold")

    def next_tick(self):
        self.calls.append("next")
        self.space_center._ut += 0.02
        callback = self.on_next
        if callback is not None:
            self.on_next = None
            callback()

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
    def make_env(self, *, control_hz=10.0, horizon_s=90.0, stable_window_s=5.0):
        raw = {
            "target_body": "Kerbin",
            "scenario": {
                "target_position": [0.0, 0.0, 0.0],
                "save_name": "TBD",
                "control_hz": control_hz,
                "episode_horizon_s": horizon_s,
            },
            "observation": {
                "frame": "target_surface",
                "main_engine_fuel_resource": "LiquidFuel",
            },
            "action": {"rcs_semantics": "normalized_input", "autopilot_inner_loop": False},
            "safety": {
                "stable_window_s": stable_window_s,
                "max_touchdown_speed_m_s": 2.0,
                "max_touchdown_tilt_deg": 10.0,
                "max_touchdown_angular_velocity_deg_s": 11.459155902616466,
                "target_radius_m": 5.0,
            },
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
        self.assertEqual(
            connection.krpc.calls,
            ["hold", "next", "next", "next", "next", "next", "release"],
        )
        self.assertAlmostEqual(result.info["ut_delta_s"], 0.1)
        self.assertAlmostEqual(result.info["dt_s"], 0.1)
        self.assertEqual(result.info["physics_ticks"], 5)
        self.assertAlmostEqual(result.info["control_interval_s"], 0.1)
        self.assertTrue(result.info["episode_evaluated"])
        self.assertEqual(result.info["status"], "flying")
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
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
        self.assertEqual(
            connection.krpc.calls,
            ["hold", "next", "next", "next", "next", "next", "release"],
        )
        env.close()

    def test_contact_records_preimpact_sample_and_completes_stability_window(self):
        env, connection = self.make_env(stable_window_s=0.2)
        env.step(Action())
        vessel = connection.space_center.active_vessel
        vessel.velocity = lambda frame: (0.0, 0.0, -0.5)
        vessel.situation = _EnumValue("landed")

        result = env.step(Action())
        touchdown = result.info["touchdown"]
        self.assertEqual(result.info["status"], "stabilizing")
        self.assertTrue(result.info["contact"])
        self.assertAlmostEqual(touchdown["speed_m_s"], 1.0)
        self.assertAlmostEqual(touchdown["preimpact_ut_s"], 100.1)
        self.assertAlmostEqual(touchdown["contact_ut_s"], 100.2)
        self.assertAlmostEqual(result.info["stable_time_s"], 0.0)

        result = env.step(Action())
        self.assertEqual(result.info["status"], "stabilizing")
        self.assertAlmostEqual(result.info["stable_time_s"], 0.1)

        result = env.step(Action())
        self.assertEqual(result.info["status"], "safe_landing")
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertTrue(result.info["safe_in_target"])
        with self.assertRaises(EpisodeDoneError):
            env.step(Action())
        env.close()

    def test_unsafe_touchdown_is_crash(self):
        env, connection = self.make_env()
        vessel = connection.space_center.active_vessel
        vessel.velocity = lambda frame: (0.0, 0.0, -5.0)
        env.step(Action())
        vessel.situation = _EnumValue("landed")

        result = env.step(Action())
        self.assertEqual(result.info["status"], "crash")
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertAlmostEqual(result.info["touchdown"]["speed_m_s"], 5.0)
        env.close()

    def test_liftoff_resets_stability_clock(self):
        env, connection = self.make_env(stable_window_s=0.3)
        env.step(Action())
        vessel = connection.space_center.active_vessel
        vessel.situation = _EnumValue("landed")
        env.step(Action())
        vessel.situation = _EnumValue("flying")
        result = env.step(Action())
        self.assertEqual(result.info["status"], "flying")
        self.assertFalse(result.info["contact"])
        self.assertEqual(result.info["stable_time_s"], 0.0)
        env.close()

    def test_timeout_is_truncated(self):
        env, connection = self.make_env(horizon_s=0.1)
        result = env.step(Action())
        self.assertEqual(result.info["status"], "timeout")
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        with self.assertRaises(EpisodeDoneError):
            env.step(Action())
        env.close()

    def test_vessel_loss_is_a_crash_with_stale_observation(self):
        env, connection = self.make_env()
        first = env.step(Action())
        connection.krpc.on_next = lambda: setattr(connection.space_center, "active_vessel", None)
        result = env.step(Action())
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.info["status"], "crash")
        self.assertTrue(result.info["vessel_destroyed"])
        self.assertTrue(result.info["observation_stale"])
        self.assertIs(result.observation, first.observation)
        env.close()

    def test_step_can_advance_one_tick_for_adapter_profile(self):
        env, connection = self.make_env(control_hz=50.0)
        result = env.step(Action())
        self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
        self.assertAlmostEqual(result.info["dt_s"], 0.02)
        self.assertEqual(result.info["physics_ticks"], 1)
        env.close()

    def test_step_requires_control_interval(self):
        env, connection = self.make_env()
        env.config.raw["scenario"]["control_hz"] = "TBD"
        with self.assertRaises(ConfigurationError):
            env.step(Action())
        self.assertEqual(connection.krpc.calls, [])
        env.close()

    def test_tilt_uses_body_z_axis(self):
        self.assertAlmostEqual(KSPEnv._tilt_deg((0.0, 0.0, 0.0, 1.0)), 0.0)
        half = math.sqrt(0.5)
        self.assertAlmostEqual(KSPEnv._tilt_deg((half, 0.0, 0.0, half)), 90.0)
        self.assertAlmostEqual(KSPEnv._tilt_deg((1.0, 0.0, 0.0, 0.0)), 180.0)

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
