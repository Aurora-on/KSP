"""Offline tests for the KSP environment adapter.

No game, kRPC server, NumPy or Gymnasium is required.
"""

import dataclasses
import hashlib
import importlib.util
import math
import sys
import tempfile
import unittest
from ctypes import c_float
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    SnapshotError,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)
from ksp_env.geometry import transform_attitude, transform_body_vector


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
        self.assertEqual(config.raw["scenario"]["target_position"], [0.0, 140.5, 0.0])
        self.assertEqual(
            config.raw["observation"]["main_engine_fuel_resources"],
            ["LiquidFuel", "Oxidizer"],
        )
        self.assertEqual(
            config.raw["observation"]["body_to_task_rotation_xyzw"],
            [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
        )
        for key in ("initial_state_ready", "success_thresholds_ready", "reset_step_validated"):
            self.assertFalse(config.raw[key])

    def test_repository_config_loads(self):
        config = load_config()
        self.assertEqual(config.address, "127.0.0.1")
        self.assertEqual(config.rpc_port, 50000)
        self.assertEqual(config.stream_port, 50001)
        self.assertEqual(config.server_version, "0.7.0")
        self.assertEqual(config.raw["target_body"], "Mun")
        self.assertEqual(config.raw["scenario"]["target_position"], [0.0, 140.5, 0.0])

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

    def test_quaternion_body_mapping(self):
        quarter = math.sqrt(0.5)
        body_to_task = (quarter, 0.0, 0.0, quarter)
        vector = transform_body_vector((0.0, 1.0, 0.0), body_to_task)
        for actual, expected in zip(vector, (0.0, 0.0, 1.0), strict=True):
            self.assertAlmostEqual(actual, expected)
        attitude = transform_attitude(body_to_task, body_to_task)
        for actual, expected in zip(attitude, (0.0, 0.0, 0.0, 1.0), strict=True):
            self.assertAlmostEqual(actual, expected)

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
        self.throttle_writes = [0.0]

    @property
    def throttle(self):
        return self.throttle_writes[-1]

    @throttle.setter
    def throttle(self, value):
        self.throttle_writes.append(value)


class _FakeResources:
    def __init__(self):
        self.all = [
            SimpleNamespace(
                name="LiquidFuel", amount=12.0, max=12.0, density=5.0
            ),
            SimpleNamespace(
                name="Oxidizer", amount=4.0, max=4.0, density=5.0
            ),
        ]

    def has_resource(self, name):
        return name in {"LiquidFuel", "Oxidizer"}

    def amount(self, name):
        if name == "LiquidFuel":
            return 12.0
        if name == "Oxidizer":
            return 4.0
        raise KeyError(name)


class _FakeBody:
    name = "Kerbin"

    def surface_reference_frame(self, latitude, longitude, altitude):
        return (latitude, longitude, altitude)


class _FakeOrbit:
    def __init__(self):
        self.body = _FakeBody()


class _FakeFlight:
    def __init__(self, vessel):
        self.vessel = vessel

    @property
    def surface_altitude(self):
        return 0.0 if self.vessel.situation.name in {"landed", "splashed"} else 1.0


class _FakeVessel:
    def __init__(self):
        self.name = "FakeVessel"
        self.situation = _EnumValue("flying")
        leg = SimpleNamespace(is_grounded=False, state=_EnumValue("deployed"))
        self.parts = SimpleNamespace(all=[1, 2, 3], legs=[leg])
        self.control = _FakeControl()
        self.resources = _FakeResources()
        self.orbit = _FakeOrbit()
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


class _FakeDebug:
    def __init__(self, vessel):
        self.vessel = vessel
        self.calls = []

    def set_position(self, position, reference_frame=None):
        self.calls.append(("position", tuple(position)))
        self.vessel.position = lambda frame: tuple(position)

    def set_velocity(self, velocity, reference_frame=None):
        self.calls.append(("velocity", tuple(velocity)))
        self.vessel.velocity = lambda frame: tuple(velocity)

    def set_direction(self, direction, roll, reference_frame=None, vessel=None):
        self.calls.append(("direction", tuple(direction), roll))

    def set_angular_velocity(self, angular_velocity, reference_frame=None, vessel=None):
        self.calls.append(("angular_velocity", tuple(angular_velocity)))
        self.vessel.angular_velocity = lambda frame: tuple(angular_velocity)

    def fill_all_resources(self, vessel=None):
        self.calls.append(("fill_all_resources",))


class _FakeSpaceCenter:
    def __init__(self):
        self._ut = 100.0
        self.active_vessel = _FakeVessel()
        self.bodies = {"Kerbin": _FakeBody(), "Mun": _FakeBody()}
        self.can_save = True
        self.ReferenceFrame = _FakeReferenceFrame
        self.save_dir = None
        self.saved = []
        self.game_state_loads = 1

    @property
    def ut(self):
        return self._ut

    def save(self, name):
        self.saved.append(name)
        if self.save_dir is not None:
            (Path(self.save_dir) / f"{name}.sfs").write_text("fake save", encoding="utf-8")

    def load(self, name):
        self.game_state_loads += 1


class _FakeKRPC:
    def __init__(self, space_center):
        self.space_center = space_center
        self.game_scene = _EnumValue("flight")
        self.paused = False
        self.game_state_loads = 1
        self.calls = []
        self.tick_durations = None
        self.after_tick = None
        self.completed_ticks = 0
        self.on_next = None

    def hold_tick(self):
        self.calls.append("hold")

    def next_tick(self):
        self.calls.append("next")
        duration = 0.02 if self.tick_durations is None else next(self.tick_durations)
        self.space_center._ut += duration
        self.completed_ticks += 1
        if self.after_tick is not None:
            self.after_tick(self.completed_ticks)
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
        self.debug = _FakeDebug(self.space_center.active_vessel)
        self.closed = False

    def close(self):
        self.closed = True


class StepSequencingTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("krpc") is None:
            # The only client type used by these in-process controls is this enum.
            module = SimpleNamespace(ControlInputMode=SimpleNamespace(override=1))
            self.enterContext(patch.dict(sys.modules, {"krpc.services.spacecenter": module}))

    def make_env(self, *, nominal=False, control_hz=10.0, horizon_s=90.0, stable_window_s=5.0):
        if nominal:
            config = load_config()
            config.raw["scenario"].update(
                target_position=[0.0, 0.0, 0.0],
                control_hz=control_hz,
                episode_horizon_s=horizon_s,
            )
            config.raw["observation"].update(
                frame="target_surface",
                main_engine_fuel_resources=["LiquidFuel"],
                body_to_task_rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
            config.raw["action"].update(
                rcs_semantics="normalized_input",
                autopilot_inner_loop=False,
            )
            config.raw["safety"]["stable_window_s"] = stable_window_s
        else:
            config = load_config(CONFIG_PATH)
            config.raw["task"] = "powered_descent"
            config.raw["scenario"]["control_hz"] = control_hz
            config.raw["scenario"]["episode_horizon_s"] = horizon_s
            config.raw["observation"]["body_to_task_rotation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
            config.raw["safety"]["stable_window_s"] = stable_window_s
        connection = _FakeConnection()
        env = KSPEnv(config, connection=connection)
        self.addCleanup(env.close)
        return env, connection

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

    def test_nominal_hold_samples_contact_and_part_loss_before_final_observation(self):
        env, connection = self.make_env(nominal=True)
        vessel = connection.space_center.active_vessel
        rpc = connection.krpc

        def simulate_contact(tick):
            if tick == 1:
                vessel.parts.legs[0].is_grounded = True
            elif tick == 2:
                vessel.parts.legs[0].state = _EnumValue("broken")
                vessel.situation = _EnumValue("landed")
            elif tick == 3:
                vessel.parts.legs[0].is_grounded = False
                vessel.parts.all.pop()
                vessel.situation = _EnumValue("flying")

        rpc.after_tick = simulate_contact
        action = Action(rcs_pitch=0.15, main_engine_throttle=0.3)
        result = env.step(action)
        self.assertEqual(vessel.control.throttle_writes.count(0.3), 1)
        self.assertEqual(rpc.calls, ["hold"] + ["next"] * 5 + ["release"])
        self.assertEqual(result.info["physics_ticks"], 5)
        self.assertAlmostEqual(result.info["control_interval_s"], 0.1)
        self.assertAlmostEqual(result.info["ut_delta_s"], 0.1)
        samples = result.info["tick_samples"]
        self.assertEqual(
            [s["situation"] for s in samples],
            ["flying", "flying", "landed", "flying", "flying", "flying"],
        )
        self.assertEqual([s["leg_count"] for s in samples], [1] * 6)
        self.assertEqual([s["grounded_legs"] for s in samples], [0, 1, 1, 0, 0, 0])
        self.assertEqual([s["broken_legs"] for s in samples], [0, 0, 1, 1, 1, 1])
        self.assertEqual([s["part_count"] for s in samples], [3, 3, 3, 2, 2, 2])
        for i, sample in enumerate(samples):
            self.assertAlmostEqual(sample["ut_s"], result.info["ut_start_s"] + i * 0.02)
            expected_altitude = 0.0 if sample["situation"] in {"landed", "splashed"} else 1.0
            self.assertEqual(sample["position_m"], (0.0, 0.0, 1.0))
            self.assertEqual(sample["surface_altitude_m"], expected_altitude)
            self.assertEqual(sample["velocity_m_s"], (0.0, 0.0, -1.0))
            self.assertEqual(sample["attitude_quaternion"], (0.0, 0.0, 0.0, 1.0))
            self.assertEqual(sample["angular_velocity_rad_s"], (0.0, 0.0, 0.0))
            self.assertAlmostEqual(sample["speed_m_s"], 1.0)
            self.assertAlmostEqual(sample["angular_speed_rad_s"], 0.0)
        self.assertEqual(result.info["ut_end_s"], samples[-1]["ut_s"])
        self.assertIsNone(result.reward)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertTrue(result.info["episode_evaluated"])
        self.assertEqual(result.info["status"], "flying")

    def test_variable_tick_lengths_use_actual_elapsed_ut(self):
        env, connection = self.make_env(nominal=True)
        durations = (0.018, 0.024, 0.040, 0.026)
        connection.krpc.tick_durations = iter(durations)
        info = env.step(Action()).info
        self.assertEqual(info["physics_ticks"], len(durations))
        self.assertAlmostEqual(info["ut_delta_s"], sum(durations))
        self.assertEqual(info["ut_delta_s"], info["ut_end_s"] - info["ut_start_s"])

    def test_task_control_rate_comes_from_configuration(self):
        env, connection = self.make_env(nominal=True)
        env.config.raw["scenario"]["control_hz"] = 25
        self.assertEqual(env.step(Action()).info["physics_ticks"], 2)
        self.assertEqual(connection.krpc.calls, ["hold", "next", "next", "release"])

    def test_float_physics_tick_roundoff_does_not_add_a_sixth_tick(self):
        env, connection = self.make_env(nominal=True)
        connection.space_center._ut = 367.78
        connection.krpc.tick_durations = iter([c_float(0.02).value] * 5)
        self.assertEqual(env.step(Action()).info["physics_ticks"], 5)

    def test_explicit_step_tick_on_nominal_task_advances_only_one_tick(self):
        env, connection = self.make_env(nominal=True)
        info = env.step_tick(Action()).info
        self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
        self.assertAlmostEqual(info["ut_delta_s"], 0.02)
        self.assertEqual(info["physics_ticks"], 1)

    def test_non_advancing_reversed_and_nonfinite_ticks_fail_and_release(self):
        for bad_duration in (0.0, -0.02, math.nan):
            with self.subTest(bad_duration=bad_duration):
                env, connection = self.make_env(nominal=True)
                connection.krpc.tick_durations = iter([0.02, bad_duration])
                with self.assertRaises(TickError):
                    env.step(Action(main_engine_throttle=0.5))
                self.assertEqual(connection.krpc.calls, ["hold", "next", "next", "release"])
                self.assertEqual(connection.space_center.active_vessel.control.throttle, 0.0)

    def test_hold_expiring_during_observation_fails_without_returning_mixed_state(self):
        env, connection = self.make_env(nominal=True)
        vessel = connection.space_center.active_vessel

        calls = 0

        def read_after_lost_hold(frame):
            nonlocal calls
            calls += 1
            if calls >= 7:
                connection.space_center._ut += 0.02
            return (0.0, 0.0, 1.0)

        vessel.position = read_after_lost_hold
        with self.assertRaises(TickError):
            env.step(Action(main_engine_throttle=0.5))
        self.assertEqual(connection.krpc.calls, ["hold"] + ["next"] * 5 + ["release"])
        self.assertEqual(vessel.control.throttle, 0.0)

    def test_slow_transport_times_out_despite_advancing_ut(self):
        env, connection = self.make_env(nominal=True)
        with patch("ksp_env.env.time.monotonic", return_value=0.0) as clock:
            def exceed_timeout(tick):
                clock.return_value = env.config.timeout_s + 1.0

            connection.krpc.after_tick = exceed_timeout
            with self.assertRaises(TickError):
                env.step(Action(main_engine_throttle=0.5))
        self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
        self.assertEqual(connection.space_center.active_vessel.control.throttle, 0.0)

    def test_vessel_loss_or_switch_aborts_without_commanding_replacement(self):
        for replacement in (None, _FakeVessel()):
            with self.subTest(replacement=replacement):
                env, connection = self.make_env(nominal=True)
                old = connection.space_center.active_vessel
                if replacement is not None:
                    replacement.control.throttle = 0.23

                def change_vessel(tick):
                    connection.space_center.active_vessel = replacement

                connection.krpc.after_tick = change_vessel
                with self.assertRaises(VesselError) as failed:
                    env.step(Action(main_engine_throttle=0.5))
                samples = failed.exception.tick_samples
                self.assertEqual([sample["ut_s"] for sample in samples], [100.0])
                self.assertEqual(samples[0]["situation"], "flying")
                self.assertEqual(connection.krpc.calls, ["hold", "next", "release"])
                self.assertEqual(old.control.throttle, 0.0)
                if replacement is not None:
                    self.assertEqual(replacement.control.throttle, 0.23)

    def test_control_session_keeps_shared_boundary_between_policy_actions(self):
        env, connection = self.make_env(nominal=True)
        throttle_by_tick = []
        control = connection.space_center.active_vessel.control
        connection.krpc.after_tick = lambda tick: throttle_by_tick.append(control.throttle)
        with env.control_session():
            # An accidental nested session must leave the original hold intact.
            with self.assertRaises(TickError):
                with env.control_session():
                    self.fail("nested control session reopened a held physics tick")
            first = env.step(Action(main_engine_throttle=0.25))
            second = env.step(Action(main_engine_throttle=0.75))
            self.assertEqual(connection.krpc.calls, ["hold"] + ["next"] * 10)
        self.assertEqual(connection.krpc.calls[-1], "release")
        self.assertEqual(throttle_by_tick, [0.25] * 5 + [0.75] * 5)
        self.assertEqual(first.info["ut_end_s"], second.info["ut_start_s"])
        self.assertAlmostEqual(second.info["ut_end_s"] - first.info["ut_start_s"], 0.2)

    def test_control_session_releases_after_client_exception_and_can_be_reused(self):
        env, connection = self.make_env(nominal=True)
        with self.assertRaisesRegex(ValueError, "invalid policy output"):
            with env.control_session():
                env.step(Action(main_engine_throttle=0.7))
                raise ValueError("invalid policy output")
        session_calls = ["hold"] + ["next"] * 5 + ["release"]
        self.assertEqual(connection.krpc.calls, session_calls)
        self.assertEqual(connection.space_center.active_vessel.control.throttle, 0.0)
        with env.control_session():
            self.assertEqual(env.step(Action()).info["physics_ticks"], 5)
        self.assertEqual(connection.krpc.calls, session_calls * 2)

    def test_control_session_detects_unrequested_ut_advance_before_applying_action(self):
        env, connection = self.make_env(nominal=True)
        with env.control_session():
            env.step(Action())
            connection.space_center._ut += 0.02
            with self.assertRaises(TickError):
                env.step(Action(main_engine_throttle=0.8))
            self.assertNotIn(0.8, connection.space_center.active_vessel.control.throttle_writes)
            self.assertEqual(connection.krpc.calls[-1], "release")
            with self.assertRaises(TickError):
                env.step(Action())
        self.assertEqual(connection.krpc.calls, ["hold"] + ["next"] * 5 + ["release"])

    def test_step_releases_tick_when_observation_fails(self):
        env, connection = self.make_env()

        calls = 0

        def fail(frame):
            nonlocal calls
            calls += 1
            if calls >= 7:
                raise RuntimeError("telemetry failure")
            return (0.0, 0.0, 1.0)

        connection.space_center.active_vessel.position = fail
        with self.assertRaises(TickError):
            env.step(Action())
        self.assertEqual(
            connection.krpc.calls,
            ["hold", "next", "next", "next", "next", "next", "release"],
        )
        env.close()

    def test_contact_records_preimpact_sample_and_completes_stability_window(self):
        env, connection = self.make_env(stable_window_s=0.3)
        env.step(Action())
        vessel = connection.space_center.active_vessel
        vessel.velocity = lambda frame: (0.0, 0.0, -0.5)
        vessel.situation = _EnumValue("landed")

        result = env.step(Action())
        touchdown = result.info["touchdown"]
        self.assertEqual(result.info["status"], "stabilizing")
        self.assertTrue(result.info["contact"])
        self.assertAlmostEqual(touchdown["speed_m_s"], 1.0)
        self.assertAlmostEqual(touchdown["preimpact_ut_s"], 100.1, places=6)
        self.assertAlmostEqual(touchdown["contact_ut_s"], 100.1, places=6)
        self.assertAlmostEqual(result.info["stable_time_s"], 0.1)

        result = env.step(Action())
        self.assertEqual(result.info["status"], "stabilizing")
        self.assertAlmostEqual(result.info["stable_time_s"], 0.2)

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

    def test_create_initial_save_bypasses_can_save(self):
        env, connection = self.make_env()
        connection.space_center.can_save = False
        with tempfile.TemporaryDirectory() as directory:
            connection.space_center.save_dir = Path(directory)
            info = env.create_initial_save(
                "snapshot",
                position_m=(0.0, 0.0, 100.0),
                velocity_m_s=(0.0, 0.0, -5.0),
                save_dir=directory,
            )
            self.assertFalse(info.can_save_before)
            self.assertFalse(info.can_save_after)
            self.assertTrue(info.verified)
            self.assertEqual(info.save_name, "snapshot")
            self.assertEqual(connection.space_center.saved, ["snapshot"])
            self.assertIn(("fill_all_resources",), connection.debug.calls)
            self.assertIn(("position", (0.0, 0.0, 100.0)), connection.debug.calls)
            self.assertTrue(Path(info.save_path).exists())
            self.assertEqual(
                info.save_sha256,
                hashlib.sha256(Path(info.save_path).read_bytes()).hexdigest(),
            )
            self.assertEqual(info.requested["body_z_direction"], [0.0, 0.0, 1.0])
            self.assertIsNotNone(info.observed)
        env.close()

    def test_create_initial_save_refuses_existing_path(self):
        env, connection = self.make_env()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.sfs"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(SnapshotError):
                env.create_initial_save(
                    "snapshot",
                    position_m=(0.0, 0.0, 100.0),
                    velocity_m_s=(0.0, 0.0, -5.0),
                    save_dir=directory,
                )
            self.assertEqual(connection.debug.calls, [])
        env.close()

    def test_observation_sums_configured_propellant_mass(self):
        env, connection = self.make_env()
        env.config.raw["observation"]["main_engine_fuel_resources"] = [
            "LiquidFuel",
            "Oxidizer",
        ]
        observation = env.observe()
        self.assertEqual(observation.main_engine_fuel, 80.0)
        env.close()

    def test_reset_loads_named_save_and_observes(self):
        env, connection = self.make_env()
        connection.space_center.can_save = False

        def load(name):
            self.assertEqual(name, "test-save")
            connection.krpc.game_state_loads = 2

        connection.space_center.load = load
        observation, info = env.reset(save_name="test-save")
        self.assertEqual(info.save_name, "test-save")
        self.assertEqual(info.scene, "flight")
        self.assertEqual(info.game_state_loads, 2)
        self.assertEqual(observation.main_engine_fuel, 60.0)
        env.close()

    def test_reset_loads_even_when_can_save_is_false(self):
        env, connection = self.make_env()
        connection.space_center.can_save = False

        def load(name):
            self.assertEqual(name, "low-save")
            connection.krpc.game_state_loads = 2

        connection.space_center.load = load
        observation, info = env.reset(save_name="low-save")
        self.assertEqual(info.save_name, "low-save")
        self.assertEqual(info.game_state_loads, 2)
        self.assertEqual(observation.main_engine_fuel, 60.0)
        self.assertFalse(connection.space_center.active_vessel.control.rcs)
        env.close()


if __name__ == "__main__":
    unittest.main()
