"""KSP environment adapter.

The adapter deliberately reuses kRPC's generated SpaceCenter proxies instead
of reimplementing frames, telemetry or control.  This module owns only the
task-specific contract: observation assembly, action validation, reset/step
sequencing and infrastructure error classification.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Mapping

from .contracts import (
    Action,
    ConfigurationError,
    EnvConfig,
    EpisodeDoneError,
    InfrastructureError,
    KSPConnectionError,
    KSPEnvError,
    Observation,
    ResetError,
    ResetInfo,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime
    import krpc


CONTACT_SITUATIONS = frozenset({"landed", "splashed"})


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


@dataclass
class _EpisodeState:
    """Mutable per-episode contact and timing state."""

    started_ut_s: float
    elapsed_s: float = 0.0
    contact: bool = False
    stable_time_s: float = 0.0
    status: str = "flying"
    touchdown: dict[str, Any] | None = None
    last_touchdown: dict[str, Any] | None = None
    touchdown_count: int = 0
    previous_sample: dict[str, Any] | None = None
    last_observation: Observation | None = None
    done: bool = False


class KSPEnv:
    """A small, task-agnostic KSP environment adapter.

    Construction does not open a connection.  Call :meth:`connect` or use the
    object as a context manager.  All kRPC imports are lazy so offline tests and
    CI do not need the game or the client package.
    """

    def __init__(self, config: EnvConfig, *, connection: Any | None = None) -> None:
        self.config = config
        self._conn: Any | None = connection
        self._target_frame_cache: Any | None = None
        self._controls_dirty = False
        self._episode: _EpisodeState | None = None

    @classmethod
    def from_config(cls, path: str | None = None) -> "KSPEnv":
        return cls(load_config(path))

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def connection(self) -> Any | None:
        """The underlying kRPC client, for diagnostics only."""

        return self._conn

    def connect(self) -> "KSPEnv":
        """Open the kRPC connection and validate the server version/services."""

        if self._conn is not None:
            return self

        try:
            import krpc
        except ImportError as exc:  # pragma: no cover - depends on local setup
            raise KSPConnectionError("the krpc Python package is not installed") from exc

        conn = None
        try:
            conn = krpc.connect(
                name="ksp_env",
                address=self.config.address,
                rpc_port=self.config.rpc_port,
                stream_port=self.config.stream_port,
                timeout=self.config.timeout_s,
            )
            status = conn.krpc.get_status()
            if status.version != self.config.server_version:
                raise KSPConnectionError(
                    f"kRPC server {status.version} does not match required {self.config.server_version}"
                )
            services = {service.name for service in conn.krpc.get_services().services}
            missing = {"KRPC", "SpaceCenter"} - services
            if missing:
                raise KSPConnectionError(f"kRPC server is missing services: {sorted(missing)}")
        except KSPEnvError:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise KSPConnectionError(str(exc)) from exc

        self._conn = conn
        return self

    def close(self) -> None:
        """Release controls, the held tick and the connection.  Idempotent."""

        conn = self._conn
        if conn is None:
            return
        try:
            conn.krpc.release_tick()
        except Exception:
            pass
        try:
            if self._controls_dirty:
                self._neutralize_controls(conn)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        finally:
            self._conn = None
            self._target_frame_cache = None
            self._episode = None

    def __enter__(self) -> "KSPEnv":
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_connection(self) -> Any:
        if self._conn is None:
            raise KSPConnectionError("not connected; call connect() first")
        return self._conn

    def _scene(self) -> str:
        return _enum_name(self._require_connection().krpc.game_scene)

    def _active_vessel(self) -> Any:
        vessel = self._require_connection().space_center.active_vessel
        if vessel is None:
            raise VesselError("there is no active vessel")
        return vessel

    def _require_flight(self) -> Any:
        if self._scene() != "flight":
            raise VesselError(f"Flight scene required, current scene is {self._scene()}")
        return self._active_vessel()

    # ------------------------------------------------------------------
    # Configuration gates
    # ------------------------------------------------------------------

    def _observation_config(self) -> dict[str, Any]:
        value = self.config.raw.get("observation")
        if not isinstance(value, dict):
            raise ConfigurationError("configuration is missing the [observation] section")
        return value

    def _action_config(self) -> dict[str, Any]:
        value = self.config.raw.get("action")
        if not isinstance(value, dict):
            raise ConfigurationError("configuration is missing the [action] section")
        return value

    def _scenario_config(self) -> dict[str, Any]:
        value = self.config.raw.get("scenario")
        if not isinstance(value, dict):
            raise ConfigurationError("configuration is missing the [scenario] section")
        return value

    def _safety_config(self) -> dict[str, Any]:
        value = self.config.raw.get("safety")
        if not isinstance(value, dict):
            raise ConfigurationError("configuration is missing the [safety] section")
        return value

    def _control_interval_s(self) -> float:
        control_hz = require_number(
            self._scenario_config().get("control_hz"), "[scenario].control_hz"
        )
        if control_hz <= 0:
            raise ConfigurationError("[scenario].control_hz must be positive")
        return 1.0 / control_hz

    def _episode_horizon_s(self) -> float:
        horizon_s = require_number(
            self._scenario_config().get("episode_horizon_s"),
            "[scenario].episode_horizon_s",
        )
        if horizon_s <= 0:
            raise ConfigurationError("[scenario].episode_horizon_s must be positive")
        return horizon_s

    def _require_safety_config(self) -> dict[str, float]:
        safety = self._safety_config()
        values = {
            "stable_window_s": require_number(
                safety.get("stable_window_s"), "[safety].stable_window_s"
            ),
            "max_touchdown_speed_m_s": require_number(
                safety.get("max_touchdown_speed_m_s"), "[safety].max_touchdown_speed_m_s"
            ),
            "max_touchdown_tilt_deg": require_number(
                safety.get("max_touchdown_tilt_deg"), "[safety].max_touchdown_tilt_deg"
            ),
            "max_touchdown_angular_velocity_deg_s": require_number(
                safety.get("max_touchdown_angular_velocity_deg_s"),
                "[safety].max_touchdown_angular_velocity_deg_s",
            ),
            "target_radius_m": require_number(
                safety.get("target_radius_m"), "[safety].target_radius_m"
            ),
        }
        for name, value in values.items():
            if value <= 0:
                raise ConfigurationError(f"[safety].{name} must be positive")
        return values

    def _require_episode_config(self) -> None:
        self._control_interval_s()
        self._episode_horizon_s()
        self._require_safety_config()

    def _require_observation_config(self) -> None:
        observation = self._observation_config()
        frame = observation.get("frame")
        if is_tbd(frame):
            raise ConfigurationError("[observation].frame is TBD")
        if frame != "target_surface":
            raise ConfigurationError(
                f"unsupported [observation].frame {frame!r}; only 'target_surface' is implemented"
            )
        resource = observation.get("main_engine_fuel_resource")
        if is_tbd(resource) or not isinstance(resource, str) or not resource.strip():
            raise ConfigurationError("[observation].main_engine_fuel_resource is TBD")
        self._target_coordinates()

    def _require_action_config(self) -> None:
        action = self._action_config()
        semantics = action.get("rcs_semantics")
        if is_tbd(semantics):
            raise ConfigurationError("[action].rcs_semantics is TBD")
        if semantics != "normalized_input":
            raise ConfigurationError(
                f"unsupported [action].rcs_semantics {semantics!r}; only 'normalized_input' is implemented"
            )
        if bool(action.get("autopilot_inner_loop", False)):
            raise ConfigurationError("an inner-loop autopilot is not implemented")

    def _target_coordinates(self) -> tuple[float, float, float]:
        scenario = self._scenario_config()
        target = scenario.get("target_position")
        if is_tbd(target):
            raise ConfigurationError("[scenario].target_position is TBD")

        if isinstance(target, dict):
            latitude = target.get("latitude_deg")
            longitude = target.get("longitude_deg")
            altitude = target.get("altitude_m", 0.0)
        elif isinstance(target, (list, tuple)) and len(target) in (2, 3):
            latitude, longitude = target[0], target[1]
            altitude = target[2] if len(target) == 3 else 0.0
        else:
            raise ConfigurationError(
                "[scenario].target_position must be [lat_deg, lon_deg, altitude_m] "
                "or an object with latitude_deg/longitude_deg"
            )

        return (
            require_number(latitude, "[scenario].target_position.latitude_deg"),
            require_number(longitude, "[scenario].target_position.longitude_deg"),
            require_number(altitude, "[scenario].target_position.altitude_m"),
        )

    def _target_reference_frame(self) -> Any:
        if self._target_frame_cache is not None:
            return self._target_frame_cache
        conn = self._require_connection()
        body_name = require_defined(self.config.raw.get("target_body"), "target_body")
        body = conn.space_center.bodies.get(body_name)
        if body is None:
            raise ConfigurationError(f"target body {body_name!r} is not present in this game")
        latitude, longitude, altitude = self._target_coordinates()
        surface = body.surface_reference_frame(latitude, longitude, altitude)
        # kRPC's surface frame is x=up, y=north, z=east.  Rotate it to a
        # right-handed target-local frame with x=east, y=south, z=up so that
        # the adapter uses the same +z-up convention as the simulator.
        rotation = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
        self._target_frame_cache = conn.space_center.ReferenceFrame.create_relative(
            surface, rotation=rotation
        )
        return self._target_frame_cache

    # ------------------------------------------------------------------
    # Read-only diagnostics and task observation
    # ------------------------------------------------------------------

    def read_diagnostic_state(self) -> dict[str, Any]:
        """Read raw flight state without claiming it is the task observation.

        The task observation needs a target-relative frame, a fixed lander and
        a decision about the main-engine fuel resource.  This method is useful
        for live connection checks in the meantime.
        """

        conn = self._require_connection()
        vessel = self._active_vessel()
        body = vessel.orbit.body
        frame = body.reference_frame
        flight = vessel.flight(frame)
        control = vessel.control

        resources = {}
        try:
            resources = {
                name: float(vessel.resources.amount(name))
                for name in sorted(vessel.resources.names)
            }
        except Exception:
            pass

        engines = []
        try:
            for engine in vessel.parts.engines:
                part = engine.part
                engines.append(
                    {
                        "part_name": getattr(part, "name", None),
                        "active": bool(engine.active),
                        "has_fuel": bool(engine.has_fuel),
                        "throttle": float(engine.throttle),
                        "thrust_n": float(engine.thrust),
                        "propellants": [prop.name for prop in engine.propellants],
                    }
                )
        except Exception:
            pass

        return {
            "scene": _enum_name(conn.krpc.game_scene),
            "paused": bool(conn.krpc.paused),
            "game_state_loads": int(conn.krpc.game_state_loads),
            "ut_s": float(conn.space_center.ut),
            "vessel_name": vessel.name,
            "body_name": body.name,
            "situation": _enum_name(vessel.situation),
            "loaded": bool(vessel.loaded),
            "mass_kg": float(vessel.mass),
            "moment_of_inertia_kg_m2": [float(value) for value in vessel.moment_of_inertia],
            "inertia_tensor_kg_m2": [float(value) for value in vessel.inertia_tensor],
            "position_body_m": [float(value) for value in vessel.position(frame)],
            "velocity_body_m_s": [float(value) for value in vessel.velocity(frame)],
            "attitude_body_quaternion_xyzw": [float(value) for value in vessel.rotation(frame)],
            "angular_velocity_body_rad_s": [
                float(value) for value in vessel.angular_velocity(frame)
            ],
            "surface_altitude_m": float(flight.surface_altitude),
            "surface_speed_m_s": float(flight.speed),
            "vertical_speed_m_s": float(flight.vertical_speed),
            "latitude_deg": float(flight.latitude),
            "longitude_deg": float(flight.longitude),
            "control_state": _enum_name(control.state),
            "control_source": _enum_name(control.source),
            "control_input_mode": _enum_name(control.input_mode),
            "throttle": float(control.throttle),
            "rcs_enabled": bool(control.rcs),
            "sas_enabled": bool(control.sas),
            "resources": resources,
            "engines": engines,
        }

    def observe(self) -> Observation:
        """Build the task observation from kRPC telemetry.

        This currently requires ``[observation].frame = 'target_surface'``,
        a fixed target position and a configured main-engine fuel resource.
        All three are still ``TBD`` in the checked-in task contract, so the
        method raises ``ConfigurationError`` until those decisions are made.
        """

        self._require_connection()
        self._require_observation_config()
        vessel = self._require_flight()
        frame = self._target_reference_frame()
        fuel_resource = self._observation_config()["main_engine_fuel_resource"]

        resources = vessel.resources
        if not resources.has_resource(fuel_resource):
            raise VesselError(f"main-engine resource {fuel_resource!r} is not present on the vessel")

        return Observation(
            position_m=tuple(float(value) for value in vessel.position(frame)),
            velocity_m_s=tuple(float(value) for value in vessel.velocity(frame)),
            attitude_quaternion=tuple(float(value) for value in vessel.rotation(frame)),
            angular_velocity_rad_s=tuple(
                float(value) for value in vessel.angular_velocity(frame)
            ),
            main_engine_fuel=float(resources.amount(fuel_resource)),
        )

    # ------------------------------------------------------------------
    # Actions and stepping
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> None:
        """Send one validated action through kRPC's ``Control`` interface."""

        self._require_action_config()
        vessel = self._require_flight()
        control = vessel.control

        # Imported lazily so offline tests do not need the kRPC stubs.
        from krpc.services.spacecenter import ControlInputMode

        control.input_mode = ControlInputMode.override
        control.sas = False
        control.reaction_wheels = False
        control.rcs = True
        control.forward = 0.0
        control.right = 0.0
        control.up = 0.0
        control.pitch = action.rcs_pitch
        control.yaw = action.rcs_yaw
        control.roll = action.rcs_roll
        control.throttle = action.main_engine_throttle
        self._controls_dirty = True

    def _contact_sample(self, observation: Observation, ut_s: float) -> dict[str, Any]:
        """Read the contact-relevant telemetry for one observation."""

        vessel = self._require_flight()
        frame = self._target_reference_frame()
        flight = vessel.flight(frame)
        position = observation.position_m
        velocity = observation.velocity_m_s
        return {
            "ut_s": float(ut_s),
            "situation": _enum_name(vessel.situation),
            "surface_altitude_m": float(flight.surface_altitude),
            "speed_m_s": math.sqrt(sum(value * value for value in velocity)),
            "vertical_speed_m_s": float(velocity[2]),
            "horizontal_speed_m_s": math.hypot(velocity[0], velocity[1]),
            "tilt_deg": self._tilt_deg(observation.attitude_quaternion),
            "angular_speed_rad_s": math.sqrt(
                sum(value * value for value in observation.angular_velocity_rad_s)
            ),
            "horizontal_error_m": math.hypot(position[0], position[1]),
            "main_engine_fuel": observation.main_engine_fuel,
        }

    @staticmethod
    def _tilt_deg(quaternion: tuple[float, float, float, float]) -> float:
        """Return the angle between body +z and target-frame +z in degrees."""

        x, y, z, w = quaternion
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            raise ValueError("attitude quaternion must not be zero")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        body_z_in_frame = 1.0 - 2.0 * (x * x + y * y)
        return math.degrees(math.acos(max(-1.0, min(1.0, body_z_in_frame))))

    @staticmethod
    def _safe_motion(sample: Mapping[str, Any], safety: Mapping[str, float]) -> bool:
        return (
            sample["speed_m_s"] <= safety["max_touchdown_speed_m_s"] + 1e-9
            and sample["tilt_deg"] <= safety["max_touchdown_tilt_deg"] + 1e-9
            and sample["angular_speed_rad_s"]
            <= math.radians(safety["max_touchdown_angular_velocity_deg_s"]) + 1e-9
        )

    @staticmethod
    def _touchdown_event(
        previous: Mapping[str, Any] | None, current: Mapping[str, Any]
    ) -> dict[str, Any]:
        impact = (
            previous
            if previous is not None
            and previous["situation"] not in CONTACT_SITUATIONS
            else current
        )
        return {
            "ut_s": current["ut_s"],
            "preimpact_ut_s": impact["ut_s"],
            "contact_ut_s": current["ut_s"],
            "speed_m_s": impact["speed_m_s"],
            "vertical_speed_m_s": impact["vertical_speed_m_s"],
            "horizontal_speed_m_s": impact["horizontal_speed_m_s"],
            "tilt_deg": impact["tilt_deg"],
            "angular_speed_rad_s": impact["angular_speed_rad_s"],
            "horizontal_error_m": impact["horizontal_error_m"],
            "surface_altitude_m": current["surface_altitude_m"],
            "situation": current["situation"],
            "contact_speed_m_s": current["speed_m_s"],
            "contact_tilt_deg": current["tilt_deg"],
            "contact_angular_speed_rad_s": current["angular_speed_rad_s"],
        }

    def _initialize_episode(
        self,
        ut_s: float,
        sample: Mapping[str, Any],
        observation: Observation,
        safety: Mapping[str, float],
    ) -> _EpisodeState:
        contact = sample["situation"] in CONTACT_SITUATIONS
        status = "flying"
        done = False
        if contact:
            if self._safe_motion(sample, safety):
                status = "stabilizing"
            else:
                status = "crash"
                done = True
        return _EpisodeState(
            started_ut_s=float(ut_s),
            contact=contact,
            stable_time_s=0.0,
            status=status,
            previous_sample=dict(sample),
            last_observation=observation,
            done=done,
        )

    def _update_episode(
        self,
        episode: _EpisodeState,
        sample: Mapping[str, Any],
        observation: Observation,
        dt_s: float,
        safety: Mapping[str, float],
        horizon_s: float,
    ) -> None:
        episode.elapsed_s += dt_s
        contact_now = sample["situation"] in CONTACT_SITUATIONS

        if not episode.done:
            if episode.contact and not contact_now:
                episode.contact = False
                episode.stable_time_s = 0.0
                episode.status = "flying"
            elif not episode.contact and contact_now:
                event = self._touchdown_event(episode.previous_sample, sample)
                episode.touchdown_count += 1
                episode.last_touchdown = event
                if episode.touchdown is None:
                    episode.touchdown = event
                episode.contact = True
                episode.stable_time_s = 0.0
                if self._safe_motion(event, safety):
                    episode.status = "stabilizing"
                else:
                    episode.status = "crash"
                    episode.done = True
            elif episode.contact and contact_now and episode.status == "stabilizing":
                if not self._safe_motion(sample, safety):
                    episode.status = "crash"
                    episode.done = True
                else:
                    episode.stable_time_s += dt_s
                    if episode.stable_time_s + 1e-9 >= safety["stable_window_s"]:
                        episode.status = "safe_landing"
                        episode.done = True

            if not episode.done and not episode.contact:
                if observation.main_engine_fuel <= 0.0:
                    episode.status = "fuel_exhausted"
                    episode.done = True
                else:
                    episode.status = "flying"

            if not episode.done and episode.elapsed_s + 1e-9 >= horizon_s:
                episode.status = "timeout"
                episode.done = True

        episode.previous_sample = dict(sample)
        episode.last_observation = observation

    def _episode_info(
        self,
        episode: _EpisodeState,
        sample: Mapping[str, Any] | None,
        ut_start: float | None,
        ut_end: float | None,
        physics_ticks: int,
        control_interval_s: float,
        control_hz: float,
        safety: Mapping[str, float],
        *,
        vessel_destroyed: bool = False,
        observation_stale: bool = False,
    ) -> dict[str, Any]:
        sample = sample if sample is not None else episode.previous_sample
        info: dict[str, Any] = {
            "control_hz": control_hz,
            "control_interval_s": control_interval_s,
            "ut_start_s": ut_start,
            "ut_end_s": ut_end,
            "ut_delta_s": None if ut_start is None or ut_end is None else ut_end - ut_start,
            "dt_s": None if ut_start is None or ut_end is None else ut_end - ut_start,
            "physics_ticks": physics_ticks,
            "episode_evaluated": True,
            "episode_started_ut_s": episode.started_ut_s,
            "episode_time_s": episode.elapsed_s,
            "status": episode.status,
            "contact": episode.contact,
            "stable_time_s": episode.stable_time_s,
            "touchdown_count": episode.touchdown_count,
            "vessel_destroyed": vessel_destroyed,
            "observation_stale": observation_stale,
            "reward_defined": False,
        }
        if sample is not None:
            info.update(
                {
                    "situation": sample["situation"],
                    "surface_altitude_m": sample["surface_altitude_m"],
                    "horizontal_error_m": sample["horizontal_error_m"],
                    "tilt_deg": sample["tilt_deg"],
                    "angular_speed_rad_s": sample["angular_speed_rad_s"],
                    "main_engine_fuel": sample["main_engine_fuel"],
                }
            )
        if episode.touchdown is not None:
            info["touchdown"] = dict(episode.touchdown)
            info["landing_error_m"] = episode.touchdown["horizontal_error_m"]
            info["safe_in_target"] = (
                episode.status == "safe_landing"
                and episode.touchdown["horizontal_error_m"] <= safety["target_radius_m"]
            )
        if episode.last_touchdown is not None:
            info["last_touchdown"] = dict(episode.last_touchdown)
        return info

    def step(self, action: Action) -> StepResult:
        """Apply an action and advance one fixed task control interval.

        The interval is ``1 / [scenario].control_hz`` game seconds.  Contact
        detection is provisional: it uses ``Vessel.situation`` and records
        pre-impact and contact telemetry, but does not claim task-level
        landing validation.
        """

        if not isinstance(action, Action):
            raise TypeError("step() expects an Action instance")
        conn = self._require_connection()
        self._require_action_config()
        self._require_observation_config()
        self._require_episode_config()
        self._require_flight()
        if self._episode is not None and self._episode.done:
            raise EpisodeDoneError("episode is done; call reset() before another step")

        control_interval_s = self._control_interval_s()
        control_hz = 1.0 / control_interval_s
        horizon_s = self._episode_horizon_s()
        safety = self._require_safety_config()

        ut_start = None
        ut_end = None
        observation = None
        sample = None
        physics_ticks = 0
        release_error = None
        destroyed = False
        try:
            conn.krpc.hold_tick()
            ut_start = float(conn.space_center.ut)
            self.apply_action(action)
            max_ticks = max(1, int(math.ceil(control_interval_s / 0.001)) + 10)
            stalled = 0
            while True:
                now = float(conn.space_center.ut)
                if now - ut_start + 1e-9 >= control_interval_s:
                    break
                before = now
                conn.krpc.next_tick()
                physics_ticks += 1
                now = float(conn.space_center.ut)
                if now <= before + 1e-9:
                    stalled += 1
                    if stalled >= 3:
                        raise TickError("physics time did not advance while holding a tick")
                else:
                    stalled = 0
                if physics_ticks > max_ticks:
                    raise TickError(
                        f"control interval {control_interval_s:.6f} s not reached "
                        f"after {physics_ticks} physics ticks"
                    )
            ut_end = float(conn.space_center.ut)
            try:
                observation = self.observe()
                sample = self._contact_sample(observation, ut_end)
            except VesselError:
                if self._scene() == "flight" and conn.space_center.active_vessel is None:
                    destroyed = True
                else:
                    raise
        except KSPEnvError:
            raise
        except Exception as exc:
            raise TickError(str(exc)) from exc
        finally:
            try:
                conn.krpc.release_tick()
            except Exception as exc:  # pragma: no cover - server-dependent
                release_error = exc

        if release_error is not None:
            raise TickError(f"failed to release held tick: {release_error}") from release_error

        if destroyed:
            episode = self._episode
            if episode is None or episode.last_observation is None:
                raise TickError("vessel disappeared before an observation was available")
            episode.status = "crash"
            episode.done = True
            info = self._episode_info(
                episode,
                episode.previous_sample,
                ut_start,
                ut_end,
                physics_ticks,
                control_interval_s,
                control_hz,
                safety,
                vessel_destroyed=True,
                observation_stale=True,
            )
            return StepResult(
                observation=episode.last_observation,
                terminated=True,
                truncated=False,
                info=info,
            )

        if observation is None or sample is None or ut_start is None or ut_end is None:
            raise TickError("step completed without an observation")

        dt_s = ut_end - ut_start
        if self._episode is None:
            self._episode = self._initialize_episode(ut_start, sample, observation, safety)
            self._update_episode(
                self._episode, sample, observation, dt_s, safety, horizon_s
            )
        else:
            self._update_episode(
                self._episode, sample, observation, dt_s, safety, horizon_s
            )

        episode = self._episode
        terminated = episode.status in {"safe_landing", "crash", "fuel_exhausted"}
        truncated = episode.status == "timeout"
        info = self._episode_info(
            episode,
            sample,
            ut_start,
            ut_end,
            physics_ticks,
            control_interval_s,
            control_hz,
            safety,
        )
        return StepResult(
            observation=observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, save_name: str | None = None, timeout_s: float = 30.0) -> tuple[Observation, ResetInfo]:
        """Load and verify a named save.

        The save name is not part of the current task contract, so callers must
        pass it explicitly until a team decision records it.
        """

        conn = self._require_connection()
        self._require_observation_config()
        self._require_episode_config()
        self._episode = None

        configured = self._scenario_config().get("save_name")
        name = save_name if save_name is not None else configured
        if is_tbd(name) or not isinstance(name, str) or not name.strip():
            raise ConfigurationError("reset save_name is TBD; pass save_name explicitly")
        name = name.strip()

        space_center = conn.space_center
        started = time.monotonic()
        loads_before = int(conn.krpc.game_state_loads)
        try:
            try:
                conn.krpc.release_tick()
            except Exception:
                pass
            if not space_center.can_save:
                raise ResetError("SpaceCenter.can_save is false; the game cannot be saved")
            space_center.load(name)
            deadline = started + timeout_s
            while int(conn.krpc.game_state_loads) <= loads_before:
                if time.monotonic() > deadline:
                    raise ResetError(f"timed out waiting for save {name!r} to load")
                time.sleep(0.05)
            self._target_frame_cache = None
            scene = self._scene()
            if scene != "flight":
                raise ResetError(f"save {name!r} loaded scene {scene!r}, expected Flight")
            self._neutralize_controls(conn)
            self._controls_dirty = False
            vessel = self._active_vessel()
            observation = self.observe()
            ut_s = float(space_center.ut)
            sample = self._contact_sample(observation, ut_s)
            self._episode = self._initialize_episode(
                ut_s, sample, observation, self._require_safety_config()
            )
        except ResetError:
            raise
        except Exception as exc:
            raise ResetError(str(exc)) from exc

        return observation, ResetInfo(
            save_name=name,
            ut_s=ut_s,
            game_state_loads=int(conn.krpc.game_state_loads),
            scene=scene,
            vessel_name=vessel.name,
            elapsed_s=time.monotonic() - started,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neutralize_controls(conn: Any) -> None:
        vessel = conn.space_center.active_vessel
        if vessel is None:
            return
        control = vessel.control
        control.throttle = 0.0
        control.pitch = 0.0
        control.yaw = 0.0
        control.roll = 0.0
        control.forward = 0.0
        control.right = 0.0
        control.up = 0.0
        control.rcs = False


__all__ = ["KSPEnv", "InfrastructureError"]
