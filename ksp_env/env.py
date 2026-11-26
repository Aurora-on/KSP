"""KSP environment adapter.

The adapter deliberately reuses kRPC's generated SpaceCenter proxies instead
of reimplementing frames, telemetry or control.  This module owns only the
task-specific contract: observation assembly, action validation, reset/step
sequencing and infrastructure error classification.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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
    SnapshotError,
    SnapshotInfo,
    StepResult,
    TickError,
    VesselError,
    is_tbd,
    load_config,
    require_defined,
    require_number,
)
from .geometry import (
    Quaternion,
    Vector3,
    inverse_transform_body_vector,
    normalize_quaternion,
    quaternion_from_unit_vectors,
    rotate_vector,
    transform_attitude,
    transform_body_vector,
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
        self._controlled_vessel: Any | None = None
        self._tick_held = False
        self._session_active = False
        self._session_ut: float | None = None
        self._session_vessel: Any | None = None
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
            self._end_control(conn)
        except TickError:
            pass
        try:
            conn.close()
        except Exception:
            pass
        finally:
            self._conn = None
            self._target_frame_cache = None
            self._tick_held = False
            self._controlled_vessel = None
            self._session_active = False
            self._session_ut = None
            self._session_vessel = None
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

    @staticmethod
    def _require_vector3(value: object, name: str) -> Vector3:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ConfigurationError(f"{name} must contain three numbers")
        components = tuple(require_number(component, name) for component in value)
        return components  # type: ignore[return-value]

    def _body_to_task_rotation(self) -> Quaternion:
        observation = self._observation_config()
        value = observation.get("body_to_task_rotation_xyzw")
        if is_tbd(value):
            raise ConfigurationError("[observation].body_to_task_rotation_xyzw is TBD")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ConfigurationError(
                "[observation].body_to_task_rotation_xyzw must contain four numbers"
            )
        try:
            quaternion = normalize_quaternion(
                tuple(float(component) for component in value)  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "[observation].body_to_task_rotation_xyzw must be a finite nonzero quaternion"
            ) from exc
        if not all(math.isfinite(component) for component in quaternion):
            raise ConfigurationError(
                "[observation].body_to_task_rotation_xyzw must be finite"
            )
        return quaternion

    def _main_engine_fuel_resources(self) -> tuple[str, ...]:
        observation = self._observation_config()
        value = observation.get("main_engine_fuel_resources")
        if value is None and "main_engine_fuel_resource" in observation:
            raise ConfigurationError(
                "[observation].main_engine_fuel_resource is obsolete; use the "
                "main_engine_fuel_resources list and report total mass in kg"
            )
        if is_tbd(value) or not isinstance(value, (list, tuple)) or not value:
            raise ConfigurationError(
                "[observation].main_engine_fuel_resources must be a non-empty list"
            )
        resources: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ConfigurationError(
                    "[observation].main_engine_fuel_resources must contain "
                    "non-empty resource names"
                )
            name = item.strip()
            if name in resources:
                raise ConfigurationError(
                    f"[observation].main_engine_fuel_resources contains duplicate {name!r}"
                )
            resources.append(name)
        return tuple(resources)

    def _require_observation_config(self) -> None:
        observation = self._observation_config()
        frame = observation.get("frame")
        if is_tbd(frame):
            raise ConfigurationError("[observation].frame is TBD")
        if frame != "target_surface":
            raise ConfigurationError(
                f"unsupported [observation].frame {frame!r}; only 'target_surface' is implemented"
            )
        self._main_engine_fuel_resources()
        self._body_to_task_rotation()
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

        This requires ``[observation].frame = 'target_surface'``, a fixed target
        position, a body-to-task rotation and a non-empty
        ``main_engine_fuel_resources`` list. The fuel field is the summed mass
        of those resources in kg.
        """

        conn = self._require_connection()
        self._require_observation_config()
        if self._session_active:
            try:
                self._check_session_boundary(conn)
                observation = self._observe_vessel(self._session_vessel)
                self._check_session_boundary(conn)
                return observation
            except BaseException as exc:
                self._end_control(conn, exc)
                raise
        return self._observe_vessel(self._require_flight())

    def _observe_vessel(self, vessel: Any) -> Observation:
        frame = self._target_reference_frame()
        fuel_resources = self._main_engine_fuel_resources()
        body_to_task = self._body_to_task_rotation()

        resources = vessel.resources
        density_by_name: dict[str, float] = {}
        for resource in resources.all:
            density_by_name.setdefault(str(resource.name), float(resource.density))
        fuel_amounts_kg = []
        for name in fuel_resources:
            if not resources.has_resource(name):
                raise VesselError(
                    f"main-engine resource {name!r} is not present on the vessel"
                )
            if name not in density_by_name:
                raise VesselError(
                    f"main-engine resource {name!r} has no readable density"
                )
            fuel_amounts_kg.append(
                float(resources.amount(name)) * density_by_name[name]
            )

        return Observation(
            position_m=tuple(float(value) for value in vessel.position(frame)),
            velocity_m_s=tuple(float(value) for value in vessel.velocity(frame)),
            attitude_quaternion=transform_attitude(
                tuple(float(value) for value in vessel.rotation(frame)), body_to_task
            ),
            angular_velocity_rad_s=transform_body_vector(
                tuple(float(value) for value in vessel.angular_velocity(frame)),
                body_to_task,
            ),
            main_engine_fuel=math.fsum(fuel_amounts_kg),
        )

    # ------------------------------------------------------------------
    # Actions and stepping
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> None:
        """Send one validated action through kRPC's ``Control`` interface."""

        self._require_action_config()
        self._write_action(self._require_flight(), action)

    def _write_action(self, vessel: Any, action: Action) -> None:
        control = vessel.control

        # Imported lazily so offline tests do not need the kRPC stubs.
        from krpc.services.spacecenter import ControlInputMode

        # Retain this exact vessel even if a later setter fails or the active
        # vessel changes. Cleanup must never send controls to its replacement.
        self._controlled_vessel = vessel
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

    @staticmethod
    def _read_ut(conn: Any) -> float:
        ut = float(conn.space_center.ut)
        if not math.isfinite(ut):
            raise TickError("game UT is not finite")
        return ut

    def _hold_tick(self, conn: Any) -> float:
        # A timed-out hold RPC may still have acquired the server's tick.
        self._tick_held = True
        conn.krpc.hold_tick()
        return self._read_ut(conn)

    def _release_tick(self, conn: Any) -> None:
        self._session_ut = None
        if self._tick_held:
            try:
                conn.krpc.release_tick()
            except Exception as exc:
                raise TickError(f"failed to release held tick: {exc}") from exc
            self._tick_held = False

    def _end_control(self, conn: Any, error: BaseException | None = None) -> None:
        failures = []
        if self._controlled_vessel is not None:
            try:
                self._neutralize_controls(self._controlled_vessel)
                self._controlled_vessel = None
            except Exception as exc:
                failures.append(f"could not neutralize commanded vessel: {exc}")
        try:
            self._release_tick(conn)
        except TickError as exc:
            failures.append(str(exc))
        if failures:
            detail = ". ".join(failures)
            if error is None:
                raise TickError(detail)
            error.add_note(detail)

    def _check_session_boundary(self, conn: Any) -> float:
        if self._session_ut is None or self._read_ut(conn) != self._session_ut:
            raise TickError("control session lost its held tick; game UT changed or session ended")
        if self._require_flight() != self._session_vessel:
            raise VesselError("active vessel changed during the control session")
        return self._session_ut

    @contextmanager
    def control_session(self) -> Iterator[KSPEnv]:
        """Hold consecutive decision boundaries through observation and policy work.

        Enter after reset. The server may expire its hold during inference;
        the next read or step rejects a changed UT before using stale data.
        """
        if self._session_active:
            raise TickError("a control session is already active")
        conn = self._require_connection()
        self._session_active = True
        error = None
        try:
            try:
                self._session_ut = self._hold_tick(conn)
                self._session_vessel = self._require_flight()
                self._check_session_boundary(conn)
            except KSPEnvError:
                raise
            except Exception as exc:
                raise TickError(f"could not start control session: {exc}") from exc
            yield self
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._session_active = False
            self._session_vessel = None
            if self._tick_held:
                self._end_control(conn, error)

    def step(self, action: Action) -> StepResult:
        """Hold the action for the powered-descent interval, or one test tick."""
        return self._step(action, grouped=self.config.raw.get("task") == "powered_descent")

    def step_tick(self, action: Action) -> StepResult:
        """Advance a single measured physics tick for adapter diagnostics."""
        return self._step(action, grouped=False)

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
                    "speed_m_s": sample["speed_m_s"],
                    "vertical_speed_m_s": sample["vertical_speed_m_s"],
                    "horizontal_speed_m_s": sample["horizontal_speed_m_s"],
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


    def _tick_sample(self, conn: Any, vessel: Any, ut: float) -> dict[str, Any]:
        parts = vessel.parts
        legs = parts.legs
        frame = self._target_reference_frame()
        body_to_task = self._body_to_task_rotation()
        flight = vessel.flight(frame)
        position = tuple(float(value) for value in vessel.position(frame))
        velocity = tuple(float(value) for value in vessel.velocity(frame))
        angular_velocity = transform_body_vector(
            tuple(float(value) for value in vessel.angular_velocity(frame)),
            body_to_task,
        )
        sample = {
            "ut_s": ut,
            "situation": _enum_name(vessel.situation),
            "part_count": len(parts.all),
            "leg_count": len(legs),
            "grounded_legs": sum(leg.is_grounded for leg in legs),
            "broken_legs": sum(_enum_name(leg.state) == "broken" for leg in legs),
            "position_m": position,
            "surface_altitude_m": float(flight.surface_altitude),
            "velocity_m_s": velocity,
            "attitude_quaternion": transform_attitude(
                tuple(float(value) for value in vessel.rotation(frame)),
                body_to_task,
            ),
            "angular_velocity_rad_s": angular_velocity,
            "speed_m_s": math.sqrt(sum(value * value for value in velocity)),
            "angular_speed_rad_s": math.sqrt(
                sum(value * value for value in angular_velocity)
            ),
        }
        if self._read_ut(conn) != ut:
            raise TickError("tick hold expired during raw telemetry")
        return sample

    def _tick_contact_sample(
        self, sample: Mapping[str, Any], main_engine_fuel: float
    ) -> dict[str, Any]:
        """Convert one raw tick sample into contact-evaluation telemetry."""

        velocity = sample["velocity_m_s"]
        position = sample["position_m"]
        return {
            "ut_s": float(sample["ut_s"]),
            "situation": sample["situation"],
            "surface_altitude_m": float(sample["surface_altitude_m"]),
            "speed_m_s": float(sample["speed_m_s"]),
            "vertical_speed_m_s": float(velocity[2]),
            "horizontal_speed_m_s": math.hypot(velocity[0], velocity[1]),
            "tilt_deg": self._tilt_deg(sample["attitude_quaternion"]),
            "angular_speed_rad_s": float(sample["angular_speed_rad_s"]),
            "horizontal_error_m": math.hypot(position[0], position[1]),
            "main_engine_fuel": float(main_engine_fuel),
        }

    def _evaluate_result(self, result: StepResult) -> StepResult:
        """Update the provisional episode state from one advanced step."""

        info = result.info
        observation = result.observation
        ut_start = info.get("ut_start_s")
        ut_end = info.get("ut_end_s")
        if ut_start is None or ut_end is None:
            raise TickError("step completed without start/end UT")
        safety = self._require_safety_config()
        horizon_s = self._episode_horizon_s()
        requested_interval = info.get("control_interval_s")
        actual_interval = float(ut_end) - float(ut_start)
        if requested_interval is not None:
            control_interval_s = float(requested_interval)
            control_hz = 1.0 / control_interval_s
        else:
            control_interval_s = None
            control_hz = 1.0 / actual_interval if actual_interval > 0.0 else 1.0

        raw_samples = info.get("tick_samples")
        if raw_samples:
            contact_samples = [
                self._tick_contact_sample(sample, observation.main_engine_fuel)
                for sample in raw_samples
            ]
        else:
            contact_samples = [
                self._contact_sample(observation, float(ut_end))
            ]

        if self._episode is None:
            first = contact_samples[0]
            self._episode = self._initialize_episode(
                float(ut_start), first, observation, safety
            )
            previous_ut = first["ut_s"]
            remaining = contact_samples[1:]
            if not remaining:
                remaining = contact_samples
                previous_ut = float(ut_start)
        else:
            previous_ut = (
                self._episode.previous_sample["ut_s"]
                if self._episode.previous_sample is not None
                else float(ut_start)
            )
            remaining = contact_samples

        for sample in remaining:
            dt_s = float(sample["ut_s"]) - float(previous_ut)
            if dt_s < 0.0:
                raise TickError("tick sample UT moved backwards")
            self._update_episode(
                self._episode, sample, observation, dt_s, safety, horizon_s
            )
            previous_ut = sample["ut_s"]

        episode = self._episode
        terminated = episode.status in {"safe_landing", "crash", "fuel_exhausted"}
        truncated = episode.status == "timeout"
        final_sample = contact_samples[-1]
        info = self._episode_info(
            episode,
            final_sample,
            float(ut_start),
            float(ut_end),
            int(info["physics_ticks"]),
            control_interval_s,
            control_hz,
            safety,
        )
        if raw_samples:
            info["tick_samples"] = raw_samples
        return StepResult(
            observation=observation,
            reward=result.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step(self, action: Action, *, grouped: bool) -> StepResult:
        conn = self._require_connection()
        try:
            if not isinstance(action, Action):
                raise TypeError("step expects an Action instance")
            self._require_action_config()
            self._require_observation_config()
            self._require_episode_config()
            self._require_flight()
            if self._episode is not None and self._episode.done:
                raise EpisodeDoneError("episode is done; call reset() before another step")
            interval = None
            if grouped:
                frequency = require_number(self._scenario_config().get("control_hz"), "[scenario].control_hz")
                if frequency <= 0 or not math.isfinite(1.0 / frequency):
                    raise ConfigurationError("[scenario].control_hz must define a positive finite interval")
                interval = 1.0 / frequency
            result = self._advance(conn, action, interval)
            result = self._evaluate_result(result)
            if self._session_active:
                self._session_ut = result.info["ut_end_s"]
            else:
                self._release_tick(conn)
            return result
        except VesselError:
            if self._tick_held:
                self._end_control(conn)
            if (
                self._scene() == "flight"
                and conn.space_center.active_vessel is None
                and self._episode is not None
                and self._episode.last_observation is not None
            ):
                episode = self._episode
                episode.status = "crash"
                episode.done = True
                safety = self._require_safety_config()
                control_interval_s = self._control_interval_s()
                control_hz = 1.0 / control_interval_s
                ut_s = (
                    float(episode.previous_sample["ut_s"])
                    if episode.previous_sample is not None
                    else episode.started_ut_s
                )
                info = self._episode_info(
                    episode,
                    episode.previous_sample,
                    ut_s,
                    ut_s,
                    0,
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
            raise
        except BaseException as exc:
            if self._tick_held:
                self._end_control(conn, exc)
            raise

    def _advance(self, conn: Any, action: Action, interval: float | None) -> StepResult:
        timeout = require_number(self.config.timeout_s, "timeout_s")
        if timeout <= 0:
            raise ConfigurationError("timeout_s must be positive")
        deadline = time.monotonic() + timeout
        samples = []
        try:
            if self._session_active:
                start = self._check_session_boundary(conn)
                vessel = self._session_vessel
            else:
                start = self._hold_tick(conn)
                vessel = self._require_flight()
            if interval is not None:
                samples.append(self._tick_sample(conn, vessel, start))
            if time.monotonic() >= deadline:
                raise TickError("step exceeded its wall-clock timeout")
            if self._read_ut(conn) != start:
                raise TickError("tick hold expired before applying action")
            self._write_action(vessel, action)
            ut = start
            ticks = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TickError("step exceeded its wall-clock timeout")
                if self._read_ut(conn) != ut:
                    raise TickError("tick hold expired while applying controls or sampling telemetry")
                conn.krpc.next_tick()
                next_ut = self._read_ut(conn)
                tick_dt = next_ut - ut
                if tick_dt <= 0 or not math.isfinite(tick_dt):
                    raise TickError(f"physics tick did not advance game UT by a finite duration: {tick_dt:g} s")
                if self._require_flight() != vessel:
                    raise VesselError("active vessel changed during a step")
                ticks += 1
                ut = next_ut
                if interval is None:
                    break
                samples.append(self._tick_sample(conn, vessel, ut))
                # UT is double but KSP's fixed timestep comes from Unity's
                # float. Allow cumulative rounding without taking a sixth tick.
                tolerance = min(tick_dt / 4, ticks * math.ulp(ut) + interval * 2**-24)
                if ut - start >= interval - tolerance:
                    break

            observation = self._observe_vessel(vessel)
            if self._read_ut(conn) != ut:
                raise TickError("tick hold expired during final observation")
            if time.monotonic() >= deadline:
                raise TickError("step exceeded its wall-clock timeout")
        except InfrastructureError as exc:
            exc.tick_samples = tuple(samples)
            raise
        except KSPEnvError:
            raise
        except Exception as exc:
            error = TickError(str(exc))
            error.tick_samples = tuple(samples)
            raise error from exc

        info = {
            "ut_start_s": start,
            "ut_end_s": ut,
            "ut_delta_s": ut - start,
            "physics_ticks": ticks,
            "control_interval_s": interval,
            "episode_evaluated": False,
        }
        if interval is not None:
            info["tick_samples"] = samples
        return StepResult(observation=observation, info=info)

    # ------------------------------------------------------------------
    # Snapshot creation
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_state(observation: Observation) -> dict[str, Any]:
        return {
            "position_m": list(observation.position_m),
            "velocity_m_s": list(observation.velocity_m_s),
            "attitude_quaternion": list(observation.attitude_quaternion),
            "angular_velocity_rad_s": list(observation.angular_velocity_rad_s),
            "main_engine_fuel": observation.main_engine_fuel,
        }

    def create_initial_save(
        self,
        save_name: str,
        *,
        position_m: tuple[float, float, float],
        velocity_m_s: tuple[float, float, float],
        body_z_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        roll_deg: float = 0.0,
        angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fill_resources: bool = True,
        verify: bool = True,
        overwrite: bool = False,
        save_dir: str | Path | None = None,
    ) -> SnapshotInfo:
        """Create a named low-altitude save with a debug-set initial state.

        The position, velocity, body +z direction and angular velocity are in
        the adapter's target frame.  The method uses kRPC's ``Debug`` service,
        then saves directly through ``SpaceCenter.save``; the latter bypasses
        KSP's UI quicksave restriction.  This is a test/calibration tool, not a
        task-grade reset or landing result.
        """

        if not isinstance(save_name, str) or not save_name.strip():
            raise ConfigurationError("save_name must be a non-empty string")
        save_name = save_name.strip()
        if any(separator in save_name for separator in ("/", "\\")):
            raise ConfigurationError("save_name must not contain path separators")

        conn = self._require_connection()
        self._require_observation_config()
        self._require_flight()
        frame = self._target_reference_frame()
        body_to_task = self._body_to_task_rotation()
        position = self._require_vector3(position_m, "position_m")
        velocity = self._require_vector3(velocity_m_s, "velocity_m_s")
        body_z = self._require_vector3(body_z_direction, "body_z_direction")
        angular = self._require_vector3(angular_velocity_rad_s, "angular_velocity_rad_s")
        roll = require_number(roll_deg, "roll_deg")

        body_z_norm = math.sqrt(sum(value * value for value in body_z))
        if body_z_norm == 0.0:
            raise ConfigurationError("body_z_direction must not be zero")
        body_z = tuple(value / body_z_norm for value in body_z)

        save_path = (
            Path(save_dir) / f"{save_name}.sfs" if save_dir is not None else None
        )
        if save_path is not None and save_path.exists() and not overwrite:
            raise SnapshotError(
                f"snapshot {save_name!r} already exists; pass overwrite=True to replace it"
            )

        vessel = self._require_flight()
        body_name = vessel.orbit.body.name
        target_position = self._target_coordinates()
        requested = {
            "position_m": list(position),
            "velocity_m_s": list(velocity),
            "body_z_direction": list(body_z),
            "roll_deg": roll,
            "angular_velocity_rad_s": list(angular),
        }
        self._neutralize_controls(vessel)
        self._episode = None
        can_save_before = bool(conn.space_center.can_save)
        save_ut_s = None
        game_state_loads = None
        can_save_after = None
        pre_state = None

        try:
            conn.krpc.paused = False
            if fill_resources:
                conn.debug.fill_all_resources()
            # set_direction controls the KSP vessel's +Y axis.  Convert the
            # requested task +Z direction into the corresponding KSP +Y
            # direction through the fixed body-to-task rotation.
            task_z = body_z
            ksp_y_in_task = rotate_vector(body_to_task, (0.0, 1.0, 0.0))
            task_to_target = quaternion_from_unit_vectors((0.0, 0.0, 1.0), task_z)
            ksp_y_in_target = rotate_vector(task_to_target, ksp_y_in_task)
            conn.debug.set_velocity(velocity, frame)
            conn.debug.set_direction(ksp_y_in_target, roll, frame)
            conn.debug.set_angular_velocity(angular, frame)
            conn.debug.set_position(position, frame)
            conn.krpc.paused = True

            pre_observation = self.observe()
            pre_state = self._snapshot_state(pre_observation)
            can_save_after = bool(conn.space_center.can_save)
            conn.space_center.save(save_name)
            save_ut_s = float(conn.space_center.ut)
            game_state_loads = int(conn.krpc.game_state_loads)
        except KSPEnvError:
            raise
        except Exception as exc:
            raise SnapshotError(str(exc)) from exc
        finally:
            try:
                conn.krpc.paused = True
            except Exception:
                pass

        save_sha256 = None
        if save_path is not None and save_path.exists():
            save_sha256 = hashlib.sha256(save_path.read_bytes()).hexdigest()

        observed = pre_state
        verified = False
        if verify:
            try:
                conn.space_center.load(save_name)
                conn.krpc.paused = True
                observed_observation = self.observe()
                observed = self._snapshot_state(observed_observation)
                verified = True
            except Exception as exc:
                raise SnapshotError(
                    f"snapshot {save_name!r} was saved but could not be verified: {exc}"
                ) from exc
            finally:
                try:
                    conn.krpc.paused = True
                except Exception:
                    pass

        return SnapshotInfo(
            save_name=save_name,
            ut_s=save_ut_s if save_ut_s is not None else float(conn.space_center.ut),
            game_state_loads=game_state_loads
            if game_state_loads is not None
            else int(conn.krpc.game_state_loads),
            vessel_name=vessel.name,
            body_name=body_name,
            target_position=target_position,
            requested=requested,
            observed=observed,
            can_save_before=can_save_before,
            can_save_after=can_save_after if can_save_after is not None else False,
            verified=verified,
            save_path=str(save_path) if save_path is not None else None,
            save_sha256=save_sha256,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, save_name: str | None = None, timeout_s: float = 30.0) -> tuple[Observation, ResetInfo]:
        """Load and verify a named save.

        The save name is not part of the current task contract, so callers must
        pass it explicitly until a team decision records it.
        """

        if self._session_active:
            raise ResetError("exit the control session before loading a save")
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
            # can_save describes saving, not loading.  kRPC's load path does
            # not require it and low-altitude saves are loadable.
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
            vessel = self._active_vessel()
            self._neutralize_controls(vessel)
            self._controlled_vessel = None
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
    def _neutralize_controls(vessel: Any) -> None:
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
