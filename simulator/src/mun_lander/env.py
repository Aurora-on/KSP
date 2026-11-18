"""Gymnasium environment with flight, impact and live stability-window dynamics."""
import copy
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import SimConfig
from .math3d import tilt_angle
from .physics import G0, integrate_free
from .reward import compute_reward
from .state import LanderState, sample_initial_state

TERMINAL = {"safe_landing", "crash", "fuel_exhausted", "timeout", "out_of_bounds"}


class LanderEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, config=None, render_mode=None):
        self.config = config if config is not None else SimConfig()
        if not isinstance(self.config, SimConfig):
            raise TypeError("config must be a SimConfig")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("render_mode must be None, ansi or rgb_array")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": max(1, round(1/self.config.control_dt))}
        self.action_space = spaces.Box(np.array([0,-1,-1,-1], np.float32), np.ones(4, np.float32))
        low = np.full(14, -np.inf, dtype=np.float32)
        high = np.full(14, np.inf, dtype=np.float32)
        low[2] = 0
        low[6:10], high[6:10] = -1, 1
        low[13], high[13] = 0, self.config.vehicle.fuel_capacity
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"initial_state"}:
            raise ValueError("reset options only supports initial_state")
        initial = options.get("initial_state")
        state = (initial.copy() if isinstance(initial, LanderState) else LanderState.from_dict(initial)) if initial is not None else sample_initial_state(self.np_random, self.config)
        self._validate_state(state)
        self.state = state
        self.time = 0.0
        self.steps = 0
        self.episode_return = 0.0
        self.initial_fuel = state.fuel
        self.rcs_fuel_used = 0.0
        self.contact = False
        self.stable_time = 0.0
        self.touchdown = None
        self.last_touchdown = None
        self.touchdown_count = 0
        self.status = "flying"
        self._done = False
        return self.state.observation(), self._info(0.0, {})

    def _validate_state(self, state):
        if state.position[2] < 0 or state.position[2] > self.config.max_altitude:
            raise ValueError("initial/checkpoint altitude outside simulation bounds")
        if np.linalg.norm(state.position[:2]) > self.config.max_horizontal_distance:
            raise ValueError("initial/checkpoint horizontal position outside simulation bounds")
        if state.fuel > self.config.vehicle.fuel_capacity + 1e-9:
            raise ValueError("main fuel exceeds capacity")

    def _safe_motion(self):
        c = self.config.landing
        return (np.linalg.norm(self.state.velocity) <= c.max_touchdown_speed + 1e-10
                and math.degrees(tilt_angle(self.state.quaternion)) <= c.max_tilt_deg + 1e-10
                and np.linalg.norm(self.state.angular_velocity) <= c.max_angular_speed + 1e-10)

    def _touch_ground(self):
        s = self.state
        event = {"speed": float(np.linalg.norm(s.velocity)),
                          "vertical_speed": abs(float(s.velocity[2])),
                          "horizontal_speed": float(np.linalg.norm(s.velocity[:2])),
                          "tilt_deg": math.degrees(tilt_angle(s.quaternion)),
                          "angular_speed": float(np.linalg.norm(s.angular_velocity)),
                          "horizontal_error": float(np.linalg.norm(s.position[:2])),
                          "time": self.time}
        self.last_touchdown = event
        self.touchdown_count += 1
        if self.touchdown is None:
            self.touchdown = copy.deepcopy(event)
        s.position[2] = 0.0
        if not self._safe_motion():
            self.status = "crash"
            return
        # Inelastic normal impact. Tangential impact impulse obeys Coulomb bound.
        speed = np.linalg.norm(s.velocity[:2])
        impulse = max(0.0, -s.velocity[2])
        if speed > 0:
            s.velocity[:2] *= max(0.0, 1 - self.config.landing.ground_friction*impulse/speed)
        s.velocity[2] = 0.0
        self.contact = True
        self.stable_time = 0.0
        self.status = "stabilizing"

    def _flight(self, action, dt):
        """Locate contact before integrating through it; return actual elapsed time."""
        before = self.state
        if before.position[2] <= 1e-12 and before.velocity[2] <= 0:
            self._touch_ground()
            return 0.0
        after, rcs = integrate_free(before, action, self.config.vehicle, dt)
        used_dt = dt
        impact_bracket = dt if after.position[2] <= 0 else None
        # A downward/upward reversal can cross the plane and emerge within one
        # substep. Bracket the interior minimum, then find the FIRST impact.
        if (impact_bracket is None and before.velocity[2] < 0 < after.velocity[2]
                and before.position[2] <= abs(before.velocity[2])*dt):
            lo, hi = 0.0, dt
            for _ in range(30):
                mid = (lo+hi)/2
                trial, _ = integrate_free(before, action, self.config.vehicle, mid)
                if trial.velocity[2] < 0:
                    lo = mid
                else:
                    hi = mid
            minimum, _ = integrate_free(before, action, self.config.vehicle, hi)
            if minimum.position[2] <= 0:
                impact_bracket = hi
        if impact_bracket is not None:
            lo, hi = 0.0, impact_bracket
            for _ in range(30):
                mid = (lo+hi)/2
                trial, _ = integrate_free(before, action, self.config.vehicle, mid)
                if trial.position[2] > 0:
                    lo = mid
                else:
                    hi = mid
            used_dt = hi
            after, rcs = integrate_free(before, action, self.config.vehicle, used_dt)
            self.state = after
            self.time += used_dt
            self.rcs_fuel_used += rcs
            self._touch_ground()
        else:
            self.state = after
            self.time += used_dt
            self.rcs_fuel_used += rcs
        return used_dt

    def _ground(self, action, dt):
        """Unilateral planar support, friction and finite angular damping.

        This is a point-contact training approximation, not landing-gear FEM.
        Attitude and controls continue evolving and thrust can cause liftoff.
        """
        before = self.state
        after, rcs = integrate_free(before, action, self.config.vehicle, dt)
        self.time += dt
        self.rcs_fuel_used += rcs
        self.state = after
        if after.position[2] > 1e-9 and after.velocity[2] > 0:
            self.contact = False
            self.stable_time = 0.0
            self.status = "flying"
            return
        normal_impulse = max(0.0, -after.velocity[2])
        speed = np.linalg.norm(after.velocity[:2])
        if speed > 0:
            after.velocity[:2] *= max(0.0, 1-self.config.landing.ground_friction*normal_impulse/speed)
        after.position[:2] = before.position[:2] + (before.velocity[:2]+after.velocity[:2])*(dt/2)
        after.position[2] = 0.0
        after.velocity[2] = 0.0
        after.angular_velocity *= math.exp(-self.config.landing.ground_angular_damping*dt)
        if not self._safe_motion():
            self.status = "crash"
        else:
            self.stable_time += dt
            self.status = "safe_landing" if self.stable_time >= self.config.landing.stability_time-1e-10 else "stabilizing"

    def step(self, action):
        if self.state is None or self._done:
            raise gym.error.ResetNeeded("Call reset() before step() and after episode completion")
        u = np.asarray(action, dtype=np.float64)
        if u.shape != (4,) or not np.all(np.isfinite(u)):
            raise ValueError("action must be a finite vector of shape (4,)")
        applied = np.clip(u, [0,-1,-1,-1], [1,1,1,1])
        start_time = self.time
        start_fuel, start_rcs = self.state.fuel, self.rcs_fuel_used
        remaining = min(self.config.control_dt, max(0.0, self.config.max_time-self.time))
        while remaining > 1e-10 and self.status not in TERMINAL:
            # An exhausted engine is an abort in flight; a supported craft can settle.
            if self.state.fuel <= 1e-10 and not self.contact:
                # Permit an already-touching safe craft to begin its stability window.
                if self.state.position[2] <= 1e-12 and self.state.velocity[2] <= 0:
                    self._touch_ground()
                    if self.status in TERMINAL:
                        break
                else:
                    self.status = "fuel_exhausted"
                    break
            dt = min(self.config.physics_dt, remaining)
            if self.contact:
                dt = min(dt, self.config.landing.stability_time-self.stable_time)
            burn_rate = applied[0]*self.config.vehicle.max_thrust/(self.config.vehicle.main_isp*G0)
            if burn_rate > 0 and self.state.fuel > 1e-10:
                dt = min(dt, self.state.fuel/burn_rate)
            elapsed = dt if self.contact else None
            if self.contact:
                self._ground(applied, dt)
            else:
                elapsed = self._flight(applied, dt)
            remaining -= elapsed
            if self.status not in TERMINAL:
                if np.linalg.norm(self.state.position[:2]) > self.config.max_horizontal_distance or self.state.position[2] > self.config.max_altitude:
                    self.status = "out_of_bounds"
                elif self.state.fuel <= 1e-10 and not self.contact:
                    self.status = "fuel_exhausted"
        if self.status not in TERMINAL and self.time >= self.config.max_time-1e-9:
            self.status = "timeout"
        dt = self.time-start_time
        terms = compute_reward(self.state, start_fuel-self.state.fuel, self.rcs_fuel_used-start_rcs, dt, self.status, self.config)
        reward = float(sum(terms.values()))
        self.steps += 1
        self.episode_return += reward
        self._done = self.status in TERMINAL
        truncated = self.status in ("timeout", "out_of_bounds")
        terminated = self._done and not truncated
        info = self._info(dt, terms)
        info.update(applied_action=applied.astype(np.float32), action_clipped=bool(np.any(u != applied)))
        return self.state.observation(), reward, terminated, truncated, info

    def _info(self, dt, terms):
        s = self.state
        main_used = max(0.0, self.initial_fuel-s.fuel)
        error = float(np.linalg.norm(s.position[:2]))
        info = {"time": self.time, "dt": dt, "status": self.status,
                "is_success": self.status == "safe_landing",
                "safe_in_target": self.status == "safe_landing" and error <= self.config.landing.target_radius,
                "horizontal_error": error, "main_fuel_used": main_used,
                "rcs_fuel_used": self.rcs_fuel_used, "total_fuel_used": main_used+self.rcs_fuel_used,
                "mass": self.config.vehicle.dry_mass+s.fuel,
                "tilt_deg": math.degrees(tilt_angle(s.quaternion)),
                "angular_speed": float(np.linalg.norm(s.angular_velocity)),
                "contact": self.contact, "stable_time": self.stable_time,
                "touchdown_count": self.touchdown_count,
                "reward_terms": dict(terms)}
        # Optional nested dictionaries must be absent, rather than None, for
        # Gymnasium vector environments to combine mixed flight/contact phases.
        if self.touchdown is not None:
            info["touchdown"] = copy.deepcopy(self.touchdown)
            info["last_touchdown"] = copy.deepcopy(self.last_touchdown)
        if self._done:
            info["episode"] = {**copy.deepcopy(info), "return": self.episode_return, "length": self.steps}
        return info

    def get_state(self):
        """JSON-serializable full episode + RNG checkpoint, including post-done states."""
        if self.state is None:
            raise gym.error.ResetNeeded("reset before checkpointing")
        return {"schema_version": 1, "config": self.config.to_dict(), "state": self.state.to_dict(),
                "rng": copy.deepcopy(self.np_random.bit_generator.state),
                "action_rng": copy.deepcopy(self.action_space.np_random.bit_generator.state),
                "time": self.time, "steps": self.steps, "episode_return": self.episode_return,
                "initial_fuel": self.initial_fuel, "rcs_fuel_used": self.rcs_fuel_used,
                "contact": self.contact, "stable_time": self.stable_time,
                "touchdown": copy.deepcopy(self.touchdown),
                "last_touchdown": copy.deepcopy(self.last_touchdown), "touchdown_count": self.touchdown_count,
                "status": self.status, "done": self._done}

    def set_state(self, checkpoint):
        """Validate fully before replacing an episode; failed restores are atomic."""
        c = copy.deepcopy(checkpoint)
        if c.get("schema_version") != 1 or c.get("config") != self.config.to_dict():
            raise ValueError("checkpoint schema/configuration mismatch")
        state = LanderState.from_dict(c["state"])
        # Out-of-bounds terminal states are valid checkpoints.
        if state.position[2] < 0 or state.fuel > self.config.vehicle.fuel_capacity+1e-9:
            raise ValueError("invalid checkpoint state")
        for name in ("time", "initial_fuel", "rcs_fuel_used", "stable_time"):
            if not math.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"invalid checkpoint {name}")
        if not math.isfinite(c["episode_return"]):
            raise ValueError("invalid checkpoint episode_return")
        for name in ("steps", "touchdown_count"):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 0:
                raise ValueError(f"invalid checkpoint {name}")
        if not isinstance(c["done"], bool) or not isinstance(c["contact"], bool):
            raise ValueError("checkpoint done/contact must be booleans")
        if c["status"] not in TERMINAL | {"flying", "stabilizing"} or c["done"] != (c["status"] in TERMINAL):
            raise ValueError("invalid checkpoint status")
        if state.fuel > c["initial_fuel"]+1e-9 or c["initial_fuel"] > self.config.vehicle.fuel_capacity+1e-9:
            raise ValueError("inconsistent checkpoint fuel accounting")
        if c["time"] > self.config.max_time+1e-8 or c["stable_time"] > self.config.landing.stability_time+1e-8:
            raise ValueError("checkpoint time exceeds configured limit")
        if c["stable_time"] > c["time"]+1e-8:
            raise ValueError("checkpoint stable time exceeds elapsed time")
        if c["contact"] and (abs(state.position[2]) > 1e-8 or abs(state.velocity[2]) > 1e-8):
            raise ValueError("supported checkpoint must be on the plane with zero vertical velocity")
        if not c["contact"] and c["stable_time"] > 1e-8:
            raise ValueError("airborne checkpoint cannot accumulate stable time")
        if (c["status"] == "stabilizing" and not c["contact"]) or (c["status"] == "flying" and c["contact"]):
            raise ValueError("checkpoint phase disagrees with contact")
        if c["status"] == "safe_landing" and (not c["contact"] or c["stable_time"] < self.config.landing.stability_time-1e-8):
            raise ValueError("safe checkpoint requires completed contact window")
        if c["status"] == "stabilizing" and c["stable_time"] >= self.config.landing.stability_time:
            raise ValueError("completed stability window must have terminal status")
        if c["touchdown_count"] == 0:
            if c["touchdown"] is not None or c["last_touchdown"] is not None or c["contact"]:
                raise ValueError("checkpoint contact history is inconsistent")
        else:
            for name in ("touchdown", "last_touchdown"):
                event = c[name]
                fields = ("speed", "vertical_speed", "horizontal_speed", "tilt_deg", "angular_speed", "horizontal_error", "time")
                if not isinstance(event, dict) or any(k not in event or not math.isfinite(event[k]) or event[k] < 0 for k in fields):
                    raise ValueError("invalid checkpoint touchdown event")
                if event["time"] > c["time"]+1e-8:
                    raise ValueError("checkpoint touchdown is in the future")
            if c["last_touchdown"]["time"] < c["touchdown"]["time"]:
                raise ValueError("checkpoint touchdown order is inconsistent")
        episode_rng, action_rng = np.random.default_rng(), np.random.default_rng()
        try:
            episode_rng.bit_generator.state = c["rng"]
            action_rng.bit_generator.state = c["action_rng"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("invalid checkpoint random generator state") from exc
        self.state = state
        self.np_random = episode_rng
        self.action_space._np_random = action_rng
        for name in ("time", "steps", "episode_return", "initial_fuel", "rcs_fuel_used", "contact", "stable_time", "touchdown", "last_touchdown", "touchdown_count", "status"):
            setattr(self, name, c[name])
        self._done = c["done"]

    def render(self):
        if self.state is None:
            raise gym.error.ResetNeeded("reset before rendering")
        if self.render_mode == "ansi":
            return f"t={self.time:6.2f}s p={np.round(self.state.position,2)} v={np.round(self.state.velocity,2)} fuel={self.state.fuel:.2f}kg {self.status}"
        if self.render_mode == "rgb_array":
            # Dependency-free side/top projections. High quality plots live in demo CLI.
            canvas = np.full((400,640,3), [15,22,37], dtype=np.uint8)
            canvas[360:363,:,:] = [95,110,128]
            scale = 2.5
            x = int(np.clip(320+self.state.position[0]*scale, 6, 633))
            y = int(np.clip(360-self.state.position[2]*scale, 6, 353))
            canvas[y-5:y+6,x-5:x+6] = [90,210,240]
            canvas[353:360,315:326] = [130,230,140]
            return canvas
        return None

