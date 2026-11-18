# KSP environment adapter

Status: importable adapter skeleton. Connection, config loading and read-only
diagnostics are implemented. Task observation, reset and step are gated by
values that remain `TBD` in [`task-spec.md`](task-spec.md) and
[`configs/mun_landing.toml`](../configs/mun_landing.toml). No Mun landing has
been validated.

The adapter lives in `ksp_env/`; [`scripts/`](../scripts) keeps operational
diagnostics (`doctor.py`, `telemetry.py`, `check_environment.py`). The package
does not import from `scripts/`.

## Layout

```text
ksp_env/
├── __init__.py       # public exports
├── contracts.py      # Observation, Action, StepResult, errors, config
└── env.py            # KSPEnv: connect / reset / step / close
```

The package has no mandatory NumPy, Gymnasium or PyTorch import. kRPC is
imported lazily inside `connect()` and `apply_action()` so offline tests and CI
do not need the game.

## kRPC reuse map

The adapter must reuse kRPC instead of reimplementing its transport or math.

| Need | kRPC API |
| --- | --- |
| Connection, version, services | `krpc.connect`, `KRPC.get_status`, `KRPC.get_services` |
| Scene and pause | `KRPC.game_scene`, `KRPC.paused` |
| Named-save reset | `SpaceCenter.save/load/can_save`, `KRPC.game_state_loads` |
| Target-relative frame | `CelestialBody.surface_reference_frame(lat, lon, alt)` |
| Position and velocity | `Vessel.position(frame)`, `Vessel.velocity(frame)` |
| Attitude | `Vessel.rotation(frame)` → `(x, y, z, w)` quaternion |
| Angular velocity | `Vessel.angular_velocity(frame)` → rad/s |
| Altitude, surface speed, vertical speed | `Vessel.flight(frame)` |
| Manual frame transforms | `SpaceCenter.transform_position/velocity/rotation/direction` |
| Main-engine fuel | `Vessel.resources`, `Engine.propellants` |
| RCS and throttle | `Vessel.control`, `Control.pitch/yaw/roll/throttle` |
| Physics-tick stepping | `KRPC.hold_tick/next_tick/release_tick` |
| Landing state | `Vessel.situation` (`landed`, `splashed`, …), `Flight.surface_altitude` |
| Scenario/OOD helpers | `Debug.gravity_multiplier`, `Debug.set_*`, `Engine.thrust_limit` |

`AutoPilot` is deliberately not used: it would hide an inner-loop controller
inside the environment and change the task dynamics.

## Adapter test profile

`configs/ksp_adapter_test.toml` is a non-task profile for exercising the live
adapter on the local Kerbin launchpad save. It fixes a test target, a test fuel
resource and normalized RCS input so `observe()`, `reset()` and `step()` can be
tested before the Mun task is frozen.

It is explicitly marked `status = "adapter_test_only"` and keeps
`initial_state_ready`, `success_thresholds_ready` and
`reset_step_validated` false. Do not report results from this profile as Mun
task results.

```python
env = KSPEnv.from_config("configs/ksp_adapter_test.toml")
```

The profile's save name is machine-local and is passed to kRPC's
`SpaceCenter.load`, which expects the save file name inside the current save
folder (for example `persistent`), not the folder name. Replace it or move it to
a local manifest when using the profile on another machine. Live `observe()`,
`reset()` and one zero-action `step()` have been verified against the local
launchpad; these are mechanical checks only, not task validation.

## Public API

```python
from ksp_env import KSPEnv, Action

env = KSPEnv.from_config()          # does not connect
env.connect()
try:
    state = env.read_diagnostic_state()   # raw, non-task telemetry
    # observation = env.observe()         # requires TBD target/frame decisions
    # result = env.step(Action(...))      # requires TBD RCS semantics
    # observation, info = env.reset(save_name="...")
finally:
    env.close()
```

`read_diagnostic_state()` is a live connection check. It reads body-frame
position/velocity, attitude, angular velocity, mass/inertia, resources and
engines. It is **not** the task observation.

`observe()` builds the `Observation` defined in `docs/task-spec.md`. It refuses
to run until all of the following are defined:

- `[observation].frame = "target_surface"`
- `[scenario].target_position` (latitude, longitude, altitude)
- `[observation].main_engine_fuel_resource`

`step(action)` additionally requires `[action].rcs_semantics =
"normalized_input"`. It applies the action through `Vessel.control` with
`input_mode = override`, SAS and reaction wheels disabled, then advances exactly
one held physics tick and records the UT delta. Episode evaluation is **not**
implemented, so `StepResult.info["episode_evaluated"]` is `False` and both
`terminated` and `truncated` are false. This is a transport-level step, not a
validated landing step.

`reset(save_name=...)` loads a named save, waits for `game_state_loads` to
advance, requires the Flight scene, and returns `(Observation, ResetInfo)`.
No save name has been chosen yet, so it must be supplied explicitly.

## Error taxonomy

| Exception | Meaning |
| --- | --- |
| `ConfigurationError` | Missing decision, including any `TBD` field |
| `KSPConnectionError` | Connection or server-version/service failure |
| `VesselError` | No active vessel, or wrong scene |
| `ResetError` | Named save could not be loaded/verified |
| `TickError` | Held tick or step timing failed |
| `InfrastructureError` | Base class for the four infrastructure failures above |

Infrastructure failures must never be converted into a successful or failed
landing. Gymnasium-style `terminated`/`truncated` are reserved for task outcomes.

## kRPC 0.7 notes

- Streams only send when the value changes. A constant mass may update once;
  do not use stream age as a liveness signal.
- `Control` getters return the input applied on the previous tick, so readback
  immediately after a set can be stale.
- `Control.input_mode` defaults to additive; the adapter uses `override`.
- `Vessel.resources_in_decouple_stage` and `Parts.in_stage` were removed in 0.7.

## Dependencies and tests

The KSP-interface-only environment needs:

```sh
uv pip install --python .venv/bin/python \
  krpc==0.7.0 protobuf==7.36.2 numpy==2.5.4 gymnasium==1.4.0 matplotlib==3.11.2
```

`numpy`, `gymnasium` and `matplotlib` are pinned in
`requirements/runtime.txt`; PyTorch is not needed for adapter work.

Offline checks:

```sh
python3 -m compileall -q ksp_env scripts tests
python3 -m unittest discover -s tests -v
```

Live read-only smoke test (game loaded, kRPC server running):

```sh
KSP_LIVE=1 .venv/bin/python -m unittest discover -s tests -p 'test_ksp_env_live.py' -v
```

Live adapter mechanics, including named-save reset and one zero-action
physics step:

```sh
KSP_LIVE_RESET_STEP=1 .venv/bin/python -m unittest discover -s tests -p 'test_ksp_env_live.py' -v
```

The second command loads the save named in `configs/ksp_adapter_test.toml` and
writes control inputs; it is not read-only and is not task validation. Both live
tests are skipped by default and are never run in CI.
