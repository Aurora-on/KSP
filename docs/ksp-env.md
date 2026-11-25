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

## Adapter test profiles

Two non-task profiles exercise the live adapter:

- `configs/ksp_adapter_test.toml`: Kerbin launchpad save file `persistent`,
  target at the launchpad.
- `configs/ksp_mun_test.toml`: Mun orbit save file `Mun landing 1`, target at
  the sub-satellite point of the saved orbit.

Both are explicitly marked `status = "adapter_test_only"` and keep
`initial_state_ready`, `success_thresholds_ready` and
`reset_step_validated` false. Do not report results from these profiles as Mun
task results. The Mun test target is not a frozen final landing site.

```python
env = KSPEnv.from_config("configs/ksp_mun_test.toml")
```

The profile save names are machine-local and are passed to kRPC's
`SpaceCenter.load`, which expects the save file name inside the current save
folder (for example `persistent` or `Mun landing 1`), not the folder name.
Replace them or move them to a local manifest when using the profiles on another
machine. Live `observe()`, `reset()` and one zero-action `step()` have been
verified against both local profiles; these are mechanical checks only, not task
validation.

## Control response probe

`scripts/probe_controls.py` exercises the main engine and RCS without changing
the task contract. It is read-only by default; `--execute` sends short pulses.

```sh
.venv/bin/python scripts/probe_controls.py \
  --config configs/ksp_mun_test.toml --execute \
  --stage-once --output reports/control-response.json
```

- `--stage-once` activates at most **one** stage, and only when no engine is
  active. It never loops through stages, because a second activation could fire
  separation motors or decouplers. Staging cannot be undone.
- The engine probe applies a short throttle pulse (default 5 ticks at 0.05) and
  records thrust, mass and propellant deltas. Engine thrust can persist for one
  tick after the throttle command; the script waits for it to settle.
- Each RCS axis gets a short pulse and an equal counter-pulse (default 10 ticks
  at 0.5). Peak and residual angular speed are recorded, along with the RCS
  propellant delta for that axis. Monopropellant consumption is reported
  explicitly and aggregated across axes.
- Throttle is zeroed and RCS/SAS/reaction-wheel/input-mode toggles are restored
  from the pre-probe snapshot. The **stage is not restored**; re-save a new
  initial state if the active engine configuration should become the episode
  start.

The script writes `reports/control-response.json` when `--output` is given.

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

### Stepping by game time

With `task = "powered_descent"`, `step(action)` applies the command once and
advances physics ticks until elapsed universal time (UT) reaches
`1 / scenario.control_hz`. The nominal task requests 0.1 s. At 0.02 s per
tick this takes five ticks. If the target falls between tick boundaries, the
step ends on the first boundary that reaches it. Use the measured UT
difference as the transition duration.

For a single-tick probe, call `step_tick(action)`. The two bundled
`adapter_test_only` profiles also retain single-tick behavior through
`step(action)`. Both entry points require a configured target frame, a main
engine fuel resource and `[action].rcs_semantics = "normalized_input"`.
They apply the action through `Vessel.control` with `input_mode = override`,
SAS and reaction wheels disabled.

Each successful hold supplies timing data in `StepResult.info`.

| Field | Meaning |
| --- | --- |
| `ut_start_s`, `ut_end_s`, `ut_delta_s` | Start and end game time in seconds and their measured difference. |
| `physics_ticks`, `control_interval_s` | Count of completed `next_tick()` calls and the requested task interval in seconds. Single-tick probes have `control_interval_s=None`. |
| `tick_samples` | In a task step, a sample at the start and after each tick. Each carries `ut_s`, vessel `situation`, `part_count` and the counts `leg_count`, `grounded_legs`, `broken_legs`. |

Leg contact, leg damage and changes in part count provide raw evidence for
calibrating touchdown and integrity checks. The action hold continues while
the vessel remains available. On a timing or vessel failure during the hold,
`InfrastructureError.tick_samples` retains the samples already verified at a
single UT. UT must stay fixed while a tick is held and advance on each
`next_tick()`. A long UT increment during that call can reflect a long physics
tick, missed ticks or warp. The adapter checks its wall-clock deadline between
RPCs. A blocked socket read in the kRPC 0.7 client can outlast that deadline.

For consecutive policy decisions, enter a control session on the configured
and connected task environment after reset. Read the first observation inside
the session.
The physics tick stays held between decisions so the next action starts at the
previous action's end UT.

```python
with env.control_session():
    observation = env.observe()
    for _ in range(number_of_decisions):
        action = policy(observation)  # returns an Action in the adapter's order
        result = env.step(action)
        observation = result.observation
```

Keep policy calls shorter than the kRPC server's hold timeout. If physics
advances while the adapter expects the boundary to be held, the next step
raises `TickError`. On exit the session attempts to neutralize the commanded
vessel before releasing the hold. A successful standalone `step` leaves its
action applied and releases the tick at the end of the interval. Game time can
then advance before the next call. Record that gap separately when inspecting
standalone traces.

The returned `StepResult` has `reward=None`, `terminated=False`,
`truncated=False` and `info["episode_evaluated"]=False`. Task rewards, the
episode horizon and the landing evaluator still need implementation. The live
checks recorded above cover the single-tick profiles. The 10 Hz task interval
still needs a game run with the selected descent save.

### Reset

`reset(save_name=...)` loads a named save, waits for `game_state_loads` to
advance, requires the Flight scene, and returns `(Observation, ResetInfo)`.
No save name has been chosen yet, so it must be supplied explicitly.
Loading does not depend on `SpaceCenter.can_save`; that flag only describes
whether the current game state can be saved. Low-altitude saves remain
loadable even when `can_save` is false.

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

# Mun orbit profile instead of the default Kerbin launchpad profile
KSP_LIVE_RESET_STEP=1 KSP_TEST_CONFIG=configs/ksp_mun_test.toml \
  .venv/bin/python -m unittest discover -s tests -p 'test_ksp_env_live.py' -v
```

The reset/step commands load the save named in the selected profile and write
control inputs; they are not read-only and are not task validation. Both live
tests are skipped by default and are never run in CI.
