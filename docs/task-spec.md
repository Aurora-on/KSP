# Mun landing task contract

The team adopted the qualitative Mun task on 2026-11-17 and froze its nominal initial-state distribution, timing, observation fields and safety limits on 2026-11-25. Craft, site and KSP landing acceptance are pending. [Decision record](nominal-task-decisions.md)

The KSP adapter and the fast pretraining simulator MUST expose the same observation, action, timing and episode semantics. `configs/mun_landing.toml` mirrors the machine-readable subset of this contract; when the two disagree, this document and the config MUST be reconciled before an experiment is run.

## 1. Nominal scenario

- Target body: Mun.
- Target site: flat surface. Coordinates and local frame are **TBD**.
- Lander: one fixed nominal craft. Mass, inertia tensor, engine and RCS properties are **TBD**.
- Gravity: fixed nominal Mun gravity. OOD gravity is an evaluation condition, not the nominal task.
- Initialization: randomize initial position, velocity and attitude for every training and evaluation method. Use the S3 distribution for target-relative position, velocity, attitude and body angular velocity. Its support and sampling law are recorded in the [decision table](nominal-task-decisions.md).
- Sample initial clearance uniformly from 80 to 120 m, hold commands at 10 Hz of game time, and end the episode at 90 s. The target point and named descent save still require KSP measurements.

The nominal task is fixed, but the agent must still see randomized initial states. "Fixed craft" does not mean fixed initial state.

## 2. Observation contract

The policy consumes numeric telemetry, not images:

| Field | Meaning | Status |
| --- | --- | --- |
| Position | Position relative to the target, in a defined surface-tangent frame | Frame and sign conventions TBD |
| Velocity | Velocity relative to the target/rotating body, in the same frame | Frame and convention TBD |
| Attitude | Attitude quaternion, with a stated body-to-world order and normalization | Convention TBD |
| Angular velocity | Body angular velocity | Frame and units TBD |
| Main-engine fuel | Remaining main-engine propellant | Unit and normalization TBD |

All values use SI units unless degrees are explicitly named. Frame transformations MUST be explicit and shared by both environments. Mass and inertia are diagnostic telemetry, excluded from policy and world-model input.

Implementation note: the adapter reads a named main-engine propellant resource; the resource name, unit and normalization remain **TBD** until the lander is fixed. kRPC returns attitude quaternions as `(x, y, z, w)` and angular velocity in rad/s; adopting that order and frame must be recorded with the convention decision.

Do not hide an autopilot or attitude controller inside the environment. If one is used, it changes the task dynamics and MUST be documented as part of the action interface.

## 3. Action contract

- RCS pitch, yaw and roll commands.
- Main-engine throttle in `[0, 1]`.
- RCS semantics and action scaling still require calibration. The nominal control interval is 0.1 s of game time.
- The action is held for one fixed control interval. Wall-clock timing is not a substitute for simulation-time stepping.
- An inner-loop autopilot is disabled by default; any exception MUST be recorded in the experiment configuration.

## 4. Reset and step semantics

- Reset uses a named save/snapshot, checks `can_save` and `game_state_loads`, reacquires state and verifies post-reset tolerances.
- The exact save name, snapshot hash and tolerance are **TBD**.
- Stepping uses kRPC 0.7 `hold_tick`, `next_tick` and `release_tick` where suitable, with the tick released in `finally`.
- Record the actual universal-time delta for every step and compare it with the intended action interval.
- Do not wait for streams while holding a tick. Use measured timing to detect hold timeouts.
- Return Gymnasium-style `terminated` for task outcomes and `truncated` for time limits. Connection, reset and tick failures are infrastructure errors, not successful or failed landings.

Implementation note: the core adapter uses Gymnasium-compatible `terminated`/`truncated` semantics but does not depend on the `gymnasium` package. A `gym.Env` wrapper is deferred until the reward, observation space and action space are fixed.

## 5. Safety, precision and failure

A safe landing requires all of the following:

- Touchdown speed, tilt and angular velocity within limits.
- Stable observation for 5 s after touchdown.
- The vessel survives and the relevant parts remain intact.

The limits are 2.0 m/s for total touchdown speed, 10° for tilt, 0.2 rad/s for the angular-speed norm and 5.0 m for the target radius. The shared angular-speed threshold is stored in degrees/s. KSP contact detection and damage checks still require validation.

Metric definitions:

- **Safe-landing rate:** safe landings divided by all attempts.
- **Precision rate:** attempts within the target radius divided by all attempts. The denominator includes unsafe attempts.
- **Landing error:** distance from the target in the surface tangent plane; summary statistics use safe landings only.
- **Propellant consumed:** total consumed propellant mass; report main-engine and RCS breakdowns on safe landings.

Crashes, timeouts and fuel-exhaustion aborts count as task failures. Connection loss, reset failure and tick timeouts are logged separately as infrastructure errors and handled consistently across methods.

The main engine consumes fuel and changes mass normally. RCS reserves are treated as ample, with negligible depletion dynamics; this assumption MUST be recorded and checked if RCS usage becomes material.

## 6. Data and budget accounting

Record at least:

- KSP training transitions and KSP evaluation transitions separately.
- Simulator steps and simulation seconds.
- Reset count and wall-clock time.
- Policy updates, model updates and GPU hours.
- Dataset hash, split, policy version and code revision.

Imagined transitions are never counted as KSP data. RQ2 variants use the same recorded KSP dataset with identical splits.

## 7. Unresolved numerical parameters

The following MUST remain marked `TBD` until a decision is recorded:

- Lander model and its mass/inertia/engine/RCS parameters.
- Target coordinates, local frame, site elevation and clearance reference.
- Named descent save, reset construction and achievable readback tolerances.
- Touchdown-detection tolerance and the mapping from KSP contact signals to an uninterrupted stable interval.
- KSP interaction budget and per-group simulator budget.
- OOD perturbation ranges.
- Observation normalization and action scaling.

`initial_state_ready`, `success_thresholds_ready` and `reset_step_validated` stay `false` until the corresponding acceptance checks pass.
