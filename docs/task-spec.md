# Mun landing task contract

Status: the qualitative task is adopted from `KSP_World_Model_Proposal_final.pptx` (reviewed 2026-11-17). Numeric values marked **TBD** are deliberately unset. This document is not a runnable environment and does not claim that any landing has been validated.

The KSP adapter and the fast pretraining simulator MUST expose the same observation, action, timing and episode semantics. `configs/mun_landing.toml` mirrors the machine-readable subset of this contract; when the two disagree, this document and the config MUST be reconciled before an experiment is run.

## 1. Nominal scenario

- Target body: Mun.
- Target site: flat surface. Coordinates and local frame are **TBD**.
- Lander: one fixed nominal craft. Mass, inertia tensor, engine and RCS properties are **TBD**.
- Gravity: fixed nominal Mun gravity. OOD gravity is an evaluation condition, not the nominal task.
- Initialization: randomize initial position, velocity and attitude for every training and evaluation method. Ranges and distributions are **TBD**.
- Descent altitude, target point, episode horizon and control rate are **TBD**.

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

All values use SI units unless degrees are explicitly named. Frame transformations MUST be explicit and shared by both environments. Whether vehicle mass and inertia are observed, or only used to condition OOD evaluation, is **TBD**; RQ3 requires the decision to be recorded before training.

Do not hide an autopilot or attitude controller inside the environment. If one is used, it changes the task dynamics and MUST be documented as part of the action interface.

## 3. Action contract

- RCS pitch, yaw and roll commands.
- Main-engine throttle in `[0, 1]`.
- RCS semantics (torque command versus normalized input), action scaling and control interval are **TBD** and MUST match between the KSP adapter and the simulator.
- The action is held for one fixed control interval. Wall-clock timing is not a substitute for simulation-time stepping.
- An inner-loop autopilot is disabled by default; any exception MUST be recorded in the experiment configuration.

## 4. Reset and step semantics

- Reset uses a named save/snapshot, checks `can_save` and `game_state_loads`, reacquires state and verifies post-reset tolerances.
- The exact save name, snapshot hash and tolerance are **TBD**.
- Stepping uses kRPC 0.7 `hold_tick`, `next_tick` and `release_tick` where suitable, with the tick released in `finally`.
- Record the actual universal-time delta for every step and compare it with the intended action interval.
- Do not wait for streams while holding a tick. Use measured timing to detect hold timeouts.
- Return Gymnasium-style `terminated` for task outcomes and `truncated` for time limits. Connection, reset and tick failures are infrastructure errors, not successful or failed landings.

## 5. Safety, precision and failure

A safe landing requires all of the following:

- Touchdown speed, tilt and angular velocity within limits.
- Stable observation for 5 s after touchdown.
- The vessel survives and the relevant parts remain intact.

The numeric speed, tilt, angular-velocity and target-radius limits are **TBD** and MUST be frozen before comparison runs.

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
- Target coordinates, local frame and target radius.
- Descent altitude and initial position/velocity/attitude distributions.
- Control rate, episode horizon and touchdown-detection tolerance.
- Maximum touchdown speed, tilt and angular velocity.
- KSP interaction budget and per-group simulator budget.
- OOD perturbation ranges.
- Observation normalization and action scaling.

`initial_state_ready`, `success_thresholds_ready` and `reset_step_validated` stay `false` until the corresponding acceptance checks pass.
