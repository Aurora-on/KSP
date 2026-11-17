# Proposal interpretation

Source: `KSP_World_Model_Proposal_final.pptx`, nine slides, reviewed 2026-11-17. The deck binary is kept locally under `downloads/proposals/` (gitignored), and its SHA-256 is recorded in `proposal-source.json`. The earlier four-slide `开题1.pptx` and its SHA-256 remain recorded there as the superseded source.

This document records what the final proposal decides and what it leaves open. The normative task details are in [task-spec.md](task-spec.md), the controlled comparison is in [experiment-protocol.md](experiment-protocol.md), and the machine-readable subset is in `configs/mun_landing.toml`.

## Research questions

| Question | Proposal content | Environment implication |
| --- | --- | --- |
| RQ1, data efficiency | Does pretraining reduce the KSP interaction budget needed to reach a fixed safe-landing rate? | Common success criterion, evaluation initial states and explicit budget accounting |
| RQ2, imagined policy updates | Do imagined policy updates provide additional gains after transfer, using the same KSP dataset? | Freeze and hash one common dataset, preserve identical train/test splits, and compare against a no-imagination baseline |
| RQ3, physical generalization | How well does the agent generalize to mass, inertia, thrust and gravity changes without additional training? | Held-out physical conditions, with world model and policy frozen during evaluation |

## Task definition

The final proposal fixes the qualitative task; numeric choices remain open.

| Area | Decision |
| --- | --- |
| Scenario | Powered descent toward the Mun in KSP, with a fixed nominal lander, fixed nominal gravity and a flat target site |
| Initialization | Randomize initial position, velocity and attitude for all methods; exact distributions are TBD |
| Observation | Numeric telemetry: position and velocity relative to the target, attitude, angular velocity and remaining main-engine fuel |
| Action | RCS pitch/yaw/roll and main-engine throttle |
| Safety | Touchdown speed, tilt and angular velocity within limits, stable for 5 s; exact limits are TBD |
| Precision | Distance from the target in the surface tangent plane; precision rate uses all attempts as the denominator |
| Propellant | Total consumed propellant mass, with main-engine and RCS breakdowns on safe landings |
| Failure | Crashes, timeouts and fuel-exhaustion aborts count as failures; connection loss is an infrastructure error |
| Nominal physics | Main-engine fuel consumption changes mass normally; RCS reserves are ample and depletion has negligible dynamics effects |

## Technical approach

The proposal describes three stages:

1. **Simulator pretraining.** Build a fast landing simulator, randomize initial states, and train the world model and policy.
2. **Zero-shot evaluation.** Transfer the pretrained agent to KSP before adaptation and measure the initial transfer gap.
3. **KSP adaptation.** Collect a limited number of KSP interactions, update the world model, and improve the policy in imagination. Adaptation alternates replay of actual interactions with model imagination.

The reward combines a landing reward with crash, landing-error, touchdown-speed and fuel-use penalties; safe landing is the primary objective. Fuel regularization is calculated from actual propellant consumption, and landing error is measured in the surface tangent plane. Hyperparameters are tuned on validation scenarios and the final test scenarios are fixed separately. Dreamer is cited as a reference, not selected as a final algorithm.

## Controlled experiment

The final proposal defines a 2x2 comparison with matched observation, action, reward and model capacity:

| Group | Pretraining | Imagination |
| --- | --- | --- |
| A | No | No |
| B | No | Yes |
| C | Yes | No |
| D | Yes | Yes |

- A vs C and B vs D estimate the pretraining benefit.
- A vs B and C vs D estimate the imagination benefit.
- All groups use the same telemetry states, action space, rewards, model capacities, KSP interaction budgets, evaluation sets and number of policy updates.
- The learning-curve threshold is 80% success on validation tasks; the final result is reported on a fixed test set. This is distinct from the touchdown safety criterion.
- Record `N_sim`, `N_KSP`, policy and model updates, GPU hours and wall-clock time separately.
- Use at least three training seeds and separate adaptation, validation and test sets.

## Generalization evaluation

Test the frozen world model and policy under held-out mass, inertia, thrust and gravity changes without additional training. Change one parameter at a time over ranges fixed before testing, use initial states disjoint from training, and evaluate in both the fast simulator and KSP. Report success rate, landing error and propellant use relative to nominal conditions.

## Decisions still open

The final proposal does not fix the lander model, target coordinates, descent altitude, initial-state distributions, control rate, episode horizon, touchdown thresholds, target radius, KSP data budget, OOD perturbation ranges, world-model architecture or compute budget. The 80% learning-curve threshold and the minimum of three seeds are stated, but the exact seed count and per-group budgets remain open.

`configs/mun_landing.toml` deliberately leaves these numerical choices as `TBD` and keeps `initial_state_ready`, `success_thresholds_ready` and `reset_step_validated` false. This repository still does not implement the pretraining simulator, KSP landing `reset/step`, agent training or the RQ1-RQ3 experiments.
