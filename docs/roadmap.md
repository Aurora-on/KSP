# Development order

The final proposal fixes the qualitative task and the 2x2 comparison. Numeric choices remain open. Do not start comparison runs until `docs/task-spec.md`, `docs/experiment-protocol.md` and `configs/mun_landing.toml` agree and no required numeric field is still `TBD`.

The standalone 6DOF simulator is now available in [`simulator/`](../simulator/README.md), including a scripted controller, repeatable resets, tests and evaluation tools. Its repository default uses the agreed 5 s stability window. Other simulator numbers are pilot assumptions; they do not resolve the shared KSP task's open parameters. See the [integration record](../simulator/docs/REPOSITORY_INTEGRATION.md) for remaining action, coordinate and metric differences.

## 1. Freeze the task

Choose the lander model and its mass/inertia/engine/RCS parameters, target coordinates, descent altitude, initial-state distribution, observation normalization, action scaling, control rate, episode horizon, touchdown thresholds and target radius. Fill the `TBD` values in `configs/mun_landing.toml` only after those decisions. Record actual craft parameters, the named save and snapshot hashes in a local asset manifest.

Acceptance: the task contract has no `TBD` in fields required to run; every numeric choice has a recorded decision; `initial_state_ready` and `success_thresholds_ready` can be set true only after the corresponding checks pass.

## 2. Validate the KSP adapter

Implement named-save reset, fresh state readback and bounded tick stepping. Check repeated resets against tolerances and actual UT increments against the intended action interval. Test episode termination, timeout truncation and connection failure separately. Keep a small held-out bank of Mun descent initial conditions.

The importable adapter skeleton is in `ksp_env/` (see [`ksp-env.md`](ksp-env.md)); diagnostics remain in `scripts/`. Connection and read-only diagnostics are validated. Observation, reset and step are gated by the unresolved target/frame/fuel/RCS decisions and are not yet validated.

Acceptance requires a controlled descent scenario, not merely an active vessel at the Kerbin launchpad. A scripted controller must complete a landing under the frozen task contract before learning experiments begin. Infrastructure failures must be logged separately and handled consistently across methods.

## 3. Align the available pretraining simulator

The implemented simulator provides variable-mass 6DOF rigid-body descent, main-engine thrust, gravity, RCS torque, fuel accounting and live post-contact stability dynamics. Independent acceptance and its limits are recorded in [`simulator/docs/VALIDATION.md`](../simulator/docs/VALIDATION.md). Run its local tests and scripted baseline while the KSP adapter is developed.

Next, calibrate craft parameters, coordinate transformations, action axes/scales and control timing against the KSP adapter. The simulator's native action order is throttle first; the shared task lists RCS commands first, with unresolved axis semantics. Resolve that mapping explicitly, then compare one-step and short-rollout behavior with recorded KSP transitions before relying on transfer.

Acceptance: the simulator and KSP adapter expose the same observation/action fields and timing semantics; one-step and short-rollout error thresholds are fixed before comparison; a held-out transition bank is available for validation.

## 4. Establish baselines and data

Use a scripted/classical controller as a reset and data-collection check, then implement the 2x2 experiment harness (A: scratch, B: imagination, C: pretrain + adapt, D: full). Record separate training/evaluation KSP transition counts, simulator time, reset costs, policy/model updates, GPU hours and wall-clock time. Freeze and hash the dataset and its adaptation/validation/test splits; store data and checkpoints outside Git.

Acceptance: all four groups use matched observation, action, reward, model capacity, policy updates, KSP budget and evaluation sets; a no-imagination run still trains the encoder and state-prediction module; the evaluation harness reports the metrics and denominators defined in the task contract.

## 5. Run the proposal experiments

- **RQ1:** compare pretraining versus no pretraining (A vs C, B vs D) with matched settings; plot safe-landing rate against KSP training transitions.
- **RQ2:** compare imagination versus no imagination (A vs B, C vs D) using the same frozen KSP dataset and splits; document the exact imagination update rule.
- **RQ3:** freeze the final policy/model and normalization statistics; evaluate held-out mass, inertia, thrust and gravity one axis at a time, in both the fast simulator and KSP, without adaptation.

Use at least three seeds and report safety, precision and propellant individually. Select checkpoints only on validation tasks at the 80% success threshold and report final numbers on the fixed test set. Runs that do not reach the threshold are recorded, not dropped.

## Deliverables

| Workstream | Deliverable |
| --- | --- |
| Simulator | Implemented fast 6DOF landing simulator; KSP parameter and interface alignment pending |
| KSP integration | Telemetry, control and repeatable resets, plus a held-out transition bank |
| World model and policy | Pretrained and adapted landing agent |
| Evaluation and analysis | Learning curves, trajectories, performance comparison and generalization results |

The standalone simulator has passed its source-domain engineering checks. Milestones M1 and M2 add simulator demonstration collection, preprocessing, replay, offline world-model fitting and open-loop evaluation. The M2R CPU fit passed the near-ground 15-step gate and fell short on posterior reconstruction, open-loop dynamics and both terminal-reward gates. S1 fits a deterministic residual model and evaluates open-loop dynamics, event timing and expected returns. The selected M2R checkpoint provides the RSSM comparison. See the [validation record](validation.md) for measured gates. KSP landing integration, policy training and RQ1–RQ3 comparison runs remain pending. Dataset sizes, physical thresholds, exact training seed count (at least three), model architecture and compute budget remain open. Source-domain scripted results do not establish KSP transfer or learning performance; keep existing task/protocol decision records and dataset split commitments unchanged until the team records an explicit revision.

