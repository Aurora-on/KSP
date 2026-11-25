# Controlled experiment protocol

Status: design adopted from `KSP_World_Model_Proposal_final.pptx` (reviewed 2026-11-17). Numeric budgets, perturbation ranges and the exact seed count remain open. No experiment has been run in this repository.

This protocol defines the comparison. `docs/task-spec.md` defines the task itself, and `configs/mun_landing.toml` is the machine-readable declaration.

## 1. Research questions

- **RQ1, data efficiency:** does pretraining reduce the KSP interaction budget needed to reach a fixed safe-landing rate?
- **RQ2, imagined policy updates:** do imagined policy updates provide additional gains after transfer, using the same KSP dataset?
- **RQ3, physical generalization:** how well does the agent generalize to mass, inertia, thrust and gravity changes without additional training?

## 2. 2x2 design

| Group | Pretraining | Imagination | Purpose |
| --- | --- | --- | --- |
| A | No | No | KSP from scratch |
| B | No | Yes | Imagination without pretraining |
| C | Yes | No | Pretraining with actual replay |
| D | Yes | Yes | Full approach |

- A vs C and B vs D estimate the pretraining benefit.
- A vs B and C vs D estimate the imagination benefit.
- The comparison is not a hyperparameter search. Groups MUST use matched observation, action, reward, model capacity, number of policy updates, KSP interaction budget and evaluation sets.
- When imagination is disabled, the encoder and state-prediction module are still trained; Actor-Critic updates use only replayed actual interactions.
- When imagination is enabled, trajectories rolled out by the world model train the policy. The exact update rule MUST be documented as part of the experimental method.
- Pretraining means obtaining model and policy initializations under the same simulator budget for each method. Scratch groups use no simulator data.

## 3. Success threshold and checkpoints

- The learning-curve threshold is **80% success on validation tasks**.
- Checkpoints are selected only on validation tasks. The final performance is reported on a fixed test set that is never used for selection.
- A run that does not reach the threshold within the budget is recorded as not meeting it; it is not silently dropped.
- The 80% threshold is an experiment-selection criterion. It is not the touchdown safety criterion, which is defined separately in the task contract.

## 4. Data splits and seeds

- Keep adaptation, validation and test sets separate.
- Held-out evaluation initial states MUST be disjoint from training initial states.
- Use at least three training seeds. Report per-seed results and aggregate statistics; do not report only the best seed.
- Freeze and hash the KSP dataset and its splits before RQ2 comparisons. All RQ2 variants use the same data with identical splits.
- The exact per-group `N_sim`, `N_KSP`, seed count and compute budget are **TBD**.

## 5. Training and adaptation flow

```text
fast simulator pretraining
        |
        v
zero-shot KSP evaluation  ->  measure initial transfer gap
        |
        v
limited KSP adaptation    ->  collect calibration transitions
        |                     update world model
        |                     improve policy in imagination
        v
fixed test evaluation
```

The zero-shot result is reported before adaptation and is not mixed with adapted results. Adaptation uses the same observation and action fields as pretraining.

## 6. Metrics and reporting

Report separately, not as a single composite score:

- Safe-landing rate.
- Precision rate, using all attempts as the denominator.
- Landing error in the surface tangent plane, summarized over safe landings.
- Total propellant consumed, with main-engine and RCS breakdowns on safe landings.
- Failure counts for crashes, timeouts and fuel-exhaustion aborts.

Also record `N_sim`, `N_KSP`, policy updates, model updates, simulator steps, GPU hours and wall-clock time. Imagined transitions are never counted as KSP data.

Learning curves plot safe-landing rate against KSP training transitions. Log simulator steps, training time and compute cost separately so data efficiency is not confused with compute efficiency.

## 7. Generalization evaluation

Apply the following to the frozen world model and policy, without additional training:

- Perturb mass, inertia, thrust and gravity independently; change one parameter at a time.
- Fix the perturbation ranges before testing.
- Use initial states disjoint from training, adaptation and validation.
- Evaluate in both the fast simulator and KSP.
- Compare success rate, landing error and propellant use against nominal conditions.
- Report the performance-versus-perturbation curve, not only a single aggregate.

Coupled parameter changes may be reported as a separate stress test, but they do not replace the one-axis protocol. Every OOD variant must be validated and restored after use; gravity modification is an explicit experimental condition.

## 8. Failure and exclusion rules

- Crashes, timeouts and fuel-exhaustion aborts are task failures.
- Connection loss, reset failure and tick timeout are infrastructure errors and are logged separately.
- Do not exclude episodes after seeing their outcome.
- State the treatment of any infrastructure error before running the experiment; use the same treatment across groups.
- No results may be claimed without the recorded code revision, config, dataset hash and seed list.

## 9. Open decisions

- Exact `N_sim`, `N_KSP` and per-group budgets.
- Exact number of seeds beyond the minimum of three.
- World-model architecture, Dreamer version (if any), visual encoder (not required for numeric telemetry) and model capacity.
- OOD parameter ranges and the nominal values for mass, inertia, thrust and gravity.
- KSP site, vehicle calibration and measured event semantics. Target radius, touchdown limits, control rate and episode horizon were [frozen on 2026-11-25](nominal-task-decisions.md).
- GPU model and total compute budget.

Until these are fixed, this document describes the intended protocol rather than a runnable experiment.
