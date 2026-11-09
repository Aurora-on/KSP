# Development order

## 1. Freeze the task

Choose the lander, target location, descent initial-state distribution, observation convention, control interface, frequency, touchdown safety limits and episode horizon. Fill the numerical omissions in `configs/mun_landing.toml` only after those decisions. Record actual craft parameters and snapshot hashes in a local asset manifest.

## 2. Validate the KSP adapter

Implement named-save reset, fresh state readback and bounded tick stepping. Check repeated resets against tolerances and actual UT increments against the intended action interval. Test episode termination, timeout truncation and connection failure separately. Keep a small held-out bank of Mun descent initial conditions.

Acceptance requires a controlled descent scenario, not merely an active vessel at the Kerbin launchpad. Infrastructure failures must be logged separately and handled consistently across methods.

## 3. Build the pretraining environment

Select or implement a rigid-body descent simulator with matching state/action definitions, variable mass, thrust, gravity and rotational dynamics. Compare one-step and short-rollout behavior with recorded KSP transitions before relying on transfer. A 3DOF prototype cannot answer inertia generalization.

## 4. Establish baselines and data

Use a scripted/classical controller as a reset and data-collection check, then choose the learning baseline and world-model implementation. Record separate training/evaluation KSP transition counts, simulator time and reset costs. Store data/checkpoints outside Git.

## 5. Run the proposal experiments

- RQ1: pretraining versus no pretraining, matched algorithm and evaluation settings, plot safe-landing rate against KSP training transitions.
- RQ2: freeze and hash one KSP dataset and its splits; compare imagined policy updates against the chosen no-imagination baseline with a stated compute budget.
- RQ3: freeze the final policy/model and normalization statistics, evaluate held-out mass/inertia/thrust/gravity conditions without adaptation. Report coupled parameter changes explicitly.

Use multiple seeds and report safety, precision and propellant individually. Dataset sizes, thresholds, seed count, model architecture and compute budget are still open; no results are claimed in this repository.
