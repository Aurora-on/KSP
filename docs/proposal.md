# Proposal interpretation

Source: `开题1.pptx`, four slides, reviewed 2026-11-09. The original slides remain outside this repository; their SHA-256 is recorded in `proposal-source.json`.

## Requirements stated in the slides

| Slide | Proposal content | Environment implication |
| --- | --- | --- |
| 1 | World Models for Lunar Landing in Kerbal Space Program | World-model research using KSP as the target environment |
| 2 | Powered descent toward the Mun; safe, precise landing with efficient propellant use | A Mun descent scenario and separate safety, position-error and fuel metrics |
| 3 | Limited KSP interaction budget; simulator mismatch; pretrain then adapt; imagined trajectories | A pretraining simulator, a KSP adapter and explicit interaction accounting |
| 4, RQ1 | Less KSP data to reach a fixed safe-landing rate | Common success criterion, evaluation initial states and budget definition |
| 4, RQ2 | Additional gains from imagined policy updates using the same KSP dataset | Freeze and hash the common dataset, preserve identical train/test splits |
| 4, RQ3 | Generalization to mass, inertia, thrust and gravity changes without further training | Held-out physical conditions, with model and policy frozen during evaluation |

## Implementation recommendations

Start with numeric states and a continuous-action rigid-body descent simulator. Numeric observation is a proposed starting point; the slides do not require images. Because RQ3 includes inertia, the complete pretraining task must include rotational dynamics, attitude and angular velocity. A point-mass prototype can support early checks but cannot establish inertia generalization.

Use a common Gymnasium-style API, coordinate definitions, SI units and action duration across the pretraining and KSP environments. Fix a lander, initial-descent snapshots and evaluation targets. Treat imagined transitions separately from real KSP transitions.

## Decisions still open

The proposal does not choose a pretraining simulator, world-model algorithm, Dreamer version, visual encoder, specific lander, descent altitude/velocity distribution, control rate, success thresholds, KSP data budget, seed count, OOD parameter ranges or GPU model. `configs/mun_landing.toml` is a draft and deliberately leaves these numerical choices unset.

This initial repository prepares dependencies, diagnostics and experimental requirements. It does not implement the pretraining simulator, KSP landing `reset/step`, agent training or the RQ1–RQ3 experiments.
