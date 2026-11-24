# Validation record

## 2026-11-24 — M2R simulator world-model quality

The selected model fell short of the M2R criteria on 50 final held-out simulator episodes. Only G3 passed. These measurements evaluate model predictions conditioned on recorded actions.

### Simulator collections

Nominal `SimConfig()` generated all starts. Sampling targeted 60% hard-landing and 5% random-action episodes, with counts rounded to whole episodes. The other episodes cycled through scripted action noise at standard deviations 0, 0.05, 0.1 and 0.2. Hard landings capped throttle near the surface and applied a short RCS pulse. The three collections have disjoint reset seeds. Each loading path checked the saved manifest and replay.

| Collection | Seed | Episodes | Transitions | Safe | Crash | Timeout | Safe hard landings | Hard episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 170 | 200 | 59362 | 79 | 120 | 1 | 10 | 120 |
| dev | 180 | 50 | 14846 | 19 | 31 | 0 | 2 | 30 |
| heldout | 190 | 50 | 14332 | 16 | 33 | 1 | 0 | 30 |

First-touchdown coverage for hard episodes, with counts in successive left-closed, right-open bins. Speed bins start at 0, 1, 2, 3, 4, 5 and 6 m/s. Tilt bins start at 0, 5, 10, 15 and 20°. The final bin in each series has no upper bound.

| Collection | Speed bin counts | 1.5–2 / 2–2.5 m/s | Tilt bin counts |
| --- | --- | --- | --- |
| train | 0 / 21 / 26 / 15 / 29 / 21 / 8 | 9 / 10 | 28 / 15 / 24 / 26 / 27 |
| dev | 1 / 5 / 6 / 4 / 4 / 6 / 4 | 1 / 3 | 7 / 8 / 6 / 4 / 5 |
| heldout | 0 / 1 / 11 / 5 / 5 / 6 / 2 | 1 / 9 | 5 / 4 / 10 / 3 / 8 |

| Collection | Manifest SHA-256 | Replay SHA-256 |
| --- | --- | --- |
| train | `43afb9b859cdf5ad67d298610b7d9ab437bf3a30fba25cb849c171c745b2ba8d` | `a5a3848297831da0fc285aec45489d10c1aeb1f762d3c291d915bb64b7a479ce` |
| dev | `3ffb03d0258e7a1c6ba4c33cb969e838705bc1bfe5a6f0fad574fb6d59ce5ed1` | `41a313d1c847e5f4564df16d4fac93ce00acba46504c24dd6a99f7bc31cdd79f` |
| heldout | `ba1921414c2b258add7bfee2a4e8da2a14c6e04951cb344d072fb450323cecdb` | `01c5f052fb2050bc4b30a3c2168de488a6fea0d56ab9aa550483899280d7075e` |

### Fits and model selection

The fits used Apple M5 CPU, seed 31, Adam at 0.0003 and 16 sequences of 32 transitions per update. Observations used 20 m, 5 m/s and 0.2 rad/s scales. Each reconstruction loss sums squared errors across 16 features before averaging over samples. Rewards use two-hot interpolation and expectation in original reward units.

| Fit | Source | Seed | Factors × classes | KL dyn / rep | End loss weight | Ground history | Update budget | Updates logged | Fit time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short A | `e9ce3da58b5d` (dirty) | 31 | 16×16 | 1.0 / 0.1 | 1.0 | no | 2000 | 2000 | 85.3 |
| Short B (exploratory) | `e9ce3da58b5d` (dirty) | 31 | 32×32 | 0.5 / 0.03 | 40 | yes | 2000 | 2000 | 151.9 |
| Short C | `97df4f5cbc7e` (clean) | 31 | 32×32 | 0.5 / 0.03 | 1.0 | yes | 2000 | 2000 | 144.1 |
| Long A | `eb95aa9d4f88` (clean) | 31 | 16×16 | 1.0 / 0.1 | 1.0 | no | 300000 | 204325 | 7140.0 |
| Long B (exploratory) | `e9ce3da58b5d` (dirty) | 31 | 32×32 | 0.5 / 0.03 | 40 | yes | 180000 | 7947 | 561.0 (last log) |

**Protocol correction.** Short B and the initial long B increased the reward and continuation loss weights on terminal transitions to 40. This change was outside the permitted adjustments in section 4 of the rework brief. The initial long B was stopped, and the weighted B results are retained as exploratory measurements. Formal model selection compared short A with short C. Short C uses B's larger categorical latent, lower KL weights and observation-derived ground-history input at terminal weight 1. All official long-fit candidates use terminal weight 1.

Scoring times for the short fits (1-, 5- and 15-step diagnostics) were Short A train 13.58 s, Short A dev 3.68 s, Short B (exploratory) train 16.77 s, Short B (exploratory) dev 5.09 s, Short C train 16.04 s, Short C dev 4.00 s.

The official long fit `m2r-long-A` used seed 31, a budget of 300,000 updates or 7,140 s, and snapshots every 4,000 updates.

Development gate count ranked checkpoints first, then worst and mean threshold ratios, then the earliest update. The selected checkpoint was `agent/runs/m2r-long-A/step-028000.pt` at update 28,000, from 53 scored candidates. SHA-256 `e761885ecebc424e8c16c2877a0cc69f187c83683e6ca6ee17d89f89926261f3`.

The safe-reward group had zero hits at every official checkpoint, making the worst and mean gate ratios infinite. Five snapshots passed G3 (28k, 36k, 40k, 60k and 68k). All other snapshots passed zero gates. The earliest-update tie-break selected 28k. The final update had development posterior errors of 3.988 m, 0.708 m/s and 2.548° and passed zero gates.

### Gate results

G1 gives posterior position (m), velocity (m/s) and tilt (degrees). G2 lists position and velocity at each horizon. G3 lists near-ground 15-step velocity and tilt. G4 counts passing safe and crash episodes separately. G5 counts episodes whose predicted 15-step reward sum has the true sign and at least half its magnitude. G1 limits are 0.5 m, 0.2 m/s and 1°. G3 limits are 0.5 m/s and 2°. G4 requires at least 80% in each group and G5 requires 70% overall. Relative G2 limits for each collection appear in the next table.

| Fit, collection | Episodes | Passing gates | G1 pos / vel / tilt | G2 pos / vel (5) | G2 pos / vel (15) | G3 vel / tilt | G4 safe / crash hits | G5 hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short A, train | 200 | None | 3.23 / 0.75 / 2.89 | 6.18 / 1.21 | 10.16 / 2.06 | 1.03 / 2.35 | 0 of 79 / 6 of 120 | 77 of 200 |
| Short A, dev | 50 | None | 3.42 / 0.78 / 3.17 | 6.37 / 1.25 | 10.53 / 2.13 | 1.05 / 2.55 | 0 of 19 / 1 of 31 | 18 of 50 |
| Short B (exploratory), train | 200 | G4, G5 | 3.79 / 0.63 / 2.81 | 4.53 / 0.94 | 7.72 / 1.64 | 1.04 / 3.03 | 79 of 79 / 119 of 120 | 165 of 200 |
| Short B (exploratory), dev | 50 | G4, G5 | 3.72 / 0.72 / 3.30 | 4.45 / 1.02 | 7.77 / 1.71 | 1.04 / 2.50 | 19 of 19 / 29 of 31 | 37 of 50 |
| Short C, train | 200 | None | 6.26 / 0.73 / 2.81 | 9.48 / 0.87 | 12.85 / 1.16 | 0.73 / 1.97 | 0 of 79 / 21 of 120 | 56 of 200 |
| Short C, dev | 50 | None | 6.49 / 0.78 / 3.01 | 9.59 / 0.95 | 12.79 / 1.24 | 0.73 / 2.13 | 0 of 19 / 2 of 31 | 14 of 50 |
| Selected m2r-long-A, train | 200 | None | 2.81 / 0.39 / 1.41 | 4.52 / 0.49 | 7.21 / 0.71 | 0.505 / 2.100 | 0 of 79 / 38 of 120 | 5 of 200 |
| Selected m2r-long-A, dev | 50 | G3 | 3.09 / 0.46 / 1.99 | 4.79 / 0.56 | 7.57 / 0.80 | 0.47 / 1.88 | 0 of 19 / 11 of 31 | 4 of 50 |
| Selected m2r-long-A, heldout | 50 | G3 | 2.98 / 0.47 / 1.75 | 4.71 / 0.57 | 7.67 / 0.79 | 0.49 / 1.98 | 0 of 16 / 11 of 33 | 2 of 50 |

| Collection | G2 limits, 5 steps (m / m/s) | G2 limits, 15 steps (m / m/s) |
| --- | --- | --- |
| train | 1.823 / 0.214 | 5.429 / 0.583 |
| dev | 1.778 / 0.217 | 5.285 / 0.596 |
| heldout | 1.841 / 0.220 | 5.472 / 0.598 |

### One-step prediction and baselines

| Collection | Predictor | Endpoints | Position (m) | Velocity (m/s) | Tilt (°) | Angular velocity (rad/s) | Fuel (kg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | Model | 59362 | 3.231 | 0.411 | 1.593 | 0.020 | 6.696 |
| train | Persistence | 59362 | 0.365 | 0.050 | 0.221 | 0.010 | 0.093 |
| train | Train mean | 59362 | 31.708 | 2.584 | 5.748 | 0.054 | 9.089 |
| dev | Model | 14846 | 3.509 | 0.481 | 2.196 | 0.025 | 7.320 |
| dev | Persistence | 14846 | 0.357 | 0.051 | 0.223 | 0.011 | 0.093 |
| dev | Train mean | 14846 | 31.309 | 2.609 | 6.264 | 0.055 | 9.223 |
| heldout | Model | 14332 | 3.385 | 0.492 | 1.927 | 0.025 | 6.781 |
| heldout | Persistence | 14332 | 0.369 | 0.051 | 0.219 | 0.011 | 0.095 |
| heldout | Train mean | 14332 | 30.417 | 2.535 | 5.946 | 0.054 | 9.015 |

Selected-checkpoint CLI wall times were train 25.58 s, dev 10.60 s, heldout 10.36 s.

The independent episode-by-episode calculation covered 19 published gate components in 2.87 s. The maximum absolute discrepancy across the 13 continuous components was 0.000321 in the units of each component. The maximum count discrepancy was 0. The largest absolute differences in terminal posterior reward and final-15 predicted sum were 0.00168 and 9.93e-05.

### Terminal predictions on the independent episodes

At the selected checkpoint, all 16 safe endings received posterior predicted rewards between −0.012 and +0.034. The 33 crash predictions ranged from −98.36 to −2.13. At one step the reward MAE was 0.336 overall versus 0.373 for a zero predictor. On the 50 actual endings it was 70.58 versus 99.01. The one-step continuation medians were 0.900 on terminal and 1.000 on nonterminal steps. At the 0.5 cutoff the model detected 15 of the 50 endings and generated eight false positives among 14,282 nonterminal steps. At 15 steps it detected two endings and generated 37 false positives among 13,582 nonterminal steps.

| Horizon | Endpoints | Position (m) | Velocity (m/s) | Tilt (°) | Angular velocity (rad/s) | Fuel (kg) | Reward MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14,332 | 3.3846 | 0.4920 | 1.9268 | 0.0249 | 6.7810 | 0.3357 |
| 5 | 14,132 | 4.7127 | 0.5665 | 2.4181 | 0.0338 | 6.0821 | 0.3782 |
| 15 | 13,632 | 7.6654 | 0.7936 | 3.6791 | 0.0521 | 6.3997 | 0.5571 |
| 50 | 11,882 | 14.0183 | 1.6093 | 8.2979 | 0.0763 | 9.4722 | 0.9177 |

The full JSON includes the near-ground subset at each horizon, the per-episode true and predicted rewards, the last-15 returns, continuation quantiles and threshold confusion counts. The 16 held-out safe landings all came from the regular scripted controller. Performance on safe episodes under the hard perturbation remains unmeasured on this split.

### Interpretation and reproduction

The selected model missed the posterior position target on both training and held-out episodes (2.813 and 2.978 m). The held-out loss of accuracy is already present when decoding an observed frame. Open-loop rollout adds further position and velocity error. On training and development only, restarting the posterior from the true observation every 32 steps reduced position error from 2.813 to 2.682 m and from 3.093 to 2.931 m, respectively, on the same endpoints. Matching the training-window length recovered a small part of the gap to 0.5 m.

After 204,325 updates the long fit had exhausted its time budget. Its fixed training-probe reconstruction loss fell from 5.807 to 0.0155 while no development checkpoint passed G1 or G2. The curves, full-history training errors and restart diagnostic support underfitting of the physical representation and dynamics in this configuration. They do not isolate the effect of categorical precision, KL regularization or the decoder. The 32-step training window also covers only 3.2 s of the 5 s safe-landing dwell time. Short C encoded the observed ground history and still missed all safe rewards at 2,000 updates. At that budget, B and C differed only in terminal loss weighting and produced very different terminal predictions. The weighted comparison is exploratory and will require a separate calibration study if adopted.

M2R failed the five-gate acceptance requirement. A controlled comparison within the same CPU or L40 budget could distinguish latent precision and KL regularization effects before extending the same training configuration. The project lead decides whether to change latent form, authorize more compute or revisit event weighting. Policy training awaits that decision.

All fits ran on an Apple M5 CPU with macOS arm64, Python 3.14.4, PyTorch 2.15.0, NumPy 2.5.4 and Gymnasium 1.4.0. The local agent suite passed 44 tests (12 more than M2), the simulator suite passed 61 and skipped one version-specific manifest test, and the offline root suite passed 26 and skipped two opt-in live checks. The simulator emitted its two existing unbounded-observation warnings. Commands from the repository root:

```sh
python -m compileall -q ksp_env scripts tests agent/src agent/scripts
python -m unittest discover -s tests -v
python -m pytest simulator/tests -q
python -m pytest agent/tests -q
```

The [agent README](../agent/README.md) gives the exact collection, fit, checkpoint scoring and plot commands. [Method and selection rules](../agent/docs/METHOD.md) document the model variants. Local artifacts are under `agent/runs/m2r-study/` (`selection.json`, `selected-{train,dev,heldout}.json`, `independent-gate-check.json` and `quality-curve.{svg,png}`). Datasets, checkpoints and generated reports are kept outside version control.

### Error across updates

Development checkpoint scores use all 50 diagnostic episodes. The table records the evaluation series used to plot the local quality curves.

| Fit | Update | Posterior pos (m) | 15-step pos / vel | Near-ground 15-step vel | G4 safe rate | G4 crash rate | G5 rate | Passing gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short A | 0 | 40.43 | 37.79 / 4.72 | 3.11 | 0.000 | 0.000 | 0.000 | None |
| Short A | 1000 | 8.05 | 27.40 / 1.59 | 1.20 | 0.000 | 0.000 | 0.000 | None |
| Short A | 2000 | 3.42 | 10.53 / 2.13 | 1.05 | 0.000 | 0.032 | 0.360 | None |
| Short B (exploratory) | 0 | 40.67 | 39.28 / 7.35 | 5.37 | 0.000 | 0.000 | 0.000 | None |
| Short B (exploratory) | 1000 | 6.31 | 12.13 / 1.26 | 0.84 | 0.158 | 0.903 | 0.420 | None |
| Short B (exploratory) | 2000 | 3.72 | 7.77 / 1.71 | 1.04 | 1.000 | 0.935 | 0.740 | G4, G5 |
| Short C | 0 | 40.67 | 39.28 / 7.35 | 5.37 | 0.000 | 0.000 | 0.000 | None |
| Short C | 1000 | 6.88 | 14.61 / 1.46 | 1.21 | 0.000 | 0.000 | 0.000 | None |
| Short C | 2000 | 6.49 | 12.79 / 1.24 | 0.73 | 0.000 | 0.065 | 0.280 | None |
| Short C | 2000 | 6.49 | 12.79 / 1.24 | 0.73 | 0.000 | 0.065 | 0.280 | None |
| Long A | 0 | 40.43 | 37.79 / 4.72 | 3.11 | 0.000 | 0.000 | 0.000 | None |
| Long A | 4000 | 3.97 | 6.23 / 1.25 | 0.61 | 0.000 | 0.355 | 0.340 | None |
| Long A | 8000 | 3.93 | 8.51 / 1.23 | 1.13 | 0.000 | 0.097 | 0.040 | None |
| Long A | 12000 | 3.95 | 8.59 / 1.02 | 0.73 | 0.000 | 0.516 | 0.440 | None |
| Long A | 16000 | 3.54 | 10.56 / 1.09 | 0.89 | 0.000 | 0.452 | 0.320 | None |
| Long A | 20000 | 3.35 | 9.64 / 0.99 | 0.70 | 0.000 | 0.677 | 0.300 | None |
| Long A | 24000 | 3.23 | 7.51 / 0.93 | 0.42 | 0.000 | 0.323 | 0.080 | None |
| Long A | 28000 | 3.09 | 7.57 / 0.80 | 0.47 | 0.000 | 0.355 | 0.080 | G3 |
| Long A | 32000 | 3.34 | 10.82 / 1.14 | 0.59 | 0.000 | 0.516 | 0.060 | None |
| Long A | 36000 | 2.94 | 8.41 / 0.90 | 0.48 | 0.000 | 0.613 | 0.080 | G3 |
| Long A | 40000 | 2.96 | 8.20 / 0.77 | 0.36 | 0.000 | 0.484 | 0.080 | G3 |
| Long A | 44000 | 3.37 | 7.61 / 0.73 | 0.51 | 0.000 | 0.484 | 0.080 | None |
| Long A | 48000 | 3.92 | 8.55 / 0.89 | 0.68 | 0.000 | 0.581 | 0.060 | None |
| Long A | 52000 | 3.50 | 7.71 / 0.86 | 0.65 | 0.000 | 0.645 | 0.080 | None |
| Long A | 56000 | 3.69 | 8.57 / 0.88 | 0.73 | 0.000 | 0.548 | 0.060 | None |
| Long A | 60000 | 2.74 | 6.43 / 0.75 | 0.49 | 0.000 | 0.613 | 0.060 | G3 |
| Long A | 64000 | 4.13 | 7.47 / 0.76 | 0.59 | 0.000 | 0.355 | 0.060 | None |
| Long A | 68000 | 3.46 | 6.93 / 0.66 | 0.41 | 0.000 | 0.645 | 0.100 | G3 |
| Long A | 72000 | 3.72 | 7.47 / 0.87 | 0.73 | 0.000 | 0.484 | 0.040 | None |
| Long A | 76000 | 3.70 | 8.25 / 0.83 | 0.67 | 0.000 | 0.387 | 0.040 | None |
| Long A | 80000 | 3.08 | 6.70 / 0.75 | 0.56 | 0.000 | 0.548 | 0.040 | None |
| Long A | 84000 | 4.14 | 9.73 / 0.94 | 0.81 | 0.000 | 0.355 | 0.040 | None |
| Long A | 88000 | 2.98 | 6.46 / 0.76 | 0.62 | 0.000 | 0.581 | 0.020 | None |
| Long A | 92000 | 3.46 | 7.20 / 0.76 | 0.68 | 0.000 | 0.516 | 0.100 | None |
| Long A | 96000 | 3.72 | 9.27 / 0.87 | 0.60 | 0.000 | 0.226 | 0.080 | None |
| Long A | 100000 | 3.67 | 8.06 / 0.92 | 0.82 | 0.000 | 0.226 | 0.020 | None |
| Long A | 104000 | 4.14 | 8.33 / 0.97 | 1.01 | 0.000 | 0.484 | 0.060 | None |
| Long A | 108000 | 3.60 | 7.52 / 0.93 | 0.94 | 0.000 | 0.516 | 0.040 | None |
| Long A | 112000 | 3.69 | 8.56 / 0.89 | 0.93 | 0.000 | 0.516 | 0.060 | None |
| Long A | 116000 | 3.98 | 6.33 / 0.78 | 0.77 | 0.000 | 0.258 | 0.060 | None |
| Long A | 120000 | 4.23 | 8.24 / 0.82 | 0.82 | 0.000 | 0.323 | 0.060 | None |
| Long A | 124000 | 3.76 | 8.20 / 0.96 | 0.96 | 0.000 | 0.452 | 0.040 | None |
| Long A | 128000 | 4.57 | 8.97 / 1.03 | 0.98 | 0.000 | 0.484 | 0.040 | None |
| Long A | 132000 | 4.89 | 12.09 / 0.98 | 0.93 | 0.000 | 0.419 | 0.040 | None |
| Long A | 136000 | 3.77 | 8.14 / 0.93 | 0.94 | 0.000 | 0.323 | 0.040 | None |
| Long A | 140000 | 3.46 | 6.84 / 0.82 | 0.78 | 0.000 | 0.355 | 0.060 | None |
| Long A | 144000 | 3.65 | 6.56 / 0.79 | 0.74 | 0.000 | 0.516 | 0.040 | None |
| Long A | 148000 | 4.01 | 7.52 / 0.90 | 0.91 | 0.000 | 0.581 | 0.040 | None |
| Long A | 152000 | 5.08 | 9.16 / 0.97 | 1.00 | 0.000 | 0.290 | 0.020 | None |
| Long A | 156000 | 4.46 | 7.93 / 0.93 | 0.94 | 0.000 | 0.484 | 0.020 | None |
| Long A | 160000 | 4.43 | 7.96 / 0.97 | 1.04 | 0.000 | 0.387 | 0.080 | None |
| Long A | 164000 | 3.95 | 8.05 / 1.04 | 1.02 | 0.000 | 0.387 | 0.080 | None |
| Long A | 168000 | 3.94 | 7.03 / 0.92 | 0.94 | 0.000 | 0.323 | 0.040 | None |
| Long A | 172000 | 3.95 | 8.08 / 0.95 | 0.90 | 0.000 | 0.516 | 0.100 | None |
| Long A | 176000 | 4.12 | 6.78 / 0.93 | 1.01 | 0.000 | 0.323 | 0.020 | None |
| Long A | 180000 | 3.76 | 7.12 / 0.84 | 0.79 | 0.000 | 0.323 | 0.040 | None |
| Long A | 184000 | 3.86 | 8.56 / 0.89 | 0.85 | 0.000 | 0.355 | 0.020 | None |
| Long A | 188000 | 5.12 | 9.11 / 1.14 | 1.25 | 0.000 | 0.258 | 0.060 | None |
| Long A | 192000 | 4.71 | 8.71 / 1.10 | 1.22 | 0.000 | 0.161 | 0.020 | None |
| Long A | 196000 | 4.35 | 7.28 / 0.91 | 0.99 | 0.000 | 0.387 | 0.020 | None |
| Long A | 200000 | 3.89 | 7.13 / 0.91 | 0.93 | 0.000 | 0.323 | 0.080 | None |
| Long A | 204000 | 4.42 | 8.81 / 0.99 | 1.06 | 0.000 | 0.290 | 0.040 | None |
| Long A | 204325 | 3.99 | 8.10 / 0.95 | 1.04 | 0.000 | 0.290 | 0.080 | None |
| Long B (exploratory) | 0 | 40.67 | 39.28 / 7.35 | 5.37 | 0.000 | 0.000 | 0.000 | None |
| Long B (exploratory) | 4000 | 4.88 | 6.61 / 0.89 | 0.69 | 1.000 | 0.871 | 0.600 | G4 |

## 2026-11-24 — M1 replay and M2 pipeline checks

Apple M5 CPU, macOS arm64, Python 3.14.4, PyTorch 2.15.0, NumPy 2.5.4 and Gymnasium 1.4.0. These runs cover offline model fitting and simulator-only prediction. Landing-policy evaluation starts in a later milestone.

### M1 collection and replay

M1 source `4e4cf4e`. Both collections used nominal `SimConfig()` and `agent/configs/debug.json`. Each used seven scripted episodes (native action noise standard deviations 0, 0.05, 0.1 and 0.2) and one uniform-random episode. Both come from the simulator train split, using disjoint reset seeds. The real trajectory arrays (observations, applied actions, rewards and both end flags) matched the stored NPZ arrays element for element when collected again. The fixed validation and test sets were reserved for later policy assessment.

| Collection | Root seed | Episodes | Transitions | Manifest SHA-256 | Collection wall time |
| --- | ---: | ---: | ---: | --- | ---: |
| Training | 17 | 8 | 2633 | `890a1e09e8f0bf619feec949c8a3c3b614aadd3285d9902ad76b707e84eb1abd` | 1.62 s |
| Held out | 18 | 8 | 2673 | `7a665e436f46ecb240b69446fe47c6e3a2d5b6dd06b11d752de19759908492e1` | 1.23 s |

### M2 pipeline passed; model quality fell short (see M2R)

M2 source `f40f6de`, clean at the start of both runs. Each preset was fitted once on the same eight training episodes using seed 31. The loss was evaluated before and after fitting using identical real transitions from the training replay, with the posterior set to its mode. The probe depends on batch size and sequence length, so each loss pair is interpreted within its own preset.

| Preset | Updates | Probe transitions (terminal) | Loss before | Loss after | Fit command wall time | Report command wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| debug | 120 | 128 (0) | 7.5179 | 1.6097 | 2.74 s | 1.25 s |
| base | 1000 | 866 (3) | 8.2536 | 2.4501 | 73.28 s | 2.03 s |

Each origin incorporates observations up to that frame. The RSSM then predicts using only recorded future actions. Every eligible origin in the held-out collection contributes. Vector errors are mean Euclidean distances at the endpoint. Tilt, fuel and reward use mean absolute errors. Terminal labels combine termination and truncation at the endpoint. Counts are given for both rates because each horizon has only eight actual endings.

| Preset | Horizon | Endpoints | Position (m) | Velocity (m/s) | Tilt (deg) | Angular velocity (rad/s) | Fuel (kg) | Reward | Terminal recall (TP/actual) | False positives (FP/actual negatives) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| debug | 1 | 2673 | 31.278 | 3.022 | 12.269 | 0.0677 | 58.540 | 0.3173 | 0/8 | 0/2665 |
| debug | 5 | 2641 | 30.357 | 2.943 | 7.897 | 0.0690 | 18.221 | 0.3205 | 0/8 | 0/2633 |
| debug | 15 | 2561 | 29.105 | 2.905 | 7.897 | 0.0647 | 14.873 | 0.3296 | 0/8 | 0/2553 |
| debug | 50 | 2281 | 24.279 | 2.793 | 6.601 | 0.0635 | 17.537 | 0.3663 | 0/8 | 0/2273 |
| base | 1 | 2673 | 23.166 | 2.412 | 8.557 | 0.0954 | 23.064 | 0.3309 | 0/8 | 0/2665 |
| base | 5 | 2641 | 24.609 | 2.459 | 8.840 | 0.0936 | 24.270 | 0.3374 | 0/8 | 0/2633 |
| base | 15 | 2561 | 25.503 | 2.524 | 9.112 | 0.0875 | 27.482 | 0.3503 | 0/8 | 0/2553 |
| base | 50 | 2281 | 23.972 | 2.500 | 7.845 | 0.0926 | 31.108 | 0.3947 | 0/8 | 0/2273 |

Both presets missed all eight real endings at every horizon and produced zero false positives. Position error at one step exceeded 23 m in both runs. The loss decreased on the training probes, while the held-out trajectories show substantial physical prediction error.

The debug CPU smoke took 6.84 s across the two collection commands, fitting and the report, including separate interpreter startups. Longer horizons exclude origins too near the episode end. Aggregate errors at different horizons therefore use different sets of origins.

From the repository root with the editable simulator and agent installed:

```sh
mun-agent collect --config agent/configs/debug.json --seed 17 --device cpu --out agent/data/m1-train
mun-agent collect --config agent/configs/debug.json --seed 18 --device cpu --out agent/data/m2-heldout
for preset in debug base; do
  mun-agent train --config "agent/configs/${preset}.json" --seed 31 --device cpu \
    --data agent/data/m1-train --out "agent/runs/m2-${preset}"
  mun-agent wm-report --config "agent/configs/${preset}.json" --device cpu \
    --data agent/data/m2-heldout --checkpoint "agent/runs/m2-${preset}/world_model.pt" \
    --out "agent/runs/m2-${preset}/wm-report.json"
done
```

Offline syntax and regression checks passed. The initial M2 acceptance criteria covered pipeline operation and loss reduction. The M2R entry assesses model quality. At M1, `python -m pytest agent/tests -q` passed 21 cases (20 added). At M2, it passed 32 (11 additional). For each milestone, `python3 -m compileall -q ksp_env scripts tests` succeeded, `python3 -m unittest discover -s tests -v` passed 26 and skipped two opt-in live checks, and `python -m pytest simulator/tests -q` passed 61 and skipped the NumPy 2.4.6 manifest reproduction test. The simulator tests emitted the two existing unbounded-observation warnings.

## 2026-11-24 — M0 agent packaging and CPU checks

Tested the M0 source at `5133edf` (base `e6a67b9`) on an Apple M5, macOS arm64.
Each Python version used a fresh virtual environment and the release pins for
PyTorch 2.15.0, NumPy 2.5.4, Gymnasium 1.4.0 and pytest 9.2.0.
The following commands ran from the repository root. Times are elapsed seconds.

| Command | Python 3.12.14 | Python 3.14.4 |
| --- | --- | --- |
| `python -m pip install -c requirements/runtime.txt -e ./simulator -e './agent[dev]'` | Installed, 29.19 s | Installed, 30.11 s |
| `python -m pip check` | Passed, 0.23 s | Passed, 0.26 s |
| `python -m pytest agent/tests -q` | 1 passed, 20.37 s | 1 passed, 13.24 s |

On Python 3.14.4, `python3 -m compileall -q ksp_env scripts tests` succeeded
(0.08 s). In an environment with the kRPC client installed,
`python3 -m unittest discover -s tests -v` passed 26 tests and skipped both
opt-in live tests (0.18 s). In the clean agent environment,
`python -m pytest simulator/tests -q -rs` passed 61 and skipped one (4.83 s).
The manifest reproduction test requires NumPy 2.4.6. Gymnasium emitted two
warnings about unbounded observations.

These checks cover simulator-side installation, a CPU tensor operation and
offline regressions. Training configuration, seed and evaluation manifest hash
are inapplicable to M0. No training or live KSP run was performed.

## 2026-11-19: standalone 6DOF simulator source-domain acceptance

The `simulator/` module was re-run with the pinned `simulator/requirements-tested.txt` stack (Python 3.14.4, NumPy 2.4.6, Gymnasium 1.4.0). Per-check detail is in [`simulator/docs/VALIDATION.md`](../simulator/docs/VALIDATION.md); the machine-readable record is [`simulator/docs/validation-summary.json`](../simulator/docs/validation-summary.json).

- 62 simulator tests and the 8 existing offline tests pass.
- 100/100 scripted landings are safe on the agreed 5 s window; mean first-touchdown error is 1.283 m; the Wilson 95% safe-landing interval is 96.30%-100%.
- Zero-action and random-action baselines crash 10/10; 9 one-axis physical conditions (mass, inertia, thrust, gravity at 0.8/1.0/1.2) over 10 generalization starts all land safely inside 5 m.
- The summary was rebuilt as schema 2 and now records dependency versions plus source and start-manifest hashes; `python examples/build_validation_summary.py --check` verifies them. Manifest hashes in the earlier summary were not reproducible from the committed code and were regenerated.
- This is source-domain engineering evidence with a scripted controller. It does not validate KSP reset/step, real craft parameters, shared safety thresholds, cross-environment alignment, transfer, training or RQ1-RQ3. All `TBD` values and readiness flags in `configs/mun_landing.toml` are unchanged.

## 2026-11-18: Windows/WSL2 live transport and adapter skeleton

The current host is Windows 11 Pro with WSL2. KSP 1.12.5 (build 03190) runs on
Windows from the Steam installation; Python 3.14.4 runs in WSL2. kRPC 0.7.0
server and ModuleManager 4.2.3 were installed into the game's `GameData` and
verified against the official ZIP SHA-256 `cd4f062e...8bf028`.

- WSL2 mirrored networking reached Windows `127.0.0.1`; no port proxy was used.
- `scripts/doctor.py` passed: RPC/stream ports, kRPC 0.7.0 handshake, required
  services, Flight scene, 17 celestial bodies, 5 Hz UT stream and advancing
  simulation time.
- `scripts/telemetry.py` recorded 50 rows over 10 s with finite numeric fields
  and advancing UT while the vessel was on the Kerbin launchpad.
- `scripts/doctor.py --control-test` passed with
  `lights False -> True -> False`; the original state was restored.
- The adapter skeleton in `ksp_env/` connects and reads diagnostic telemetry.
  `configs/ksp_adapter_test.toml` was added for live adapter mechanics on the
  Kerbin launchpad. Its `observe()` path was verified with a target-relative
  Kerbin frame. A live mechanical reset and one zero-action step also passed:
  `SpaceCenter.load("persistent")` advanced `game_state_loads`, the Flight
  scene was reacquired, and the step advanced UT by one 0.02 s physics tick
  with `terminated=False`, `truncated=False` and `episode_evaluated=False`.
  These are adapter mechanics only, not task validation. Live reset and step
  used `KSP_LIVE_RESET_STEP=1` and remain opt-in.
- kRPC's `SpaceCenter.load` takes the save file name inside the current save
  folder (for example `persistent` or `Mun landing 1`), not the folder name.
- The new `configs/ksp_mun_test.toml` profile was exercised against
  `Mun landing 1`. `reset()` returned the `两级着陆器` in Flight with UT 367.78
  and `game_state_loads=15`; one zero-action step advanced UT by 0.02 s with
  `terminated=False`, `truncated=False` and `episode_evaluated=False`. This is
  adapter mechanics on a Mun orbit test target, not a landing.
- `scripts/probe_controls.py` was first run with `--execute --stage-once` on the
  pre-stage save. The saved state had current stage 3 and no active engine; one
  stage activation changed stage `3 -> 2` and active engines `0 -> 4`. A 0.05
  throttle pulse produced a peak of 12,000 N thrust and consumed 0.3545 kg of
  propellant; thrust returned to zero after the one-tick throttle lag.
- After `Mun landing 1.sfs` was overwritten with the stage-2 state, a second
  probe ran without staging. RCS pitch/yaw/roll pulses produced peak angular
  speeds of 0.006873/0.006753/0.006312 rad/s and residual rates around
  0.00027 rad/s after the counter-pulse. Monopropellant consumption was
  0.02013/0.02013/0.01699 units per axis, total 0.05725 units.
- Throttle was zeroed and control toggles were restored after both probes;
  staging is not reversible, so the live session remains at stage 2 with four
  active engines. This control-response run is mechanical evidence only; no
  Mun landing or task-level threshold was validated.
- After the probe, `Mun landing 1.sfs` was overwritten from the live game with
  current stage 2 and four active `liquidEngine3.v2` engines. A fresh
  `reset()` on that save returned stage 2, active engines `4 / 9`, throttle 0
  and 450 LiquidFuel. The test profile target was updated to the new saved
  sub-satellite longitude `139.70616`. This makes the save usable for
  reset/step mechanics, but it is still a test state, not a frozen task.
- Task observation, reset and step remain blocked by unresolved `TBD`
  target/frame/fuel/RCS decisions in the normative config; no landing was
  validated.
- kRPC 0.7 streams only send when a value changes. A constant mass on the
  launchpad updates once, so stream age is not a liveness signal.
- Offline `compileall` and unit tests passed, including the fake-transport step
  sequencing tests when the kRPC client is installed; the opt-in live smoke
  test passed separately with KSP running. The live test is gated by
  `KSP_LIVE=1` and is not part of CI.

These results establish the Windows/WSL2 transport and diagnostic layer only.
The Mun descent scenario, deterministic reset/step, landing criteria and all
learning experiments remain unimplemented.

## 2026-11-17: final proposal review

Reviewed the nine-slide `KSP_World_Model_Proposal_final.pptx` and recorded its qualitative decisions in [`task-spec.md`](task-spec.md), [`experiment-protocol.md`](experiment-protocol.md), [`proposal.md`](proposal.md) and `configs/mun_landing.toml`.

- The task is fixed as a nominal Mun powered descent with a fixed lander, fixed nominal gravity and a flat site; initial position, velocity and attitude are randomized.
- The observation is numeric telemetry, and the action is RCS pitch/yaw/roll plus main-engine throttle.
- Safety uses a 5 s stability window; crashes, timeouts and fuel-exhaustion aborts count as failures.
- The controlled comparison is the 2x2 pretraining/imagination design, with an 80% validation learning-curve threshold and at least three seeds.
- No code, KSP run, training run or experiment was performed. Numeric thresholds, initial-state distributions, target radius, budgets, OOD ranges and model architecture remain open.
- Offline Python syntax compilation and all 8 unit tests pass, including the new task-config invariants.
- The deck's images are context, not experimental results. The binary is kept in the gitignored `downloads/proposals/` directory; its SHA-256 is recorded in `proposal-source.json`.

## 2026-11-09: repository and scientific environment

Local machine: Apple M5, arm64, 16 GiB memory, Python 3.14.4. All dependencies were installed in this repository's `.venv`; the earlier standalone KSP control environment was preserved.

- `pip check`: no broken requirements.
- Imports: kRPC 0.7.0, protobuf 7.36.2, NumPy 2.5.4, SciPy 1.18.1, Gymnasium 1.4.0, PyTorch 2.15.0 and Matplotlib 3.11.2 passed.
- NumPy/SciPy linear solve: residual approximately `1.78e-15`.
- Gymnasium sampling: finite samples within the declared spaces.
- PyTorch CPU forward/backward: finite outputs, loss and gradients.
- MPS: available, simple tensor calculation returned the expected `17.0`. Full-model MPS training, numerical equivalence and throughput have not been tested.
- Local kRPC metadata: 0.7.0, KSP supported range 1.12.5–1.12.5.
- Offline readiness tests: 5 passed. These cover Python support, optional CPU-only/no-game execution, missing paths, plugin metadata/version and command-line defaults.
- Python syntax compilation passed for scripts/tests.

The environment check did not start KSP or connect to the live server. It does not certify a functioning Mun landing scenario. Raw local output is in ignored `reports/`; `validation-2026-11-09.json` is the shareable machine-check result without local absolute paths.

GitHub Actions runs only syntax and offline diagnostic-rule tests for Python 3.12 and 3.14. It does not install the scientific stack, launch KSP, evaluate a policy or certify Linux/CUDA training compatibility. Python 3.12 is within the declared dependency support range, but the complete scientific environment was tested locally only on 3.14.4.

## 2026-11-06: historical game control acceptance

The earlier local setup was tested with KSP 1.12.5.3190, kRPC server/client 0.7.0 and protobuf 7.36.2. In a sandbox test save:

- RPC and stream ports were bound to loopback `127.0.0.1:50000/50001`.
- SpaceCenter queries returned 17 celestial bodies and increasing universal time.
- Python placed a stock-derived Kerbal X on the Kerbin launchpad.
- The Flight check toggled lights `False → True → False`, reading back every state.
- Ten seconds of telemetry yielded 50 complete rows with finite numeric fields and advancing simulator time.

No ignition, lunar descent, automatic landing, training or long-duration stability test was performed. This historical result was not rerun on 2026-11-09; it is preserved as transport-level evidence. The original game, craft, saves and raw machine-specific reports remain outside Git.
