# Validation record

## 2026-11-25 — Debug-generated low-altitude snapshot

`KSPEnv.create_initial_save` was exercised live from `Mun landing 1`. The
requested state was 140 m above the test target, `(0, 0, -5)` m/s, task body
`+z` up, zero angular velocity and full resources. The vessel started in the
high save with `can_save=True`; after the debug setup the game reported
`can_save=False`, and `SpaceCenter.save` still wrote the named snapshot. The
saved file had SHA-256
`d27f26fa764c7df39d03bb3b4e223d9ad48fbf8ed2b6d22c56fffdb62ddb4d38`.

The debug calls themselves advanced the game by roughly five seconds per
position/velocity call, so the saved state was not identical to the request.
After reload, the adapter observed:

```text
position:            (0.118, 0.000, 92.699) m
velocity:            (-0.015, 0.000, -14.868) m/s
attitude quaternion: (0.000009, -0.000009, -0.707107, 0.707106)
tilt (task +z):      0.00145 deg
fuel:                450.0
```

The task-frame tilt was therefore effectively upright, confirming the fixed
`body_to_task_rotation_xyzw` mapping in the test profile. The generated test
save was removed and the game was restored to `Mun landing-100m`. A separate
`reset(save_name="Mun landing-100m")` also succeeded while `can_save=False`,
confirming that the load path does not depend on the save restriction. This is a
synthetic debug snapshot, not a task-grade initial state or landing result. The
three readiness flags remain `false`.

A second snapshot was kept as `Mun landing-debug-10m`. Its saved state was
10.05 m above the target, `(0, 0, -1.31)` m/s, task-frame tilt 0.0014° and
`can_save=False`; the file SHA-256 is
`7ab85fa78f2cf1a246c0439e5b71db4f3aac2cfbb070639da422b48a48d78530`.
With zero throttle, the adapter detected contact after 20 control steps at a
pre-impact speed of 5.153 m/s and tilt 0.0068°, classified it as `crash` and set
`terminated=True`. A simple hover-throttle descent reduced vertical speed to
about 0.45 m/s but was still airborne at 5.4 m after 60 steps, with tilt drifting
to about 8° because no attitude controller was active. The snapshot was restored
and the game left paused; no safe landing or task-level validation is claimed.

## 2026-11-25 — Candidate scene measurement

The existing `两级着陆器` and the equatorial point at latitude 0° and longitude
139.70616° were measured as the candidate craft and site. The initial pass was
read-only: it sent no controls and did not save or load the game. A separate
terrain check moved the craft out of the ray path with kRPC `Debug`, measured
the target, and then restored the candidate save.

The active craft reported 11,460 kg wet mass, 6,060 kg dry mass and 5,400 kg
of resources. Its four active `liquidEngine3.v2` engines consume LiquidFuel and
Oxidizer; their summed vacuum thrust is 240,000 N at Isp 345 s. The remaining
main-engine part and four separation motors were inactive. The available
LiquidFuel/Oxidizer was 450/550 units, or 2,250/2,750 kg; Monopropellant was 40
units (160 kg) and SolidFuel 32 units (240 kg). Principal moments were
18,760.8, 20,964.0 and 18,052.6 kg·m².

A 5 m terrain grid around the target, measured with the craft out of the ray
path, gave a fitted slope of 6.425°, terrain height at the target of -0.076 m
and an RMS residual of 0.054 m. A wider check found the terrain rising to about
100 m within 10 m to the north, so the point is next to a steep rise rather
than on an unqualified flat site. The craft used for the base snapshot reported
80.852 m surface altitude. The configured target altitude is still 0 m, and the
clearance reference remains `TBD`.

A coarse 151-point scan along the equator from longitude 139.0° to 140.5° found
the lowest 5 m-grid slope at latitude 0°, longitude 140.5°: 2.623°, with 1.23 m
of 10 m-grid relief. This is a replacement candidate for follow-up, not an
accepted site.

A 110 m target-frame base save (`Mun landing-candidate-100m`) was also created
and verified; the debug calls advanced the state, so the observed surface
altitude was 80.852 m rather than the requested value. This base save is only a
candidate asset and is not a task-grade initial state.

Following the measurement, the team decided that `main_engine_fuel` is the
total mass of the configured main-engine propellants in kg. For this craft the
configured LiquidFuel and Oxidizer correspond to 5,000 kg. The adapter now sums
amount × density; a live read-only `observe()` of the loaded candidate returned
`main_engine_fuel=5000.0` kg. The opt-in live reset/step test also passed with
the new field. Normalization and the target clearance reference remain `TBD`. This is candidate measurement
evidence, not a frozen task scene; the three readiness flags remain `false`.

## 2026-11-25 — Integrated contact and snapshot check

The fixed-frame, snapshot and contact work was rebased onto the game-time
stepping branch. From `Mun landing-debug-10m` with `can_save=False`, `reset()`
loaded the save and reported 10.0 m altitude, `-1.31` m/s vertical speed and
0.0015° task-frame tilt. A 10 Hz step advanced five physics ticks and 0.1 s of
UT; consecutive `control_session()` steps shared their UT boundary.

With zero throttle, the integrated evaluator detected contact after 21 control
steps. The touchdown event used the preceding airborne tick: 5.312 m/s total
speed, 5.312 m/s vertical speed, 0.0069° tilt and 0.093 m horizontal error.
It classified the attempt as `crash` and set `terminated=True`. A temporary
`create_initial_save` snapshot was also created and verified at 9.77 m with
`can_save=False` and then deleted. These are synthetic debug checks, not a
task-grade landing result; all readiness flags remain `false`.

## 2026-11-25 — KSP adapter control interval and contact mechanics

The adapter now advances one configured control interval per `step()` and keeps
provisional contact/stability state. Offline tests use a fake kRPC transport:
the 10 Hz case advances five 0.02 s ticks, records `dt_s = 0.1`, detects the
first `landed` transition, takes impact metrics from the preceding airborne
sample, completes a stable window, and classifies unsafe contact, liftoff,
timeout and vessel loss. The kRPC-enabled `.venv` root suite ran 57 cases with
two opt-in live checks skipped; an interpreter without kRPC also skips the
thirteen step/snapshot cases. The simulator suite passed 61 with one skip and
the agent suite passed 97 with one skip.

The live kRPC server was started after the offline work. `KSP_LIVE_RESET_STEP=1`
with `configs/ksp_mun_test.toml` passed: `SpaceCenter.load("Mun landing 1")`
returned the `两级着陆器` in Flight, and one zero-action step advanced UT from
`369.87999999995355` to `369.97999999995346` over five physics ticks. The
measured `dt_s` was `0.09999999999990905` s for the configured 0.1 s interval.
The vessel remained `orbiting`, `contact=False`, `status="flying"`,
`surface_altitude_m=12633.5`, `tilt_deg=82.739`, `angular_speed_rad_s=0.0000849`
and `main_engine_fuel=450.0` under the previous resource-unit field (the current field reports kg); no staging was used and `close()` neutralized the
controls. This is live control-interval evidence only. Live touchdown detection
is still pending because the available Mun save is an orbit at about 12.6 km,
not a descent snapshot. The earlier 2026-11-18 live evidence covers the previous
one-tick adapter. The three readiness flags remain `false`.

## 2026-11-25 — Stable S3 training in the nominal 6DOF simulator

From a fresh behavior-cloning warm start, seeds 31, 32 and 33 each completed 24 online iterations. Each had 100/100 safe landings at every scheduled validation and 100/100 safe in-target landings at iteration 24. Validation selected seed 33, iteration 22, by safety, safe in-target rate, raw simulator mean return and then later iteration. The selected policy met all gates on the same test manifest used in the November 24 study.

### Complete training runs

| Seed | Online validations | Minimum safe / in target (per 100) | Last safe / in target | Last landing error (m) | Last return | Best iteration / return | Training time (min) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 13 | 100 / 97 | 100 / 100 | 1.472 | 93.2908 | 16 / 93.3094 | 8.4 |
| 32 | 13 | 100 / 95 | 100 / 100 | 1.304 | 93.2648 | 24 / 93.2648 | 8.6 |
| 33 | 13 | 100 / 100 | 100 / 100 | 1.385 | 93.3745 | 22 / 93.3988 | 8.6 |

The actors and online world models from all three seeds matched the independent shaping probe tensor-for-tensor at each of the 13 online validation points and after behavior cloning. The probe and fresh run also agreed exactly on outcome counts, in-target rates, landing errors and raw mean returns at those points. The one-seed 48-iteration extension used the unshaped reward and is a separate diagnostic.

### Frozen comparison on 100 starts per split

| Split | Controller | Safe | Safe in target | Safe landing error (m) | Total fuel (kg) | Flight time (s) | First impact (m/s) | Raw return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | bc_only | 100 | 100 | 1.739 | 32.008 | 34.557 | 0.514 | 92.0332 |
| validation | online | 100 | 100 | 1.361 | 24.803 | 27.503 | 1.356 | 93.3988 |
| validation | scripted | 100 | 100 | 1.328 | 31.515 | 34.080 | 0.654 | 92.1085 |
| test | bc_only | 100 | 100 | 1.644 | 32.047 | 34.542 | 0.511 | 92.1042 |
| test | online | 100 | 100 | 1.307 | 24.818 | 27.481 | 1.353 | 93.4473 |
| test | scripted | 100 | 100 | 1.283 | 31.553 | 34.066 | 0.657 | 92.1716 |

On validation and test, the online actor matched cloning on safety and improved mean return by 1.3656 and 1.3431, respectively. It passed the 80% safety qualification, the cloning comparisons and the test targets of 95% safety and 90% safe in-target. Simulator scoring uses the original reward for all three controllers.

### Model and policy diagnostics

| Data distribution | Horizon (steps) | Position (m) | Velocity (m/s) | Tilt (deg) | Horizontal position (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected policy | 1 | 0.001 | 0.010 | 0.025 | 0.000 |
| Selected policy | 5 | 0.012 | 0.033 | 0.125 | 0.006 |
| Selected policy | 15 | 0.075 | 0.087 | 0.438 | 0.052 |
| Selected policy | 32 | 0.302 | 0.179 | 1.074 | 0.226 |
| Permanent prefill | 1 | 0.002 | 0.010 | 0.047 | 0.001 |
| Permanent prefill | 5 | 0.014 | 0.040 | 0.224 | 0.007 |
| Permanent prefill | 15 | 0.089 | 0.108 | 0.636 | 0.060 |
| Permanent prefill | 32 | 0.343 | 0.214 | 1.375 | 0.252 |

Across 400 paired 32-step windows on the selected actor's fixed validation flights, mean unshaped return was 24.3713 in the simulator and 24.2092 in the world model (mean absolute difference 0.1746). There were 0 safety-rate declines between consecutive evaluations.

### Inputs and reproduction

The [configuration](../agent/configs/s3_nominal.json) removes the actor's terminal value and uses a 5× horizontal-distance cost for imagined rewards. The critic fits shaped λ-returns. The online budget was 720 actor and 960 model updates per seed on Apple M5 CPU, two threads per process. The training implementation is commit `bed9f12`. Run metadata captured the edits on base `ce2994daa1ab` and the training source SHA-256 was `8e72d598604da352ee2866d43fc26fa7936419adb91f60c2a94c73572ba8619e`.

Replay SHA-256 `b26f50ac573ea3f7fa1113cc7137bb4868cf3757caf4e53a7e2fb33ac269021c`, initial world model `cade1219d72c89bf5219690be3ec8a7837a09f4a681a1771a124d66a3fad7cff`, validation starts `2bf7a0898b9ed4f38dd46f996765f675a97fe703f0c9dcc0fee83727d0e51321`, test starts `c12a88bc1047314fb482bbeec67ba5395e608cdd8be444e7f212f23e8a483f35` and selected checkpoint `a4451b97f733712c4d7832572d787ff24b816d214bb9f55fc03cdfc0837eae65`.

Run `agent/scripts/launch_seeds.sh --device cpu --seeds 31,32,33 --jobs-per-device 3 --out-root agent/runs/s3-stable` from the repository root after the data preparation in the [agent instructions](../agent/README.md). Use the three run directories for `report`, `plot` and `policy-risk`. The local outputs are `report/report.md`, `risk/index.json`, `validation-curve.png` and `probe-reproduction.json` under that run root. The implementation passed 96 agent tests and 61 simulator tests, with one skipped in each suite. Offline unittest discovery ran 28 cases with five skips. The critic-slope tool reproduced all 630 paired perturbations from the seed-32 reference, including both the predicted and the measured return differences. The archived recipe and its November 24 acceptance follow below.

## 2026-11-24 — S3 nominal 6DOF simulator landing

The validation-selected seed-32 actor landed safely inside the target on 100/100 fixed validation starts and 100/100 fixed test starts. The corresponding behavior clone and independently run scripted controller also scored 100/100 on both measures on each split. One online update cycle improved the selected actor's average return over cloning on both splits. Continuing the identical training recipe for 24 cycles led to 100/100 timeouts on the validation manifest for every seed.

### Gates and paired controller comparison

| Split | Method | Safe / 100 | Safe in target / 100 | Safe landing error (m) | Fuel (kg) | Flight (s) | First-touch speed (m/s) | Mean return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | Behavior clone | 100 | 100 | 1.556 | 31.173 | 33.715 | 0.591 | 92.2116 |
| validation | Selected online | 100 | 100 | 1.421 | 30.373 | 32.920 | 0.652 | 92.3602 |
| validation | Scripted | 100 | 100 | 1.328 | 31.515 | 34.080 | 0.654 | 92.1085 |
| test | Behavior clone | 100 | 100 | 1.423 | 31.220 | 33.694 | 0.585 | 92.2949 |
| test | Selected online | 100 | 100 | 1.367 | 30.410 | 32.895 | 0.645 | 92.4312 |
| test | Scripted | 100 | 100 | 1.283 | 31.553 | 34.066 | 0.657 | 92.1716 |

Fuel, time and return average all episodes. Landing error averages safe landings. Touchdown statistics concern the first contact in each episode that touched the surface. The fixed comparisons cover all 100 starts, with zero failed episodes. Each 100/100 safety rate has a Wilson 95% interval of 96.3% to 100%.

| Criterion | Validation | Test |
| --- | --- | --- |
| Safety qualification (≥80%) | 100%, pass | 100%, pass |
| Safe rate at least cloning, mean return strictly above cloning | Same safety, +0.1486 return, pass | Same safety, +0.1363 return, pass |
| Test target (≥95% safe and ≥90% safe in target) | — | 100% safe, 100% in target, pass |

All 100 matched test flights consumed less fuel and finished sooner with the selected policy than with its clone. Average differences were −0.810 kg and −0.799 s. Returns improved in 86/100 test flights and 88/100 validation flights. The scripted controller retained the lowest average landing error on both splits. On test the selected policy's first-touch speed rose from the clone's 0.585 to 0.645 m/s. The selected policy's maximum first-touch speed, tilt and angular speed on test were 0.765 m/s, 1.758° and 0.00372 rad/s, compared with nominal limits of 2.0 m/s, 10° and 0.2 rad/s.

### Training, inputs and provenance

The three seeds were fixed at 31, 32 and 33. Each independently initialized and fit its own actor and critic, sampled cloning minibatches and collected its own trajectories. All started from the same S1 world model and immutable nominal prefill. The prefill contained 124,953 transitions in 400 train episodes, with 300 low-noise scripted, 80 hard-landing and 20 random episodes. It contained 305 safe landings and 95 crashes. All 300 scripted episodes (102,329 transitions) met the cloning filter on behavior and noise alone. An initial training-split-only probe using 53 S1 demonstrations on 24 disjoint train starts found 13/24 safe and 3/24 in target with world-frame inputs and 24/24 on both measures with heading-aligned inputs (seed 31). With the final prefill, independent train probes scored 40/40 on both measures for each of the three seeds. Formal selection used the validation file.

| Artifact | SHA-256 |
| --- | --- |
| Frozen S1 starting model | `cade1219d72c89bf5219690be3ec8a7837a09f4a681a1771a124d66a3fad7cff` |
| Nominal train manifest, collection seed 240 | `7ceed2552768a6efd31e80c2e60da7ba4478cd257b69f813f8b929d663b6edd0` |
| S3 permanent replay | `b26f50ac573ea3f7fa1113cc7137bb4868cf3757caf4e53a7e2fb33ac269021c` |
| Frozen validation, seed 2026 / 100 | `2bf7a0898b9ed4f38dd46f996765f675a97fe703f0c9dcc0fee83727d0e51321` |
| Frozen test, seed 2026 / 100 | `c12a88bc1047314fb482bbeec67ba5395e608cdd8be444e7f212f23e8a483f35` |
| Selected seed-32 cloning-only checkpoint | `60911be1a687d8dc6697369871260866fd9c6acae7db3225ed8c57978cb4ebf5` |
| Selected seed-32 online checkpoint | `67a3ca83692d2f83c231e5750bfad0cbc84c400f078bd2f5585a4ed947ab652b` |

The fits ran on an Apple M5 CPU, two threads per seed, Python 3.14.4, NumPy 2.5.4 and PyTorch 2.15.0. The run metadata recorded base commit `e9c5795cf7013ceba894159352ab9b9b27bcf7a1` with S3 edits. Training-time `train.py` SHA-256 was `8b85449f0f17b5ab88a406ebb8b7b5c354338344ae915fe034336055f173d70e`. The implementation and three-seed launcher are in `0508ba2a6cbe4ba2f966527d237c18bb74ba2dcb`. All three original runs used the [archived nominal configuration](../agent/configs/s3_nominal_legacy.json). The MLP uses heading-aligned observation features and all four learned action channels. Cloning ran 6,000 updates (batch 256). Each of 24 cycles collected 6 full episodes, updated the model 40 times and applied 30 pathwise actor updates (H=64, batch 128, actor LR 0.00001). Training used the existing task reward, nominal initial distribution, contact thresholds and timeout. Each run completed 960 online model updates and 720 actor updates.

### Validation across the fixed 24-cycle budget

| Seed | BC return | Best round | Best safe / in target | Best return | Last return | First online qualification (s) | Full run (s) | Total real training transitions |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 31 | 92.214273 | 1 | 100 / 100 | 92.338163 | -72.820 | 38.11 | 371.55 | 198,420 |
| 32 | 92.211605 | 1 | 100 / 100 | 92.360219 | -67.685 | 39.61 | 381.76 | 202,321 |
| 33 | 92.033199 | 1 | 100 / 100 | 92.129925 | -77.328 | 38.86 | 350.17 | 189,806 |

All three untrained actors scored 0/100. The best online checkpoint was the first validation after cloning for all seeds, even though training ran to completion. Selection ranked safe rate, then mean return, then iteration. The global winner's round 1 beat its round 2 by 0.000524 mean return. Independent recomputation from the 45 archived evaluations (4,500 episode records) reproduced the validation summaries and checkpoint ordering. Reloading the selected and clone checkpoints produced the exact original validation status counts and mean returns. The three final comparison policies each ran on the frozen test file once, after selection.

Safe-in-target counts fell ahead of safety. Columns give the number of safe landings inside the 5 m target on the 100-start validation manifest.

| Seed | Clone | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 100 | 100 | 94 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | 100 | 100 | 96 | 92 | 81 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 33 | 100 | 100 | 93 | 42 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Seed-wise median collection rates were 3,208 / 3,179 / 3,211 simulator transitions/s. Corresponding model update rates were 113.48 / 110.97 / 115.97 per second and actor/critic update rates were 7.50 / 7.81 / 7.74 per second. These are within-phase CPU measurements. L40 scheduling and failure propagation were exercised with test subprocesses, and the launcher was also smoke-tested with real CPU training and resume. L40 throughput was not measured.

### World-model risk on visited states and prefill

The selected actor visited 33,000 simulator transitions across the 100 validation episodes. For each eligible observed origin, open-loop physical predictions followed recorded actions. Each cell gives mean position error (m), velocity-vector error (m/s) and tilt error (degrees). The final column reports only the horizontal component of position error on the selected actor's visited states.

| Steps | Selected policy, all origins | Permanent prefill, all origins | Selected policy, true endpoint below 10 m | Visited horizontal error (m) |
| ---: | --- | --- | --- | ---: |
| 1 | 0.001 / 0.005 / 0.010 | 0.002 / 0.010 / 0.040 | 0.001 / 0.007 / 0.005 | 0.000 |
| 5 | 0.008 / 0.019 / 0.049 | 0.012 / 0.038 / 0.183 | 0.007 / 0.017 / 0.024 | 0.004 |
| 15 | 0.041 / 0.049 / 0.183 | 0.079 / 0.101 / 0.519 | 0.035 / 0.040 / 0.122 | 0.029 |
| 32 | 0.157 / 0.095 / 0.601 | 0.299 / 0.191 / 1.188 | 0.119 / 0.073 / 0.436 | 0.121 |

For return comparisons, both branches start from the same observation and history, then independently recompute actions with the frozen actor. Returns are undiscounted over 32 steps and include survival weighting in imagination. Every validation episode contributed four windows for the winner.

| Origin | Windows | Simulator return | Imagined return | Mean absolute gap |
| --- | ---: | ---: | ---: | ---: |
| reset | 100 | -1.40040 | -1.39952 | 0.00802 |
| middle | 100 | -0.61824 | -0.61754 | 0.00297 |
| first_contact | 100 | -0.30822 | -0.28662 | 0.02165 |
| terminal | 100 | 99.86727 | 99.84934 | 0.01914 |
| all | 400 | 24.38510 | 24.38642 | 0.01295 |

### Safety drops and late flight behavior

All three safety declines were from 100 safe episodes to 100 timeouts and persisted to round 24. The six archived checkpoints were rerun on the same 100 starts. Physical errors below are the mean 32-step position (m), speed-vector (m/s) and tilt (degrees) errors on each checkpoint's own visited states. First-touch distributions give median / 95th percentile. A dash means no episode reached the ground.

| Seed / round | Safe / 100 | First contacts / 100 | Impact speed (m/s) | Impact tilt (degrees) | 32-step physical errors |
| --- | ---: | ---: | --- | --- | --- |
| 31 / 16 | 100 | 100 | 0.659 / 0.677 | 1.824 / 1.909 | 0.581 / 0.368 / 2.782 |
| 31 / 18 | 0 | 0 | — | — | 0.674 / 0.422 / 3.757 |
| 32 / 14 | 100 | 100 | 0.609 / 0.685 | 0.556 / 0.912 | 0.444 / 0.281 / 2.118 |
| 32 / 16 | 0 | 100 | 0.673 / 0.674 | 0.880 / 0.881 | 0.433 / 0.266 / 2.134 |
| 33 / 18 | 100 | 100 | 0.881 / 0.900 | 1.313 / 1.501 | 0.549 / 0.340 / 2.783 |
| 33 / 20 | 0 | 0 | — | — | 0.522 / 0.275 / 3.718 |

At the last safe check before the drop, mean first-touch horizontal errors were 32.53 m (seed 31), 15.74 m (32) and 23.16 m (33). Seeds 31 and 33 never touched down at the next evaluated checkpoint. For seed 32, round-16 impacts occurred at 87.39–88.09 s. Even the earliest left less than 3 s to complete the 5 s stability window before the 90 s timeout. Impact speeds stayed below 0.675 m/s. Later seed-32 policies never touched down. The drop coincided with larger position errors for seed 31 and smaller ones for seeds 32 and 33.

At round 24, seed 32 reached zero safe landings and had zero first contacts. Its 32-step position, velocity-vector, tilt and horizontal-position errors were 0.245 m, 0.158 m/s, 1.558° and 0.151 m. There were 300 paired reset, middle and terminal windows, with mean simulator return −17.4701, imagined return −17.4879 and MAE 0.0251. Short-range predictions agreed with both the successful and the failing actor's realized returns. These measures do not establish the accuracy of long-horizon policy gradients or the cause of drift.

### Reproduction, artifacts and checks

The [agent README](../agent/README.md) gives preparation, L40 launch, resume, selection and plotting commands. The [method](../agent/docs/METHOD.md) describes information boundaries and sampling. The Mac runs used the commands below from the repository root.

```sh
mun-agent collect --config agent/configs/s3_prefill.json --seed 240 --device cpu --out agent/data/s3-nominal-prefill
mun-agent prepare-nominal --validation agent/data/s3-validation.json --test agent/data/s3-test.json
# Run with SEED=31, 32 and 33 in separate processes.
mun-agent train-policy --config agent/configs/s3_nominal_legacy.json --seed "$SEED" --device cpu --data agent/data/s3-nominal-prefill --world-model agent/runs/s1-contact-seed31/best.pt --validation agent/data/s3-validation.json --out "agent/runs/s3-nominal-seed$SEED"
```

In `agent/runs/`, `s3-nominal-learning-curve.png` shows validation safety, target rate, return and first-touch speed. `s3-nominal-report/{report.json,report.md}` holds the three-seed results and same-manifest baselines. `s3-nominal-risk/index.json` links both winner/last policy diagnostics and all six drop snapshots. The selected inference and training checkpoint is `s3-nominal-seed32/checkpoints/best.pt`. Data, checkpoints, generated reports and plots are excluded from Git. This milestone used the standalone nominal simulator. Live KSP, physical OOD tests and the transfer research questions are separate experiments.


The four repository checks passed locally with the pinned environment. Compilation succeeded, root unittest ran 28 cases (23 passed, 5 opt-in skips), simulator pytest had 61 passes and one NumPy-specific skip, and agent pytest had 92 passes and one CUDA-only skip. The 14 additional agent cases since S2 cover full-action cloning and heading symmetry, fixed manifests, on-policy prediction diagnostics, checkpoint ranking and interrupted evaluation, the multi-seed launcher and the learning curve. The existing two Gymnasium warnings concern unbounded observation-space limits. The pinned Matplotlib version is included in the agent development extra so CI can run the plotting tests. `pip check` found no broken requirements.

## 2026-11-24 — S2 vertical simulator policy

The frozen post-update actor safely landed on all 100 fixed vertical validation starts, above the 80% gate (Wilson 95% interval 96.30% to 100%). Behavior cloning alone also landed on all 100. The online update reduced fuel and flight time on every paired start, with a higher first-touchdown speed. This is a single training seed in the vertical simulator diagnostic. The nominal 6DOF test manifest and live KSP are outside this measurement.

### Acceptance and selected checkpoints

| Check | Measurement |
| --- | --- |
| End-to-end smoke | Real train-split collection, online eight-step model fitting, actor/critic updates, checkpoint loading, official simulator evaluation and interrupted-run recovery passed |
| Initial untrained policy | 0/100 safe, 100 out of bounds |
| Behavior cloning only | 100/100 safe, mean fuel 29.803 kg, flight time 33.043 s, undiscounted return 95.861 |
| After one online iteration | 100/100 safe, mean fuel 23.917 kg, flight time 27.331 s, undiscounted return 96.564 |
| Failed episode counts | Selected policy and clone each had zero crashes, timeouts, out-of-bounds episodes and fuel aborts. The selected checkpoint's failure-only `status_counts` is `{}` |
| Single-run time limit | First eligible passing evaluation at 71.72 s on Apple M5 CPU, two threads, within the two-hour limit. The configured cutoff was 7,100 s |
| Within-phase update rates | 75.18 world-model updates/s and 5.10 actor updates/s including associated critic fitting |

Epoch-zero cloning and subsequent online validations used separate checkpoints. Reloading each produced exactly the recorded outcomes, fuel, flight times and returns on all 100 starts. Per-episode fuel savings ranged from 5.726 to 6.084 kg (mean 5.886 kg or 19.75%). Flights finished 5.559 to 5.931 s sooner. Mean first-touchdown speed rose from 0.567 to 1.233 m/s. The maximum after the update was 1.247 m/s against the simulator limit of 2.0 m/s. Horizontal error and RCS use are zero by construction in this task.

### Inputs, code and training

All seven configurations below used the same policy initialization seed 31 and frozen S1 world model. The vertical validation manifest projects 100 nominal validation starts (seed 2026) to zero horizontal position, horizontal velocity, tilt and angular rate, preserving altitude, vertical speed, yaw and fuel. Both simulated and imagined actions mask the three RCS channels. Evaluation uses the simulator's `evaluation.evaluate`, default `SimConfig` and a frozen file of initial states. Code rejects train/validation reset-seed overlap. Online collection seeds are `31 * 1,000,000 + iteration` on the train split.

| Input or output | SHA-256 |
| --- | --- |
| S1 starting model | `cade1219d72c89bf5219690be3ec8a7837a09f4a681a1771a124d66a3fad7cff` |
| Pretraining replay, vertical collection seed 230 | `5ebe8dcde9446577dca5085ecd825630e2d8c7e5606d77bf91986c7e3adea221` |
| Vertical prefill manifest | `69e3fc0fc29828764f5b9ec30cc57ea4a3e6e0f6905d6038183c60a5f09c677d` |
| Fixed vertical validation starts | `32d28afbb7e58f782aede7cb509e83111c141ae48d01a5762a3b957e1df39185` |
| Behavior-cloning-only checkpoint | `7773bbc3b2e17a8017a81af09cee01acd40ae297e86333d04c27d5b589d019ec` |
| Selected model, actor and critic | `cca47bd92f82c6c93e771fa2cbd489a6d5f138fec0129a4b85c489167eeea398` |

The machine used Python 3.14.4, PyTorch 2.15.0 and NumPy 2.5.4. G ran on top of `89b8b6922f3547b2d511904eb64e8da58e752c1c` with warm-start changes. Its run-time SHA-256 for `train.py` was `77e1094f88b040554c759677924e7e75fe4ade59e99eb099399cf36fe5b82bc6`. A same-seed rerun from the committed implementation `5d4fce7d66baeb627f1fb3af941d54af9d5c5d77` and the published configuration passed in 71.06 s. All actor, critic and world-model tensors were bitwise identical to the selected run at both the cloning-only and the post-update checkpoints. The rerun outputs are in `agent/runs/s2-reproduce-G31/`.

The permanent replay contained 67,962 transitions across 200 episodes (70 scripted, 120 hard-landing variants, 10 random). It included 87 safe landings, 99 crashes, 4 timeouts and 10 out-of-bounds episodes. G cloned 18,042 actions from 53 scripted episodes with noise standard deviation at most 0.1. Selection used behavior provenance, with no filter on episode outcome. Cloning used 6,000 Adam updates of batch 256, with half the batch sampled from below 10 m. One online cycle then collected six episodes (1,979 real transitions, all safe), fit the world model 40 times and updated the actor 30 times on 128 imagined trajectories per batch. Training used the simulator task reward and the S1 event calibration throughout.

### Validation curves and fallback order

Every validation point uses all 100 frozen starts. A through F had no cloning. Each row reports the last completed training iteration even when the last scheduled validation occurred earlier. Wall times are from the last saved iteration in interrupted runs. B and G completed normally.

| Run on seed 31 | Change from its stated baseline | Completed / last evaluated iteration | Best safe count | Last evaluation statuses | Wall time (s) |
| --- | --- | ---: | ---: | --- | ---: |
| A, preliminary pathwise | Initial H=32, exploration σ=0.10. Stopped to clip the diagnostic-head gradient separately | 15 / 12 | 0 | 100 crash | 89.3 |
| B, corrected pathwise | Corrected clipping, 50% permanent replay, 50% near-ground starts, actor LR 0.0002 | 90 / 90 | 35 | 100 timeout | 1,161.4 |
| C, sampling | B with 85% permanent replay, 75% near-ground starts stratified by contact phase, exploration σ=0.15 | 19 / 16 | 0 | 100 out of bounds | 189.4 |
| D, longer horizon | C with H=64, actor LR 0.0001 and a smooth +100 upper bound on critic output | 23 / 20 | 0 | 100 timeout | 391.6 |
| E, heading invariance | D with actor and critic inputs rotated to a common horizontal heading | 39 / 36 | 0 | 100 timeout | 735.8 |
| F, sampled gradient | E with advantage-weighted Gaussian score-function updates, tanh-input σ=0.30 for throttle | 48 / 48 | 0 | 100 crash | 456.5 |
| G, cloning + pathwise | E with low-noise behavioral warm start, checkpoint selection begins after online updates | 1 / 1 | 100 | 100 safe | 71.7 |

B's curve was 0/100 at iterations 0 through 24, 35/100 at 28 (173,931 real training transitions), 1/100 at 32 (194,533), then 0 at every four-iteration check through 88 and at 90 (507,733). At the selected B checkpoint, the 65 failures were all timeouts and 60 of them never touched down. G's curve was 0/100 untrained at 67,962 transitions, 100/100 after cloning at the *same* count, and 100/100 after online training at 69,941 transitions. Metrics count prefill and online training interactions, excluding validation rollouts and imagined steps.

All C, D, E and F validation points scored zero. C's critic diverged outside the attainable reward range. D bounded optimistic values, while E's FIFO contained 119 timeouts out of 120 at the iteration-35 diagnostic. Its mean terminal-state critic prediction on those timeouts was −661 versus an observed return of −50. F was still at 0/100 at its planned iteration-36 fallback review, and its last 120 collected episodes all crashed (median first touchdown 13.99 m/s). The final F critic predicted +100 at the terminal origin of these trajectories against roughly −100 realized. The real/imagined comparison at B and the outcome diagnostics led to the sampled-gradient comparison and then the warm start. No event recalibration, initial-state curriculum or additional reward shaping was applied.

### Frozen actor, paired 32-step returns

A simulator rollout supplied an observation and its causal history at each window origin. Both branches then ran the selected deterministic actor independently, on their own resulting states. The simulator return is the undiscounted sum of environment rewards. Imagination uses the learned dynamics, the calibrated events and survival-weighted task reward, with no critic bootstrap. Each of the 100 episodes supplied one window for each phase. All 400 origins were evaluated.

| Window | Count | Simulator mean | Imagined mean | Mean imagined − simulator | Mean absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reset | 100 | −0.4150 | −0.3426 | +0.0724 | 0.0724 |
| Mid-episode | 100 | −0.4995 | −0.4604 | +0.0391 | 0.0391 |
| Around first touchdown | 100 | −0.2472 | −0.2295 | +0.0177 | 0.0177 |
| Ending at episode termination | 100 | 99.9375 | 99.8738 | −0.0638 | 0.0638 |
| All windows | 400 | 24.6940 | 24.7103 | +0.0163 | 0.0482 |

B's selected actor also had a small closed-loop difference on its own validation trajectories (340 windows, signed gap −0.0020, MAE 0.0200), despite its 65 timeouts. Broader replay probes reveal larger errors. When following recorded actions on G's six noisy training episodes, the frozen model's 1-step position/speed errors were 0.002 m and 0.017 m/s, rising to 0.866 m and 0.558 m/s after 32 steps. On the full permanent replay, the 32-step errors were 1.284 m and 0.872 m/s. The mean predicted minus observed return over the terminal 32 recorded steps was −0.041 on the six online episodes and +46.117 on the mixed prefill. At the last prefill state before termination, mean critic predictions were +21.57 for safe trajectories and +10.77 for crashes, while realized rewards were approximately +100 and −100. Calibration on failure trajectories and long-horizon value estimation remain limitations.

### Reproduction and local artifacts

The [agent README](../agent/README.md) gives the exact `prepare-vertical`, `train-policy`, `evaluate-policy` and `plot` commands. [The method](../agent/docs/METHOD.md) describes the pathwise gradient, sampled fallback and cloning details. Run B with `agent/configs/diag_vertical_pathwise.json` and G with `agent/configs/diag_vertical.json`, both using `--seed 31 --device cpu`. Run the following command to plot both curves.

```sh
mun-agent plot --run agent/runs/s2-pathwise-B31 --run agent/runs/s2-clone-G31 --out agent/runs/s2-clone-G31/validation-curve.png
```

Within `agent/runs/s2-clone-G31/`, `checkpoints/{untrained,bc-only,best}.pt` stores the three stages, `validation.jsonl` contains their validation points, and `metrics.jsonl` records collection, update rates and losses. `best-eval.json` contains individual simulator episodes and paired return windows. `bc-only-eval.json` contains the independent clone evaluation, `bc-vs-g-paired.json` the matched changes, and `replay-diagnostics.json` the stratified model/critic probes. `policy-change.json` compares both actors on the same 67,962 observed origins. Native mean throttle fell by 0.0844 and changed by more than 0.01 on 77.6% of origins. The validation plot, training arrays and checkpoints are local artifacts excluded from Git.

All four repository checks passed. Syntax compilation succeeded. Root unittest ran 28 tests (23 passed, 5 skipped), simulator pytest recorded 61 passes and one historical NumPy-version-specific skip, and agent pytest recorded 78 passes with one CUDA hardware skip. The S1 agent suite had 57 passing cases. The 22 additional cases cover both gradient estimators, derivatives through survival and terminal value, lambda returns, action projection and history, the fixed-manifest gate, clone selection, paired reports, staged curves and end-to-end recovery. The simulator emitted its two existing Gymnasium observation-bound warnings.

## 2026-11-24 — S1 deterministic simulator world model

The frozen model passed W1, W2, W4 and W5 on the original 50 held-out simulator episodes. W3 and W6 have reporting requirements only. All trajectories use recorded simulator actions. No policy was trained or evaluated in S1.

### Implementation and data

Evaluation used clean implementation `c2f89163e7ec928a1f030d1bf72bfc3fae73aaf8`. Training ran on Apple M5 CPU, Python 3.14.4, torch 2.15.0, with four CPU threads, from `6b8f2b1` with S1 changes in the working tree. For the selected fit, the training-time SHA-256 values of `deterministic.py` and `deterministic_fit.py` were `1fb4563e4a017406556d3d15dde2dd67353514d3768c0d5b3c6692cd8cd7f978` and `e40df046cc1ce4c7116ec0f091ee2766c5a915e1eb95d74ea2a86f56b7212529` respectively. The final checkpoint SHA-256 is `cade1219d72c89bf5219690be3ec8a7837a09f4a681a1771a124d66a3fad7cff`. The RSSM comparison uses `m2r-long-A/step-028000.pt`, SHA-256 `e761885ecebc424e8c16c2877a0cc69f187c83683e6ca6ee17d89f89926261f3`.

Each collection was regenerated with the original behavior mix, reset seeds and a new recorded transition duration. All six existing arrays matched the original replay exactly. Collection metadata matched apart from the replay checksum. Durations are used for reward verification. Neither durations nor event labels are model inputs.

| Split | Seed | Episodes / transitions | Manifest SHA-256 | Replay with durations SHA-256 |
| --- | ---: | ---: | --- | --- |
| train | 170 | 200 / 59,362 | `43afb9b859cdf5ad67d298610b7d9ab437bf3a30fba25cb849c171c745b2ba8d` | `965205a9e3cbb22b80928da12ab7d069cc97f6f2d8be0fb4bae6677d45aa4c25` |
| dev | 180 | 50 / 14,846 | `3ffb03d0258e7a1c6ba4c33cb969e838705bc1bfe5a6f0fad574fb6d59ce5ed1` | `8e5c6a9f523f62e2b7d26b24029a75e424411585675c3d67656466c61b57807c` |
| heldout | 190 | 50 / 14,332 | `ba1921414c2b258add7bfee2a4e8da2a14c6e04951cb344d072fb450323cecdb` | `36636630de6082b54c71d6ba66f37752ee8ccfb7928956ff8df6d1eba910fb99` |

### Development and selection

Three fits used identical splits and initialization seed 31. Every fit evaluated the frozen development scoring rule every 750 optimizer updates. Within each fit, the smallest development score selected its checkpoint. Each subsequent variant was motivated by development diagnostics and initialized independently. The smallest of the three scores selected the sole checkpoint evaluated on the final replay. The specified 30-minute single-fit limit was observed throughout.

| Fit configuration | Updates / selected | Wall time, s | Dev score | Dev first event | Dev final 15 rewards |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s1.json` | 15,000 / 10,500 | 102.76 | 1.6108 | 47/50 | 25/50 |
| `s1-kinematic.json` | 22,500 / 15,750 | 179.41 | 0.8132 | 47/50 | 42/50 |
| `s1-contact.json` | 22,500 / 15,750 | 205.08 | 0.6198 | 48/50 | 46/50 |

The first model left the ground spuriously during 11/19 safe development suffixes, resetting the stability counter. The kinematic support and liftoff variant recovered all 19 safe suffixes. Adding continuous contact and motion margins to the event readout improved final-window return agreement to 46/50 on development. The 195 observed training liftoffs initialized the positive velocity branch. All final model selection used development data only. The 15,750-update checkpoint passed all four gates on the complete development replay before the final run.

### Physical predictions on the final replay

Each cell gives mean endpoint position error (m), speed-vector error (m/s) and absolute tilt error (degrees). All four columns use the same origins and endpoints. The RSSM filters the real prefix up to each origin. The two baselines use the observed origin or a single mean fitted on training frames.

| Horizon | Endpoints | Deterministic | Selected RSSM | Persistence | Training mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14,332 | 0.0017 / 0.0107 / 0.0443 | 3.3846 / 0.4920 / 1.9268 | 0.3693 / 0.0512 / 0.2188 | 30.4170 / 2.5352 / 5.9456 |
| 5 | 14,132 | 0.0139 / 0.0420 / 0.1966 | 4.7127 / 0.5665 / 2.4181 | 1.8413 / 0.2199 / 1.0042 | 29.8505 / 2.5297 / 5.9522 |
| 15 | 13,632 | 0.0893 / 0.1090 / 0.5716 | 7.6654 / 0.7936 / 3.6791 | 5.4716 / 0.5981 / 2.3831 | 28.5021 / 2.5144 / 5.8503 |
| 32 | 12,782 | 0.3204 / 0.1999 / 1.1951 | 11.3042 / 1.2343 / 5.9378 | 11.5178 / 1.1481 / 3.7039 | 26.4681 / 2.4866 / 5.2211 |

| Gate | Final measurement | Criterion | Result |
| --- | --- | --- | --- |
| W1 | All 12 comparisons passed. At 15 steps the errors were 0.0893 m, 0.1090 m/s and 0.5716° | ≤ 50% of persistence at each horizon and ≤ 0.5 m, 0.2 m/s, 1° at 15 steps | Pass |
| W2 | 5,109 endpoints below 10 m, 0.0547 m/s and 0.2884° | ≤ 0.2 m/s and 1° at 15 steps | Pass |
| W3 | 31 contacts in safe episodes, 0.3443 m/s and 0.0239°. 33 in crash episodes, 0.0520 m/s and 0.6956° | One-step errors on air-to-ground transitions, including repeat contacts | Reported |
| W4 | 50/50 first outcomes within one step (16/16 safe, 33/33 crash, 1/1 timeout). Mean terminal probability on 14,282 nonterminal steps was 0.00836% | Overall ≥ 95%, safe and crash each ≥ 90%, mean nonterminal risk ≤ 1% | Pass |
| W5 | 47/50 within 25 reward units. Safe 16/16, crash 30/33, timeout 1/1 | ≥ 85% | Pass |
| W6 | Reward-head MAE 0.0377 over all 14,332 transitions and 6.4926 on the 50 terminal transitions | Diagnostic head only | Reported |

49/50 first predicted events occurred on the exact observed step, one occurred one step early. W5 missed three crashes (zero-based episodes 19, 33, 44), with true versus predicted 15-step returns of −101.24 versus −1.21, −101.47 versus −1.52, and −100.26 versus −68.71. In real windows surviving all 32 transitions, mean predicted survival was 0.99852 across 3,584 origins below 10 m and 0.99992 across 9,148 origins at or above 10 m.

As a separate frozen-model check, each episode supplied its first actual air-to-ground endpoint and the recorded actions over the preceding 1, 5 or 15 control steps. Errors below are velocity-vector error (m/s) and post-transition tilt error (degrees). These are sampled endpoints after the simulator's contact response. Ground contact is inferred from clearance on the observation grid.

| Final outcome | Episodes | 1 step | 5 steps | 15 steps | Exact first contact from 15 steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Safe landing | 16 | 0.619 / 0.019 | 0.488 / 0.050 | 0.378 / 0.093 | 10/16 |
| Crash | 33 | 0.052 / 0.696 | 0.259 / 1.782 | 0.583 / 3.493 | 27/33 |

Six crash trajectories had yet to contact the ground by the observed first-contact frame in the 15-step prediction. For safe episodes, three predicted contact early and three had yet to touch by that frame. The timeout episode had a contact and is separately recorded in `agent/runs/s1-contact-seed31/first-contact.json`. As W4 supplies a fresh true state at each origin, its 50/50 agreement can coexist with these timing errors in open loop.

True-state reward replay used actual durations, including 49 shortened held-out steps. Absolute nonterminal process error (median / 95th percentile / max) was 0.00000183 / 0.00000552 / 0.00000863. Terminal total error was 0.00000186 / 0.00000481 / 0.00000574. Full split distributions, input hashes, and per-episode W4/W5 records are in the ignored artifacts `agent/runs/s1-replay-parity.json` and `agent/runs/s1-contact-seed31/heldout.json`.

### Reproduction and checks

The [agent README](../agent/README.md) gives the three collection commands, model fitting and the four-way final report. The [method](../agent/docs/METHOD.md) specifies gradient isolation, window masking, calibration, benchmark roles, and the development selection rule. The final report took 3.83 s on CPU. The four repository checks passed. Compilation succeeded, the root unittest suite ran 28 tests (5 skipped), simulator pytest had 61 passes and 1 skip, and agent pytest had 57 passes (44 previously). The simulator skip is the historical NumPy-version-specific manifest check. No changes to the simulator, KSP adapter or shared task configuration were required.

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
