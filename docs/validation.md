# Validation record

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
