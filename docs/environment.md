# Environment requirements

## Runtime

| Component | Selected version / requirement | Purpose |
| --- | --- | --- |
| KSP | 1.12.5 | Target physics simulation, installed separately |
| kRPC server and client | 0.7.0 on both sides | State queries, controls, reset and physics-tick coordination |
| ModuleManager | 4.2.3 from the official kRPC release ZIP | Bundled mod dependency |
| Python | 3.12–3.14; locally tested with 3.14.4 | Minimum 3.12 follows the selected NumPy/SciPy versions |
| protobuf | 7.36.2 | kRPC wire format |
| NumPy / SciPy | 2.5.4 / 1.18.1 | State arrays, rotations, numerical work |
| Gymnasium | 1.4.0 | Proposed standard environment API |
| PyTorch | 2.15.0 | World-model and policy computation; algorithm not yet chosen |
| Matplotlib | 3.11.2 | Offline learning curves and landing-error plots |
| pytest | 9.2.0 | Offline diagnostic tests |

Direct versions are pinned in `requirements/runtime.txt`; `requirements/macos-arm64-py314.lock.txt` is a historical local snapshot, not the current host lock. KSP-interface work uses `krpc`/`protobuf`; `numpy`, `gymnasium` and `matplotlib` are installed for later observation, wrapper and plotting work. PyTorch and SciPy are only needed for model work.

The current development host is Windows 11 Pro with WSL2. KSP 1.12.5 runs natively on Windows; Python 3.14.4 runs in WSL2 and reaches the kRPC server over TCP. WSL2 uses mirrored networking with `hostAddressLoopback=true`, so `127.0.0.1:50000/50001` in WSL2 reaches the Windows loopback interface. If that setting changes, bind kRPC to the Windows LAN address instead and pass it with `--address`. CPU is the functional baseline; training hardware remains open.

## Installation

On macOS/Linux with an appropriate Python installed:

```sh
bash scripts/bootstrap.sh
```

The current WSL2 host does not provide `ensurepip`, so use `uv` for the
KSP-interface-only environment:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
  krpc==0.7.0 protobuf==7.36.2 numpy==2.5.4 gymnasium==1.4.0 matplotlib==3.11.2
```

Native Windows Python is not used by this repository.

Install KSP through its normal distribution. Download the [official kRPC 0.7.0 release](https://github.com/krpc/krpc/releases/tag/v0.7.0), verify the release digest, then merge its `GameData` into the KSP game directory while the game is closed. Preserve any existing settings and avoid duplicate ModuleManager versions. Do not commit the game, DLLs, saves or craft assets.

Known official ZIP SHA-256:

```text
cd4f062eaea6b74eff68a3be2dc3958a0f9373bafc2c9daee08aed6b988bf028
```

On a fresh kRPC 0.7 install the server list is empty. Load a save, open the
kRPC window, click **Add server** (the default is `Protobuf over TCP`,
`127.0.0.1`, RPC `50000`, stream `50001`), then click **Start server**. Enable
auto-start and auto-accept if desired; the main menu has no server. No global
Mono/.NET installation, CKAN or kOS is required for this Python interface.

To include local plugin-file checks, provide the game root (the directory containing `GameData`). In WSL2, the game installed under `E:\` appears at `/mnt/e/`:

```sh
export KSP_PATH="/mnt/e/Program Files/Steam/steamapps/common/Kerbal Space Program"
.venv/bin/python scripts/check_environment.py --require-ksp --output reports/environment.json
```

`KSP_PATH` is used only by the offline file check. The live doctor and
`ksp_env` use the configured address/port. There is no automatic `.env` loading.

## Requirements for a valid Mun landing environment

The final proposal (reviewed 2026-11-17) fixes the qualitative task: a fixed nominal lander, fixed nominal gravity and a flat target site, with randomized initial position, velocity and attitude. The team froze the nominal distribution, control interval, episode horizon and safety limits on 2026-11-25. Craft measurements, site and data budgets remain open. [`task-spec.md`](task-spec.md) is the normative task contract and [`experiment-protocol.md`](experiment-protocol.md) is the controlled comparison protocol.

1. **Initial conditions and reset.** A named descent save must specify the lander, Mun location, target, position, velocity, attitude, angular velocity and fuel state. Use `SpaceCenter.save/load`, record `can_save` when creating a save, check `game_state_loads`, reacquire state and verify post-reset tolerances. `can_save` does not gate loading a low-altitude save.
2. **Step timing.** Use kRPC 0.7 `hold_tick`, `next_tick`, `release_tick` for synchronized control where suitable; record actual UT deltas. Do not wait for streams while holding a tick, and always release in `finally`. Hold timeouts must be detected through measured timing. Wall-clock sleep and the existing 5 Hz CSV logger are not deterministic training steps.
3. **Observation/action contract.** Use numeric telemetry: position and velocity relative to the target, attitude quaternion, angular velocity and remaining main-engine propellant mass, in SI units with explicit frames. Actions are RCS pitch/yaw/roll plus main-engine throttle; the control interval and RCS command semantics must match across the KSP adapter and pretraining simulator. An autopilot inner loop changes the task dynamics and must be documented if used. See `task-spec.md` for the full contract.
4. **Episode outcomes.** Safety requires touchdown speed, tilt and angular velocity within limits and stable for 5 s. Use the [approved limits](nominal-task-decisions.md). Define touchdown from contact/situation plus stable observation over multiple ticks, impact speeds, attitude, surviving vessel/parts and target error. Crashes, timeouts and fuel-exhaustion aborts count as failures; record safety, precision and propellant separately. Gymnasium `terminated` and `truncated` have different meanings; connection loss is an infrastructure error, not a successful landing.
5. **Metrics and data budgets.** Report safe-landing rate, precision rate and landing error with stated denominators: precision rate uses all attempts; error statistics use safe landings. Report total propellant consumed and the main-engine/RCS breakdown on safe landings. Log transitions, actual simulator seconds, reset count, training and evaluation interactions, policy/model updates, GPU hours, wall-clock time, seeds, policy version, split and dataset hash. Do not count imagination as KSP data. All RQ2 variants use the same recorded KSP data and common evaluation protocol.
6. **Physical variations.** The final proposal requires one parameter changed at a time, ranges fixed before testing, and the world model and policy frozen without additional training; evaluate in both the fast simulator and KSP. Mass and inertia are read-only telemetry, so generate measured fuel/payload/geometry variants and record their coupled changes. Engine `thrust_limit` can vary suitable liquid-engine thrust; it is distinct from the policy's throttle. `Debug.gravity_multiplier` modifies gravity globally for vessels; record and restore it, rather than assuming the read-only body's `surface_gravity` reflects the multiplier. Validate every OOD variant before using it.

Debug APIs may help build/reset scenarios. Freeze the intended physical variant and disable unrelated cheats such as infinite fuel during evaluation. Gravity variation is an explicit experimental condition. Test and restore global debug state in an isolated save.

## Acceptance boundaries

| Check | Status |
| --- | --- |
| KSP startup, loopback RPC and streams | Passed on 2026-11-06 |
| Windows 11 + WSL2 loopback to kRPC 0.7.0 | Passed on 2026-11-18 |
| API deployment of stock-derived test craft to Kerbin launchpad | Passed on 2026-11-06 |
| 10 s / 50 telemetry samples; lights toggle/readback/restoration | Passed on 2026-11-06 and 2026-11-18 |
| Scientific Python and training host | PyTorch CPU training completed on an Apple M5. Windows/WSL2 hosted the live kRPC checks. [Validation record](validation.md) |
| KSP adapter mechanics (`ksp_env/`) | Observation, named-save reset and a 10 Hz zero-action step passed live on the Mun orbit test profile. Provisional contact state is also exercised live on a synthetic 10 m debug snapshot; task-level touchdown detection still awaits an accepted descent initial-state library. [Adapter test profiles](ksp-env.md#adapter-test-profiles) |
| Debug low-altitude snapshot | `create_initial_save` saved a 10.05 m state with `can_save=False`, reloaded it, and measured task-frame tilt 0.0014°. Zero-throttle contact was detected live and classified as crash. Synthetic test only. [Snapshot validation](validation.md) |
| Shared Mun landing task | Qualitative contract adopted 2026-11-17. The 2026-11-25 decision freezes initial-state ranges, timing, safety limits and target radius. Site, descent save and vehicle calibration remain open. [Task contract](task-spec.md) |
| KSP powered descent | Descent snapshot, repeatable task reset, control-interval step and landing criteria await task-level validation. [Shared configuration](../configs/mun_landing.toml) |
| Standalone simulator pretraining | The S1 deterministic world model passed all four required gates on 50 held-out episodes. In the S2 vertical diagnostic, the clone and online policy each landed safely on 100/100 validation starts. [S1 and S2 results](validation.md) |
| Revised S3 nominal simulator policy | All three seeds achieved 100/100 safe landings at each of 13 online validations and 100/100 safe in-target at iteration 24. Validation selected seed 33, iteration 22, which scored 100/100 safe in-target on the fixed simulator test. The three iteration-24 timeout results belong to the superseded November 24 recipe. [Revised results](validation.md#complete-training-runs) · [Earlier training curves](validation.md#validation-across-the-fixed-24-cycle-budget) |
| KSP policy transfer and RQ1–RQ3 | Zero-shot landing and adaptation in KSP, the proposed 2×2 comparison, and frozen-policy physical generalization experiments remain pending. [Experiment protocol](experiment-protocol.md) |

The 2026-11-06 and 2026-11-18 launchpad evidence establishes the control
transport, not task-level lunar-landing readiness. KSP is not automatically
launched by offline setup or CI.

## Primary sources

- [kRPC 0.7.0 release and compatibility](https://github.com/krpc/krpc/releases/tag/v0.7.0)
- [kRPC 0.7.0 control-loop synchronization](https://krpc.github.io/krpc/0.7.0/tutorials/control-loops.html)
- [SpaceCenter API](https://krpc.github.io/krpc/0.7.0/python/api/space-center/space-center.html)
- [Vessel API](https://krpc.github.io/krpc/0.7.0/python/api/space-center/vessel.html)
- [Engine API](https://krpc.github.io/krpc/0.7.0/python/api/space-center/parts.html#Engine.thrust_limit)
- [Debug API](https://krpc.github.io/krpc/0.7.0/python/api/debug/debug.html)
- [Gymnasium Env contract](https://gymnasium.farama.org/api/env/)
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [MPS backend](https://docs.pytorch.org/docs/2.15/notes/mps.html)
- Package versions and wheel support were checked against the official [PyPI registry](https://pypi.org/) on 2026-11-09.
