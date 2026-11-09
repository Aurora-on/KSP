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

Direct versions are pinned in `requirements/runtime.txt`; `requirements/macos-arm64-py314.lock.txt` captures the tested local environment including transitive dependencies. This lock is a local snapshot, not a verified CUDA/Linux lock.

The current development machine is an Apple M5 with 16 GiB unified memory. KSP runs as x86_64 through Rosetta; Python runs natively as arm64 and communicates over TCP. CPU is the functional baseline, with MPS checked as an optional local accelerator. Full training throughput and memory requirements depend on the model and batch sizes, which remain open. A CUDA machine can be considered for larger experiments; CUDA is not required for the current local preparation.

## Installation

On macOS/Linux with an appropriate Python installed:

```sh
bash scripts/bootstrap.sh
```

On Windows, create a venv with `py -3.14 -m venv .venv`, then use `.venv\Scripts\python.exe -m pip install -r requirements/dev.txt`. Windows has not been tested by this repository.

Install KSP through its normal distribution. Download the [official kRPC 0.7.0 release](https://github.com/krpc/krpc/releases/tag/v0.7.0), verify the release digest, then merge its `GameData` into the KSP game directory while the game is closed. Preserve any existing settings and avoid duplicate ModuleManager versions. Do not commit the game, DLLs, saves or craft assets.

Known official ZIP SHA-256:

```text
cd4f062eaea6b74eff68a3be2dc3958a0f9373bafc2c9daee08aed6b988bf028
```

Use the kRPC server window to bind only `127.0.0.1`, RPC port `50000`, stream port `50001`. Load a save before connecting. The installed local setup already starts the service automatically inside a save. The main menu has no server. No global Mono/.NET installation, CKAN or kOS is required for this Python interface.

To include local plugin-file checks, provide the game root (the directory containing `GameData`), not the macOS app bundle:

```sh
export KSP_PATH="/path/to/Kerbal Space Program"
.venv/bin/python scripts/check_environment.py --require-ksp --output reports/environment.json
```

`KSP_PATH` is used only by the offline file check. The live doctor uses explicit address/port flags. There is no automatic `.env` loading.

## Requirements for a valid Mun landing environment

1. **Initial conditions and reset.** A named descent save must specify the lander, Mun location, target, position, velocity, attitude, angular velocity and fuel state. Use `SpaceCenter.save/load`, check `can_save` and `game_state_loads`, reacquire state and verify post-reset tolerances.
2. **Step timing.** Use kRPC 0.7 `hold_tick`, `next_tick`, `release_tick` for synchronized control where suitable; record actual UT deltas. Do not wait for streams while holding a tick, and always release in `finally`. Hold timeouts must be detected through measured timing. Wall-clock sleep and the existing 5 Hz CSV logger are not deterministic training steps.
3. **Observation/action contract.** Define position and velocity relative to the target, attitude convention, angular velocity, fuel and vehicle parameters in SI units. Keep frame transformations explicit. Choose how throttle and attitude/torque commands are shared across simulators. An autopilot inner loop changes the task dynamics and must be documented if used.
4. **Episode outcomes.** Define touchdown from contact/situation plus stable observation over multiple ticks, impact speeds, attitude, surviving vessel/parts and target error. Record safety, precision and propellant separately. Gymnasium `terminated` and `truncated` have different meanings; connection loss is an infrastructure error, not a successful landing.
5. **Metrics and data budgets.** Log transitions, actual simulator seconds, reset count, training and evaluation interactions, seeds, policy version, split and dataset hash. Do not count imagination as KSP data. All RQ2 variants use the same recorded KSP data and common evaluation protocol.
6. **Physical variations.** Mass and inertia are read-only telemetry, so generate measured fuel/payload/geometry variants and record their coupled changes. Engine `thrust_limit` can vary suitable liquid-engine thrust; it is distinct from the policy's throttle. `Debug.gravity_multiplier` modifies gravity globally for vessels; record and restore it, rather than assuming the read-only body's `surface_gravity` reflects the multiplier. Validate every OOD variant before using it.

Debug APIs may help build/reset scenarios. Freeze the intended physical variant and disable unrelated cheats such as infinite fuel during evaluation. Gravity variation is an explicit experimental condition. Test and restore global debug state in an isolated save.

## Acceptance boundaries

| Check | Status |
| --- | --- |
| KSP startup, loopback RPC and streams | Passed on 2026-11-06 |
| API deployment of stock-derived test craft to Kerbin launchpad | Passed on 2026-11-06 |
| 10 s / 50 telemetry samples; lights toggle/readback/restoration | Passed on 2026-11-06 |
| Current scientific Python imports and computation | See `validation.md` for the new check |
| Mun descent snapshot and lander | Not prepared |
| Deterministic step/reset and landing criteria | Not implemented or validated |
| Pretraining, world model, policy training, RQ1–RQ3 | Not implemented or run |

The 2026-11-06 launchpad evidence establishes the control transport, not task-level lunar-landing readiness. KSP is not automatically launched by offline setup or CI.

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
