# Validation record

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
