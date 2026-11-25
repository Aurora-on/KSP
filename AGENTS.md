# AGENTS.md

## Project status

The standalone 6DOF simulator in `simulator/` provides the landing environment,
scripted controller and evaluation tools. The `mun-agent` package provides simulator observation transforms,
uniform transition sampling, demonstration collection, deterministic residual dynamics, event and reward evaluation, and
an offline RSSM comparison. S1 passed its four required gates on 50 held-out simulator episodes. S2 passed its vertical simulator gate with 100/100 safe landings after a behavior-cloning warm start and one online imagination iteration. Cloning alone also scored 100/100 on the same starts. The revised S3 recipe kept every scheduled validation at 100/100 safe across three seeds. All three round-24 policies landed safely inside the target on all 100 validation starts. The validation-selected seed-33 round-22 policy achieved 100/100 safe in-target on the fixed test and improved raw mean return over its clone. KSP landing integration and the RQ1–RQ3 experiments are pending.
Do not claim results that are not recorded in `docs/validation.md`.

## Read first

- `README.md`
- `docs/task-spec.md`
- `docs/experiment-protocol.md`
- `docs/roadmap.md`

## Hard rules

- Numeric fields marked `TBD` are open decisions, not guesses. Do not fill
  thresholds, initial-state ranges, target radius, budgets, OOD ranges or
  hardware choices without an explicit team decision.
- Keep `initial_state_ready`, `success_thresholds_ready` and
  `reset_step_validated` false until the corresponding checks pass.
- KSP live checks are read-only by default. `--control-test` must remain
  reversible. Never run KSP-dependent checks in CI or without a running game.
- Do not commit KSP game files, saves, craft files, PPTX binaries, datasets,
  checkpoints or generated reports. `downloads/` and `reports/` are ignored.
- Keep `configs/mun_landing.toml`, the task documents and the tests consistent.
  If an interface or task decision changes, update the documents and config first.
- Use SI units and explicit frames. Follow the observation/action contract in
  `docs/task-spec.md`; do not hide an autopilot inside the environment.
- Python 3.12-3.14 is required. Do not add dependencies without updating the
  pinned requirement files.
- Treat imagined transitions as separate from KSP data. RQ2 comparisons use the
  same frozen KSP dataset and identical splits.

## Verify

From the repository root, use an environment with both packages and their test
dependencies installed as described in [the agent README](agent/README.md).
Run these checks before committing code or configuration changes.

```sh
python3 -m compileall -q ksp_env scripts tests
python3 -m unittest discover -s tests -v
python -m pytest simulator/tests -q
python -m pytest agent/tests -q
```

For live KSP checks, follow `README.md` and use only the documented flags.
Do not run `--control-test` without entering a vessel in the Flight scene.
