# AGENTS.md

## Project status

Research scaffolding for world models for Mun landing in Kerbal Space Program.
No landing environment, training loop or RQ1-RQ3 experiment has been implemented.
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

Run these before committing code or configuration changes:

```sh
python3 -m compileall -q ksp_env scripts tests
python3 -m unittest discover -s tests -v
```

For live KSP checks, follow `README.md` and use only the documented flags.
Do not run `--control-test` without entering a vessel in the Flight scene.
