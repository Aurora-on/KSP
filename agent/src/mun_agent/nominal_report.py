"""Select online checkpoints by validation alone and compare frozen policies on both splits."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .dataset import file_hash
from .nominal import LEGACY_SELECTION_ORDER, SELECTION_ORDER, evaluate_nominal, load_manifest, validation_rank
from .policy import load_policy


def select_candidate(candidates):
    # Archived cohorts reproduce their original freeze. For a mixed cohort,
    # compare each run's published best under the current validation protocol.
    historical = all(tuple(c["selection_order"]) == LEGACY_SELECTION_ORDER for c in candidates)
    order = LEGACY_SELECTION_ORDER if historical else SELECTION_ORDER
    selected = max(candidates, key=lambda c: validation_rank(c["best"], order))
    return selected, list(order)


def read_candidate(run, validation_sha):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines() if line]
    if config["config"]["task"] != "nominal" or config["validation_sha256"] != validation_sha:
        raise ValueError(f"{run} did not use the supplied nominal validation manifest")
    if any(row["starts_sha256"] != validation_sha for row in rows):
        raise ValueError(f"validation manifest changed during {run}")
    online = [row for row in rows if row.get("phase") == "online"]
    clones = [row for row in rows if row.get("phase") == "bc_only"]
    if len(clones) != 1 or not online:
        raise ValueError(f"{run} needs one clone baseline and at least one online validation")
    order = tuple(summary.get("selection_order", config.get("selection_order", LEGACY_SELECTION_ORDER)))
    if "selection_order" in config and tuple(config["selection_order"]) != order:
        raise ValueError(f"conflicting selection orders recorded in {run}")
    best = max(online, key=lambda row: validation_rank(row, order))
    if best["iteration"] != summary["best_iteration"]:
        raise ValueError(f"stored best checkpoint disagrees with the validation log in {run}")
    if summary["iteration"] < config["config"]["iterations"] and not summary.get("time_budget_exhausted"):
        raise ValueError(f"{run} has not completed its fixed training budget")
    target_rates = [r.get("safe_in_target_rate") for r in online]
    minimum_target = min(target_rates) if all(v is not None for v in target_rates) else None
    return dict(seed=config["seed"], run=str(run), selection_order=list(order), best=best, clone=clones[0], last=online[-1],
                min_online_safe_in_target_rate=minimum_target,
                completed_iterations=summary["iteration"], elapsed_s=summary["elapsed_s"],
                training_config=config["config"], prefill_replay_sha256=config["prefill_replay_sha256"],
                world_model_sha256=config["pretrained_sha256"])


def gates(comparisons, seed_count, official_size):
    per_split = {}
    for split, results in comparisons.items():
        final, clone = results["online"], results["bc_only"]
        per_split[split] = dict(safe_at_least_80=final["safe_landing_rate"] >= .8,
                                safety_matches_or_exceeds_clone=final["safe_landing_rate"] >= clone["safe_landing_rate"],
                                mean_return_above_clone=final["mean_return"] > clone["mean_return"])
    test = comparisons["test"]["online"]
    formal = official_size and seed_count >= 3
    return dict(fixed_100_starts_per_split=official_size, three_seeds=seed_count >= 3, by_split=per_split,
                qualified=formal and all(all(checks.values()) for checks in per_split.values()),
                test_target=test["safe_landing_rate"] >= .95 and test["safe_in_target_rate"] >= .90)


def render_report(report):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    def rate(value):
        return "—" if value is None else f"{value:.0%}"

    selected = report["selected"]
    lines = ["# Nominal simulator landing", "",
             f"Validation selected seed {selected['seed']} at online iteration {selected['iteration']}.",
             "Validation uses the following priority order. " + " → ".join(selected["selection_order"]) + ".",
             "", "| Seed | BC safe | BC return | Best iteration | Best safe | Best in target | Best return | Last safe | Last in target | Min online in target |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        bc, best = run["clone"], run["best"]
        lines.append(f"| {run['seed']} | {bc['safe_landing_rate']:.0%} | {bc['mean_return']:.3f} | "
                     f"{best['iteration']} | {best['safe_landing_rate']:.0%} | {rate(best.get('safe_in_target_rate'))} | "
                     f"{best['mean_return']:.3f} | {run['last']['safe_landing_rate']:.0%} | "
                     f"{rate(run['last'].get('safe_in_target_rate'))} | {rate(run.get('min_online_safe_in_target_rate'))} |")
    lines.extend(["", "| Split | Policy | Safe | In target | Safe landing error (m) | Fuel (kg) | Flight (s) | First impact (m/s) | Return |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for split, methods in report["evaluations"].items():
        for method, row in methods.items():
            metrics = [row[field] for field in ("mean_landing_error_safe", "mean_total_fuel_used_kg",
                       "mean_flight_time_s", "mean_touchdown_speed_m_s", "mean_return")]
            lines.append(f"| {split} | {method} | {row['safe_landing_rate']:.0%} | {row['safe_in_target_rate']:.0%} | "
                         + " | ".join(map(number, metrics)) + " |")
    lines += ["", "Landing error averages safe episodes. Impact speed averages episodes with a first contact. "
              "Fuel, flight duration and return average all starts. Rates use the full manifest.", "",
              f"Qualification **{report['gates']['qualified']}**. Test target **{report['gates']['test_target']}**."]
    for method, check in report["validation_recheck"].items():
        if not check["same_status_counts"] or check.get("same_target_rate") is False or abs(check["mean_return_delta"]) > 1e-5:
            lines.append(f"Frozen {method} validation recheck differs from the training log. "
                         f"Matching outcome counts {check['same_status_counts']}, "
                         f"matching target rate {check.get('same_target_rate')}, mean return difference {check['mean_return_delta']:.6f}.")
    return "\n".join(lines) + "\n"


def save_json(path, data):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def make_report(args):
    import torch
    from .fit import computation_device

    validation = load_manifest(args.validation, "validation")
    runs = list(args.run or [])
    if args.checkpoint is not None and not runs:
        runs = [args.checkpoint.parent.parent]
    if not runs:
        raise ValueError("provide the completed run directories with --run or the selected --checkpoint")
    candidates = sorted((read_candidate(run, validation["sha256"]) for run in runs), key=lambda c: c["seed"])
    if len({c["seed"] for c in candidates}) != len(candidates):
        raise ValueError("each listed run must have a distinct training seed")
    selected, order = select_candidate(candidates)
    path = Path(selected["run"]) / "checkpoints"
    paths = {"bc_only": path / "bc-only.pt", "online": path / "best.pt"}
    if args.checkpoint and args.checkpoint.resolve() != paths["online"].resolve():
        raise ValueError("the specified checkpoint differs from the validation-selected winner")
    checksums = {name: file_hash(p) for name, p in paths.items()}
    for name, checkpoint in paths.items():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["provenance"]["seed"] != selected["seed"]
                or saved["provenance"]["validation_sha256"] != validation["sha256"]
                or saved["actor_spec"]["vertical_only"]
                or saved["progress"]["iteration"] != (0 if name == "bc_only" else selected["best"]["iteration"])):
            raise ValueError(f"{name} checkpoint disagrees with its validation provenance")
        del saved
    # Open test only after selecting and verifying the frozen winner.
    test = load_manifest(args.test, "test")
    if {s["seed"] for s in validation["starts"]}.intersection(s["seed"] for s in test["starts"]):
        raise ValueError("validation and test starts overlap")
    out = args.out
    out.mkdir(parents=True, exist_ok=args.resume)
    provenance = dict(validation_sha256=validation["sha256"], test_sha256=test["sha256"],
                      seed=selected["seed"], checkpoint_sha256=checksums, selection_order=order,
                      candidate_logs={str(c["seed"]): file_hash(Path(c["run"]) / "validation.jsonl") for c in candidates},
                      device=str(computation_device(args.device)))
    source = out / "selection.json"
    if source.is_file():
        archived = json.loads(source.read_text())
        archived.setdefault("selection_order", list(LEGACY_SELECTION_ORDER))
        if archived != provenance:
            raise ValueError("report inputs have changed since selection")
    else:
        if any(out.iterdir()):
            raise ValueError("cannot resume in a directory without the original selection")
        save_json(source, provenance)
    selected_info = dict(seed=selected["seed"], iteration=selected["best"]["iteration"], selection_order=order,
                         checkpoints={k: str(p) for k, p in paths.items()}, checkpoint_sha256=checksums)
    report = dict(selected=selected_info, runs=candidates, evaluations={}, validation_recheck={})
    device = computation_device(args.device)
    policies = {method: load_policy(checkpoint, device) for method, checkpoint in paths.items()}
    for split, manifest in (("validation", validation), ("test", test)):
        evaluations = {}
        for method, policy in {**policies, "scripted": "scripted"}.items():
            cached = out / f"{split}-{method}.json"
            if cached.is_file():
                value = json.loads(cached.read_text())
                if value["starts_sha256"] != manifest["sha256"] or value["checkpoint_sha256"] != checksums.get(method):
                    raise ValueError(f"previous evaluation has different inputs: {cached}")
            else:
                value = evaluate_nominal(policy, manifest)
                value["checkpoint_sha256"] = checksums.get(method)
                save_json(cached, value)
            if split == "validation" and method in paths:
                ref = selected["clone" if method == "bc_only" else "best"]
                report["validation_recheck"][method] = {
                    "same_status_counts": value["status_counts"] == ref["status_counts"],
                    "same_target_rate": (value["safe_in_target_rate"] == ref["safe_in_target_rate"]
                                         if "safe_in_target_rate" in ref else None),
                    "mean_return_delta": value["mean_return"] - ref["mean_return"]}
            evaluations[method] = value
        report["evaluations"][split] = evaluations
    official_size = all(len(m["starts"]) == 100 and m["seed"] == 2026 for m in (validation, test))
    report["gates"] = gates(report["evaluations"], len(candidates), official_size)
    save_json(out / "report.json", report)
    (out / "report.md").write_text(render_report(report))
    return dict(out=str(out), selected=selected_info, gates=report["gates"])
