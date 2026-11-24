"""Command-line entry points for simulator-side learning."""
import argparse
import json
from pathlib import Path

from mun_lander import SimConfig

from .collect import collect
from .dataset import file_hash, load_dataset, save_dataset


def common_options(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, required=True)


def override(settings, assignments):
    for assignment in assignments:
        key, sep, value = assignment.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2 or parts[0] not in settings or parts[1] not in settings[parts[0]]:
            raise ValueError(f"expected a known section.key=value, got {assignment!r}")
        settings[parts[0]][parts[1]] = json.loads(value)


def report_deterministic(args, settings, device, parser):
    from .deterministic import load_deterministic
    from .deterministic_report import deterministic_report
    from .fit import load_world_model
    from .task_reward import reward_parity

    if args.rssm_checkpoint is None:
        parser.error("deterministic wm-report requires --rssm-checkpoint for its comparison")
    model, saved = load_deterministic(args.checkpoint, device)
    if settings != saved["config"]:
        parser.error("deterministic report configuration differs from checkpoint")
    rssm, baseline = load_world_model(args.rssm_checkpoint, device, SimConfig())
    replay, metadata = load_dataset(args.data)
    starts = {s["seed"] for s in metadata["manifest"]["starts"]}
    if args.training_diagnostic:
        role = "training"
        if metadata["replay_sha256"] != saved["train_replay_sha256"]:
            parser.error("training diagnostic requires the checkpoint's exact training replay")
    elif metadata["train_manifest_sha256"] == saved["dev_manifest_sha256"]:
        role = "development"
    else:
        role = "heldout"
        used = set(saved["train_reset_seeds"]) | set(saved["dev_reset_seeds"])
        if starts.intersection(used):
            parser.error("final evaluation must be disjoint from fitting and checkpoint selection")
    if role != "training" and starts.intersection(baseline["train_reset_seeds"]):
        parser.error("RSSM comparison overlaps the RSSM training split")
    result = deterministic_report(model, replay, saved["interface_spec"],
                                  provenance=metadata["episode_provenance"], train_mean=saved["train_mean"],
                                  rssm=rssm, rssm_spec=baseline["interface"])
    if replay.has_dt:
        result["reward_parity"] = reward_parity(replay, metadata, saved["interface_spec"])
    rssm_sha256 = file_hash(args.rssm_checkpoint)
    acceptance = result["acceptance"]
    acceptance["meets_gates_on_this_dataset"] = acceptance["pass"]
    matches = False
    if args.benchmark is not None:
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        matches = (metadata["train_manifest_sha256"] == benchmark["manifest_sha256"]
                   and metadata["replay_sha256"] in benchmark["replay_sha256"]
                   and rssm_sha256 == benchmark["rssm_checkpoint_sha256"])
        acceptance.update(benchmark=benchmark["name"], benchmark_sha256=file_hash(args.benchmark))
    acceptance["matches_benchmark"] = matches
    acceptance["pass"] = acceptance["meets_gates_on_this_dataset"] and role == "heldout" and matches
    result.update(dataset_role=role, fit_seed=saved["seed"], updates=saved["updates"],
                  collection_seed=metadata["seed"], device=str(device),
                  checkpoint_sha256=file_hash(args.checkpoint),
                  rssm_checkpoint_sha256=rssm_sha256,
                  train_manifest_sha256=saved["train_manifest_sha256"],
                  dev_manifest_sha256=saved["dev_manifest_sha256"],
                  dataset_manifest_sha256=metadata["train_manifest_sha256"],
                  dataset_replay_sha256=metadata["replay_sha256"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mun-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("collect", help="collect nominal simulator episodes on CPU")
    common_options(gather)
    gather.add_argument("--episodes", type=int, help="override the collection size")
    train = commands.add_parser("train", help="fit the RSSM baseline offline on collected replay")
    common_options(train)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--set", action="append", default=[], metavar="section.key=value")
    fit_det = commands.add_parser("fit-deterministic", help="fit the residual model and select on development replay")
    common_options(fit_det)
    fit_det.add_argument("--data", type=Path, required=True, help="training collection")
    fit_det.add_argument("--dev-data", type=Path, required=True, help="disjoint checkpoint-selection collection")
    report = commands.add_parser("wm-report", help="evaluate world-model open-loop predictions")
    common_options(report)
    report.add_argument("--checkpoint", type=Path, required=True)
    report.add_argument("--rssm-checkpoint", type=Path, help="RSSM baseline for the deterministic comparison")
    report.add_argument("--benchmark", type=Path, help="frozen replay and RSSM hashes for formal S1 acceptance")
    report.add_argument("--data", type=Path, required=True, help="independent held-out collection")
    report.add_argument("--training-diagnostic", action="store_true",
                        help="score the checkpoint's exact training replay and label it as training")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "wm-report" and "config" in settings and "code" in settings:
        settings = settings["config"]
    sections = {"collect", "model", "fit"}
    deterministic = isinstance(settings, dict) and settings.get("model_type") == "deterministic"
    if args.command == "fit-deterministic" and not deterministic:
        parser.error("fit-deterministic requires the deterministic model configuration")
    if not deterministic and (not isinstance(settings, dict) or not set(settings) <= sections
            or not all(isinstance(value, dict) for value in settings.values())):
        parser.error("expected a JSON object with collect, model or fit sections")
    required = {"collect": {"collect"}, "train": {"model", "fit"}, "wm-report": {"model"}}
    if deterministic and args.command not in ("fit-deterministic", "wm-report"):
        parser.error("this command requires a collection or RSSM configuration")
    if not deterministic and (missing := required[args.command].difference(settings)):
        parser.error(f"missing configuration sections: {sorted(missing)}")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.command == "collect":
        config = settings["collect"].copy()
        if args.episodes is not None:
            config["episodes"] = args.episodes
        replay, metadata = collect(seed=args.seed, **config)
        save_dataset(args.out, replay, metadata)
        result = {"episodes": len(replay.episodes), "env_steps": metadata["env_steps"],
                  "manifest_sha256": metadata["train_manifest_sha256"],
                  "status_counts": metadata["status_counts"], "device": "cpu"}
    else:
        from .fit import computation_device, fit, load_world_model
        device = computation_device(args.device)
        if args.command == "train":
            override(settings, args.set)
            result = fit(settings, args.data, args.out, args.seed, device)
        elif args.command == "fit-deterministic":
            from .deterministic_fit import fit_deterministic
            result = fit_deterministic(args.data, args.dev_data, args.out, settings, args.seed, device)
        elif deterministic:
            result = report_deterministic(args, settings, device, parser)
        else:
            if args.rssm_checkpoint is not None or args.benchmark is not None:
                parser.error("--rssm-checkpoint and --benchmark are only used for a deterministic comparison")
            from .wm_report import open_loop_report
            model, checkpoint = load_world_model(args.checkpoint, device, SimConfig())
            defaults = dict(ground_history=False, event_weight=1., free_nats=1.,
                            dynamics_scale=1., representation_scale=.1)
            if (defaults | settings["model"]) != (defaults | checkpoint["config"]["model"]):
                parser.error("report model configuration differs from checkpoint")
            replay, metadata = load_dataset(args.data)
            starts = {s["seed"] for s in metadata["manifest"]["starts"]}
            if args.training_diagnostic and metadata["replay_sha256"] != checkpoint["train_replay_sha256"]:
                parser.error("training diagnostic requires the checkpoint's exact training replay")
            if not args.training_diagnostic and starts.intersection(checkpoint["train_reset_seeds"]):
                parser.error("report requires held-out episodes with disjoint reset seeds")
            result = open_loop_report(model, replay, checkpoint["interface"],
                                      train_mean=checkpoint.get("train_mean"),
                                      provenance=metadata["episode_provenance"])
            role = "training" if args.training_diagnostic else "heldout"
            result["acceptance"]["pass"] &= not args.training_diagnostic
            result.update(dataset_role=role, updates=checkpoint["updates"], device=str(device), checkpoint_sha256=file_hash(args.checkpoint),
                          fit_seed=checkpoint["seed"], collection_seed=metadata["seed"],
                          train_manifest_sha256=checkpoint["train_manifest_sha256"],
                          **{f"{role}_manifest_sha256": metadata["train_manifest_sha256"],
                             f"{role}_replay_sha256": metadata["replay_sha256"]})
        if args.command == "wm-report":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
            with args.out.open("x", encoding="utf-8") as output:
                output.write(payload)
    print(json.dumps(dict(result, out=str(args.out)), allow_nan=False))
