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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mun-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("collect", help="collect nominal simulator episodes on CPU")
    common_options(gather)
    gather.add_argument("--episodes", type=int, help="override the collection size")
    train = commands.add_parser("train", help="fit the world model offline on collected replay")
    common_options(train)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--set", action="append", default=[], metavar="section.key=value")
    report = commands.add_parser("wm-report", help="evaluate world-model open-loop predictions")
    common_options(report)
    report.add_argument("--checkpoint", type=Path, required=True)
    report.add_argument("--data", type=Path, required=True, help="independent held-out collection")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    sections = {"collect", "model", "fit"}
    if (not isinstance(settings, dict) or not set(settings) <= sections
            or not all(isinstance(value, dict) for value in settings.values())):
        parser.error("expected a JSON object with collect, model or fit sections")
    required = {"collect": {"collect"}, "train": {"model", "fit"}, "wm-report": {"model"}}
    if missing := required[args.command].difference(settings):
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
        else:
            from .wm_report import open_loop_report
            model, checkpoint = load_world_model(args.checkpoint, device, SimConfig())
            if settings["model"] != checkpoint["config"]["model"]:
                parser.error("report model configuration differs from checkpoint")
            replay, metadata = load_dataset(args.data)
            starts = {s["seed"] for s in metadata["manifest"]["starts"]}
            if starts.intersection(checkpoint["train_reset_seeds"]):
                parser.error("report requires held-out episodes with disjoint reset seeds")
            result = open_loop_report(model, replay, checkpoint["interface"],
                                      train_mean=checkpoint.get("train_mean"),
                                      provenance=metadata["episode_provenance"])
            result.update(device=str(device), checkpoint_sha256=file_hash(args.checkpoint),
                          fit_seed=checkpoint["seed"], collection_seed=metadata["seed"],
                          train_manifest_sha256=checkpoint["train_manifest_sha256"],
                          heldout_manifest_sha256=metadata["train_manifest_sha256"],
                          heldout_replay_sha256=metadata["replay_sha256"])
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as output:
                output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(result, out=str(args.out)), allow_nan=False))
