"""Command-line entry points for simulator-side learning."""
import argparse
import json
from pathlib import Path

from .collect import collect
from .dataset import save_dataset


def common_options(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mun-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("collect", help="collect nominal simulator episodes on CPU")
    common_options(gather)
    gather.add_argument("--episodes", type=int, help="override the collection size")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = settings["collect"].copy()
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    replay, metadata = collect(seed=args.seed, **config)
    save_dataset(args.out, replay, metadata)
    print(json.dumps({"episodes": len(replay.episodes), "env_steps": metadata["env_steps"],
                      "manifest_sha256": metadata["train_manifest_sha256"],
                      "device": "cpu", "out": str(args.out)}))
