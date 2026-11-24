"""Command-line entry points for the vertical simulator diagnostic."""
import json
from pathlib import Path

from .dataset import file_hash, save_dataset


def add_policy_commands(commands):
    prepare = commands.add_parser("prepare-vertical", help="collect vertical prefill and freeze a validation manifest")
    prepare.add_argument("--out", type=Path, required=True, help="new prefill directory")
    prepare.add_argument("--validation", type=Path, required=True, help="fixed vertical validation manifest")
    prepare.add_argument("--seed", type=int, default=230)
    prepare.add_argument("--episodes", type=int, default=200)
    prepare.set_defaults(handler=prepare_vertical)

    train = commands.add_parser("train-policy", help="train the simulator policy on imagined trajectories")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data", type=Path, required=True, help="permanent vertical prefill")
    train.add_argument("--world-model", type=Path, required=True, help="initial S1 checkpoint")
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True, help="run directory")
    train.add_argument("--resume", action="store_true", help="resume the latest complete iteration in --out")
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    train.add_argument("--seed", type=int, default=31)
    train.set_defaults(handler=train_policy_command)

    evaluate = commands.add_parser("evaluate-policy", help="evaluate a checkpoint on the fixed vertical diagnostic")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--validation", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True, help="new evaluation JSON file")
    evaluate.add_argument("--paired-returns", action="store_true", help="compare 32-step closed-loop returns")
    evaluate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    evaluate.set_defaults(handler=evaluate_policy)

    plot = commands.add_parser("plot", help="plot safe landing rate against simulator training transitions")
    plot.add_argument("--run", action="append", type=Path, required=True, help="run directory, repeatable")
    plot.add_argument("--out", type=Path, required=True, help="new PNG, PDF or SVG figure")
    plot.set_defaults(handler=plot_curves)


def prepare_vertical(args):
    from .vertical import collect_prefill, prepare_validation

    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = prepare_validation(args.validation)
    replay, metadata = collect_prefill(seed=args.seed, episodes=args.episodes)
    save_dataset(args.out, replay, metadata)
    return {"episodes": len(replay.episodes), "env_steps": metadata["env_steps"],
            "validation_manifest_sha256": manifest["sha256"], "out": str(args.out)}


def train_policy_command(args):
    from .fit import computation_device
    from .train import train_policy

    config = json.loads(args.config.read_text(encoding="utf-8"))
    return train_policy(config, args.data, args.world_model, args.validation, args.out, args.seed,
                        computation_device(args.device), resume=args.resume)


def evaluate_policy(args):
    from .fit import computation_device
    from .policy import load_policy
    from .train import load_training_world_model
    from .vertical import evaluate_vertical, load_validation, paired_returns

    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = load_validation(args.validation)
    device = computation_device(args.device)
    policy = load_policy(args.checkpoint, device=device)
    result = evaluate_vertical(policy, manifest)
    result["checkpoint_sha256"] = file_hash(args.checkpoint)
    if args.paired_returns:
        model = load_training_world_model(args.checkpoint, device=device)
        paired = paired_returns(model, policy.actor, policy, manifest, horizon=32)
        official = [(ep["start_id"], ep["status"]) for ep in result["episodes"]]
        observed = [(ep["start_id"], ep["status"]) for ep in paired["episode_outcomes"]]
        if official != observed:
            raise ValueError("paired simulator rollouts disagree with the official deterministic evaluation")
        result["paired_returns"] = paired
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(payload)
    return {key: result[key] for key in ("safe_landing_rate", "n_episodes", "status_counts", "checkpoint_sha256")}


def plot_curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    if args.out.exists():
        raise FileExistsError(args.out)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    try:
        for run in args.run:
            with (run / "validation.jsonl").open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if not rows:
                raise ValueError(f"no validation observations in {run}")
            ax.plot([r["env_steps"] for r in rows], [r["safe_landing_rate"] for r in rows],
                    marker="o", markersize=4, label=run.name)
        ax.axhline(.8, color="0.38", ls="--", lw=1, label="S2 threshold (80%)")
        ax.set(xlabel="Simulator training transitions (including prefill)", ylabel="Safe landing rate",
               title="Vertical landing on fixed validation starts", ylim=(-.04, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=160)
    finally:
        plt.close(fig)
    return {"out": str(args.out)}
