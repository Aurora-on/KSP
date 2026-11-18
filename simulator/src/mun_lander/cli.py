"""Command-line validation and demonstration utilities (no policy training)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import SimConfig
from .evaluation import benchmark, evaluate, generate_initial_states, parameter_sweep, rollout
from .math3d import rotation_matrix


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_trajectory(output: Path, trajectory: dict, result: dict, config: SimConfig, gif=True):
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", {**result, "config": config.to_dict()})
    np.savez_compressed(output / "trajectory.npz", **trajectory)
    obs = trajectory["observations"]
    columns = ["time", "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
               "wx", "wy", "wz", "main_fuel_remaining", "main_fuel_used", "rcs_fuel_used",
               "next_throttle", "next_rcs_x", "next_rcs_y", "next_rcs_z", "next_reward",
               "next_dt", "next_terminated", "next_truncated"]
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, observation in enumerate(obs):
            action = trajectory["actions"][i].tolist() if i < len(obs)-1 else [""] * 4
            reward = trajectory["rewards"][i] if i < len(obs)-1 else ""
            ending = [trajectory[key][i] for key in ("dt", "terminated", "truncated")] if i < len(obs)-1 else [""] * 3
            writer.writerow([trajectory["times"][i], *observation.tolist(),
                             trajectory["main_fuel_used"][i], trajectory["rcs_fuel_used"][i],
                             *action, reward, *ending])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires optional dependencies: pip install -e '.[demo]'") from exc
    times = trajectory["times"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0, 0].plot(obs[:, 0], obs[:, 1], color="#2463a4")
    axes[0, 0].scatter([obs[0, 0]], [obs[0, 1]], label="Start", color="#2463a4")
    axes[0, 0].scatter([obs[-1, 0]], [obs[-1, 1]], marker="x", label="Finish", color="#bf5b17")
    axes[0, 0].add_patch(plt.Circle((0, 0), config.landing.target_radius, fill=False,
                                 linestyle="--", color="#42804c", label="Target radius"))
    axes[0, 0].set(xlabel="Target-relative x (m)", ylabel="Target-relative y (m)", title="Ground track")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(times, obs[:, 2], color="#2463a4")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Clearance (m)", title="Descent")
    axes[1, 0].plot(times, np.linalg.norm(obs[:, 3:6], axis=1), label="Total speed")
    axes[1, 0].plot(times, obs[:, 5], label="Vertical velocity")
    axes[1, 0].axhline(config.landing.max_touchdown_speed, color="#a65238", linestyle="--", label="Touchdown speed limit")
    axes[1, 0].set(xlabel="Time (s)", ylabel="m/s", title="Velocity")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(times, trajectory["main_fuel_used"], label="Main engine")
    axes[1, 1].plot(times, trajectory["rcs_fuel_used"], label="RCS")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Consumed mass (kg)", title="Propellant use")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    controller_name = result.get("controller", "scripted").capitalize()
    figure.suptitle(f"{controller_name} landing | {result['status']} | error {result['horizontal_error']:.2f} m")
    figure.savefig(output / "trajectory.png", dpi=160)
    plt.close(figure)
    if not gif:
        return
    from matplotlib.animation import FuncAnimation, PillowWriter
    figure, axes = plt.subplots(1, 2, figsize=(9, 5), layout="constrained")
    artists = []
    for axis_index, ax in enumerate(axes):
        horizontal = obs[:, axis_index]
        extent = max(15.0, float(np.max(np.abs(horizontal))) + 6)
        ax.plot(horizontal, obs[:, 2], color="#9bb9cf", linewidth=1)
        ax.axhline(0, color="#646b75", linewidth=2)
        ax.plot([-config.landing.target_radius, config.landing.target_radius], [0, 0], color="#42804c", linewidth=5)
        ax.set(xlim=(-extent, extent), ylim=(-3, max(obs[:, 2])*1.08+3),
               xlabel=f"Target-relative {'xy'[axis_index]} (m)", ylabel="Clearance (m)")
        ax.grid(alpha=0.2)
        marker, = ax.plot([], [], "o", color="#bf5b17", markersize=6)
        axis_line, = ax.plot([], [], color="#bf5b17", linewidth=3)
        artists.append((marker, axis_line))
    label = figure.suptitle("")
    frames = np.unique(np.r_[np.arange(0, len(obs), 5), len(obs)-1])

    def update(i):
        thrust_axis = rotation_matrix(obs[i, 6:10])[:, 2]
        for axis_index, (marker, axis_line) in enumerate(artists):
            x, z = obs[i, axis_index], obs[i, 2]
            marker.set_data([x], [z])
            axis_line.set_data([x, x+5*thrust_axis[axis_index]], [z, z+5*thrust_axis[2]])
        phase = result["status"] if i == len(obs)-1 else "powered descent"
        label.set_text(f"{controller_name} reference | t = {times[i]:.1f} s | {phase}\n"
                       f"speed = {np.linalg.norm(obs[i, 3:6]):.2f} m/s | fuel used = "
                       f"{trajectory['main_fuel_used'][i] + trajectory['rcs_fuel_used'][i]:.1f} kg")
        return [label, *[item for pair in artists for item in pair]]

    animation = FuncAnimation(figure, update, frames=frames, interval=80, blit=False)
    animation.save(output / "landing.gif", writer=PillowWriter(fps=12), dpi=80)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mun 6-DOF simulator utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parsers = {}
    for command in ("demo", "starts", "evaluate", "sweep", "benchmark"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, help="JSON config; defaults are engineering placeholders")
        sub.add_argument("--output", type=Path)
        sub.add_argument("--seed", type=int, default=2026 if command == "starts" else 7)
        parsers[command] = sub
    parsers["demo"].add_argument("--no-gif", action="store_true")
    parsers["demo"].add_argument("--controller", choices=("scripted", "zero", "random"), default="scripted")
    parsers["starts"].add_argument("--split", choices=("train", "adaptation", "validation", "test", "generalization"), default="test")
    parsers["starts"].add_argument("--count", type=int, default=30)
    for command in ("evaluate", "sweep"):
        parsers[command].add_argument("--starts", type=Path, required=True)
        parsers[command].add_argument("--controller", choices=("scripted", "zero", "random"), default="scripted")
    parsers["sweep"].add_argument("--scales", type=float, nargs="+", default=(0.8, 1.0, 1.2))
    parsers["benchmark"].add_argument("--steps", type=int, default=2000)
    args = parser.parse_args(argv)
    config = SimConfig.from_json(args.config) if args.config else SimConfig()
    if args.command == "demo":
        output = args.output or Path("artifacts/demo")
        result, trajectory = rollout(config, seed=args.seed, controller=args.controller, record=True)
        save_trajectory(output, trajectory, result, config, gif=not args.no_gif)
        print(json.dumps({"output": str(output.resolve()), **result}, indent=2))
        return
    if args.command == "starts":
        result = generate_initial_states(config, args.count, args.seed, args.split)
    elif args.command == "benchmark":
        result = benchmark(config, args.steps, args.seed)
    else:
        manifest = json.loads(args.starts.read_text(encoding="utf-8"))
        if args.command == "evaluate":
            result = evaluate(config, manifest, args.controller)
        else:
            result = parameter_sweep(config, manifest, args.scales, args.controller)
    output = args.output or Path(f"artifacts/{args.command}.json")
    _write_json(output, result)
    concise = {k: v for k, v in result.items()
               if k not in {"episodes", "runs", "starts", "config", "controller_config", "initial_config"}}
    print(json.dumps({"output": str(output.resolve()), **concise}, indent=2))


if __name__ == "__main__":
    main()
