"""Plot M2R diagnostic metrics against optimizer updates.

From the repository root, with matplotlib installed:
    python agent/scripts/plot_wm_quality.py --study agent/runs/m2r-study
Writes quality-curve.png and quality-curve.svg beside the run directories.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter


RUNS = (
    ("m2r-short-A", "Short A", "#2166a4", "o", "-"),
    ("m2r-short-B", "Short B · weighted pilot", "#a24f14", "s", ":"),
    ("m2r-short-C", "Short C", "#8260a2", "D", "-"),
    ("m2r-long-A", "Long A", "#197757", "^", "--"),
    ("m2r-long-C", "Long C", "#197757", "^", "--"),
)
POSITION, VELOCITY, TILT = "position_error_m", "velocity_error_m_s", "tilt_mae_deg"


def read_runs(study, long_run):
    runs = {}
    sample = None
    for name, *_ in RUNS:
        if name.startswith("m2r-long-") and long_run and name != f"m2r-long-{long_run}":
            continue
        points = {}
        for path in sorted((study / name).glob("*-dev.json")):
            with path.open() as stream:
                report = json.load(stream)
            role = report.get("split", report.get("dataset_role"))
            if role not in ("dev", "heldout"):
                raise ValueError(f"{path} is not a diagnostic-set report")
            signature = (report["episodes"],
                         report.get("data_sha256") or report["heldout_replay_sha256"],
                         report.get("manifest_sha256") or report["heldout_manifest_sha256"])
            if sample is not None and signature != sample:
                raise ValueError(f"{path} uses a different diagnostic set")
            sample = signature
            update, gates = report["updates"], report["gates"]
            if update in points and points[update] != gates:
                raise ValueError(f"conflicting reports for {name}, update {update}")
            points[update] = gates
        if points:
            runs[name] = sorted(points.items())
    if "m2r-long-A" in runs and "m2r-long-C" in runs:
        raise ValueError("select the formal fit with --long-run A or --long-run C")
    if not runs:
        raise ValueError(f"no *-dev.json reports found in {study}")
    return runs, sample[0]


def metric(gates, gate, field, horizon):
    data = gates[gate]
    if gate == "G2":
        pair = data["values"][str(horizon)][field]
        return pair["model"], pair["persistence"]
    if gate == "G4":
        return data["groups"][field]["rate"], data["required_rate"]
    if gate == "G5":
        return data["rate"], data["required_rate"]
    return data["values"][field], data["limits"][field]


def plot(study, long_run=None):
    runs, episodes = read_runs(study, long_run)
    panels = (
        ("G1", POSITION, 0, "Posterior position", "m"),
        ("G1", VELOCITY, 0, "Posterior velocity", "m/s"),
        ("G1", TILT, 0, "Posterior tilt", "deg"),
        ("G4", "safe_landing", 0, "Terminal reward · safe landing", ""),
        ("G2", POSITION, 5, "5-step position", "m"),
        ("G2", VELOCITY, 5, "5-step velocity", "m/s"),
        ("G3", VELOCITY, 15, "Near-ground 15-step velocity", "m/s"),
        ("G4", "crash", 0, "Terminal reward · crash", ""),
        ("G2", POSITION, 15, "15-step position", "m"),
        ("G2", VELOCITY, 15, "15-step velocity", "m/s"),
        ("G3", TILT, 15, "Near-ground 15-step tilt", "deg"),
        ("G5", None, 15, "Final 15-step reward sum", ""),
    )
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 9.2), sharex=True)
    fig.subplots_adjust(left=.055, right=.986, top=.845, bottom=.096, wspace=.28, hspace=.50)
    fig.suptitle("Mun lander world-model quality", x=.055, y=.983, ha="left",
                 fontsize=23, fontweight="bold")
    fig.text(.055, .923, f"{episodes} diagnostic episodes   ·   Near-ground endpoints below 10 m",
             fontsize=12, color="#414b52")
    handles = [Line2D([], [], label=label, color=color, ls=line, marker=marker, ms=6,
                      markerfacecolor="white" if name.startswith("m2r-long-") else color)
               for name, label, color, marker, line in RUNS if name in runs]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.991, .981), ncols=len(handles),
               frameon=False, handlelength=2.1, fontsize=11)
    xmax = max(update for rows in runs.values() for update, _ in rows)
    tick_pool = [0, 1000, 2000, 5000, 10000, 30000, 100000]
    xticks = [t for t in tick_pool if t <= max(xmax * 1.1, 1)]
    if len(xticks) == 1 and xmax > 0:
        xticks.append(xmax)
    for ax, (gate, field, horizon, title, unit) in zip(axes.flat, panels, strict=True):
        bound = None
        values = []
        for name, _, color, marker, line in RUNS:
            if name not in runs:
                continue
            xs, ys = [], []
            for update, gates in runs[name]:
                y, this_bound = metric(gates, gate, field, horizon)
                if bound is not None and not math.isclose(this_bound, bound, rel_tol=1e-6):
                    raise ValueError(f"{gate} {field} reference varies between reports")
                bound = this_bound
                xs.append(update)
                ys.append(float("nan") if y is None else y)
            ax.plot(xs, ys, color=color, marker=marker, ls=line, lw=1.8, ms=4.5,
                    markerfacecolor="white" if name.startswith("m2r-long-") else color, zorder=3)
            values.extend(y for y in ys if math.isfinite(y))
        ax.axhline(bound, ls=(0, (4, 3)), lw=1.15, color="#596167", zorder=2)
        if gate in ("G4", "G5"):
            ax.set_ylim(-.04, 1.07)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_ylabel("Episodes meeting criterion")
            bound_label = f"Minimum {bound:.0%}"
        else:
            ax.set_ylabel(f"Error ({unit})")
            vals = [bound, *values]
            if min(vals) > 0:
                ax.set_yscale("log")
                low, high = min(vals) / 1.35, max(vals) * 1.35
                ticks = [m * 10**e for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
                         for m in (1, 2, 5) if low <= m * 10**e <= high]
                ax.set_ylim(low, high)
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ref = "Persistence" if gate == "G2" else "Maximum"
            bound_label = f"{ref} {bound:.3g} {unit}"
        ax.set_title(f"{gate}   {title}", loc="left", fontsize=11, fontweight="bold", pad=16)
        ax.text(0, 1.018, bound_label, ha="left", va="bottom", transform=ax.transAxes,
                color="#495158", fontsize=9)
        ax.set_xscale("symlog", linthresh=1000, linscale=.6)
        ax.set_xlim(-55, max(xmax * 1.1, 100))
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t/1000:g}k" if t >= 1000 else f"{t:g}"))
        ax.grid(axis="y", alpha=.22)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9da3a9")
        ax.tick_params(labelsize=9)
    fig.supxlabel("Optimizer updates   ·   Linear through 1,000, logarithmic above", y=.024, fontsize=11)
    for extension in ("png", "svg"):
        fig.savefig(study / f"quality-curve.{extension}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("agent/runs/m2r-study"))
    parser.add_argument("--long-run", choices=("A", "C"), help="formal long fit if both A and C exist")
    args = parser.parse_args()
    plot(args.study, args.long_run)
