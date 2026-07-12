#!/usr/bin/env python
"""Plot v69 MAPPO reward/loss curves from TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb-root", default="outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn")
    parser.add_argument("--out-dir", default="outputs/v69_training_curves")
    return parser.parse_args()


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_event_file(path: Path) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(str(path), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        data[tag] = [(int(ev.step), float(ev.value)) for ev in ea.Scalars(tag)]
    return data


def read_run(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    scalars: dict[str, list[tuple[int, float]]] = {}
    for event_file in log_dir.rglob("events.out.tfevents.*"):
        try:
            data = read_event_file(event_file)
        except Exception:
            continue
        for tag, values in data.items():
            scalars.setdefault(tag, []).extend(values)
    for tag in list(scalars):
        scalars[tag] = sorted(set(scalars[tag]), key=lambda item: item[0])
    return scalars


def mean_agent_series(scalars: dict[str, list[tuple[int, float]]], metric: str):
    by_step: dict[int, list[float]] = {}
    for agent_id in range(4):
        tag = f"agent{agent_id}/{metric}"
        for step, value in scalars.get(tag, []):
            by_step.setdefault(step, []).append(value)
    return sorted((step, float(np.mean(vals))) for step, vals in by_step.items() if vals)


def named_series(scalars: dict[str, list[tuple[int, float]]], tag: str):
    return scalars.get(tag, [])


def write_curve_csv(path: Path, records: list[dict]):
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def append_records(records: list[dict], run: str, metric: str, series):
    for step, value in series:
        records.append({"run": run, "metric": metric, "step": step, "env_steps_m": step / 1e6, "value": value})


def plot_series(axis, run_data: dict[str, dict[str, list[tuple[int, float]]]], metric: str, title: str, ylabel: str,
                tag: str | None = None, agent_metric: bool = False, logy: bool = False):
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for idx, (run_name, scalars) in enumerate(sorted(run_data.items())):
        series = mean_agent_series(scalars, metric) if agent_metric else named_series(scalars, tag or metric)
        if not series:
            continue
        x = np.asarray([step for step, _ in series], dtype=np.float64) / 1e6
        y = np.asarray([value for _, value in series], dtype=np.float64)
        axis.plot(x, y, marker="o", markersize=2.4, color=colors[idx % len(colors)], label=run_name)
    axis.set_title(title, pad=2)
    axis.set_xlabel("Environment steps (million)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.32)
    if logy:
        axis.set_yscale("symlog", linthresh=1.0)
    if not logy:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.yaxis.set_major_formatter(formatter)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, loc="best")


def plot_grad_norms(axis, run_data: dict[str, dict[str, list[tuple[int, float]]]]):
    colors = {"actor_grad_norm": "#0072B2", "critic_grad_norm": "#D55E00"}
    linestyles = {"run1": "-", "run2": "--", "run3": ":", "run4": "-."}
    for run_name, scalars in sorted(run_data.items()):
        for metric, color in colors.items():
            series = mean_agent_series(scalars, metric)
            if not series:
                continue
            x = np.asarray([step for step, _ in series], dtype=np.float64) / 1e6
            y = np.asarray([value for _, value in series], dtype=np.float64)
            label = f"{run_name} {metric.replace('_grad_norm', '')}"
            axis.plot(x, y, marker="o", markersize=2.2, color=color,
                      linestyle=linestyles.get(run_name, "-"), label=label)
    axis.set_title("(d) Gradient norms", pad=2)
    axis.set_xlabel("Environment steps (million)")
    axis.set_ylabel("Norm")
    axis.grid(True, alpha=0.32)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, loc="best", ncol=2)


def main():
    args = parse_args()
    tb_root = Path(args.tb_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    run_data = {}
    for run_dir in sorted(tb_root.glob("run*")):
        log_dir = run_dir / "logs"
        if log_dir.exists():
            scalars = read_run(log_dir)
            if scalars:
                run_data[run_dir.name] = scalars

    records = []
    for run_name, scalars in run_data.items():
        for metric in ["average_step_rewards", "policy_loss", "value_loss", "dist_entropy", "actor_grad_norm", "critic_grad_norm", "ratio"]:
            append_records(records, run_name, metric, mean_agent_series(scalars, metric))
        for tag, metric in [
            ("train_episode_rewards/aver_rewards", "train_episode_rewards"),
            ("eval_average_episode_rewards/eval_average_episode_rewards", "eval_average_episode_rewards"),
            ("eval_max_episode_rewards/eval_max_episode_rewards", "eval_max_episode_rewards"),
            ("env/success", "env_success"),
            ("env/hit_count", "env_hit_count"),
        ]:
            append_records(records, run_name, metric, named_series(scalars, tag))
    write_curve_csv(out_dir / "training_curves.csv", records)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.15), constrained_layout=True)
    plot_series(axes[0, 0], run_data, "average_step_rewards", "(a) Dense training reward", "Step reward", agent_metric=True)
    plot_series(axes[0, 1], run_data, "policy_loss", "(b) MAPPO policy loss", "Policy loss", agent_metric=True)
    plot_series(axes[1, 0], run_data, "value_loss", "(c) MAPPO value loss", "Value loss", agent_metric=True)
    plot_grad_norms(axes[1, 1], run_data)
    fig.savefig(out_dir / "fig11_training_reward_loss_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig11_training_reward_loss_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), constrained_layout=True)
    plot_series(axes[0], run_data, "dist_entropy", "(a) Action entropy", "Entropy", agent_metric=True)
    plot_series(axes[1], run_data, "ratio", "(b) PPO probability ratio", "Ratio", agent_metric=True)
    fig.savefig(out_dir / "fig12_policy_entropy_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig12_policy_entropy_curves.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "tb_root": str(tb_root),
        "runs": sorted(run_data.keys()),
        "num_records": len(records),
        "figures": ["fig11_training_reward_loss_curves", "fig12_policy_entropy_curves"],
    }
    with (out_dir / "training_curves_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
