#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_cases(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def unique_cases(rows):
    cases = {}
    for row in rows:
        cases[row["case_id"]] = {
            "case_id": row["case_id"],
            "group": row["group"],
            "param": row["param"],
            "value": row["value"],
        }
    return list(cases.values())


def summarize_case(input_dir, case):
    case_dir = input_dir / case["case_id"]
    curves = sorted(case_dir.glob("Advanced-MAPPO_seed*_rewards.npy"))
    if not curves:
        curves = sorted(case_dir.glob("*_seed*_rewards.npy"))
    finals = []
    aucs = []
    lengths = []
    for curve_path in curves:
        curve = np.asarray(np.load(curve_path), dtype=np.float64).reshape(-1)
        if curve.size == 0:
            continue
        window = max(5, int(0.2 * curve.size))
        finals.append(float(np.mean(curve[-window:])))
        aucs.append(float(np.mean(curve)))
        lengths.append(int(curve.size))
    if not finals:
        return None
    return {
        **case,
        "num_seeds": len(finals),
        "episodes_min": min(lengths),
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def write_summary(path, rows):
    fieldnames = [
        "case_id", "group", "param", "value", "num_seeds", "episodes_min",
        "final_mean", "final_std", "auc_mean", "auc_std", "relative_final",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_for(row):
    if row["group"] == "base":
        return "base"
    name = row["param"]
    name = name.replace("reward_w_", "w_")
    name = name.replace("clip_param", "clip")
    name = name.replace("entropy_coef", "entropy")
    name = name.replace("gae_lambda", "GAE")
    name = name.replace("target_kl", "KL")
    return "%s=%s" % (name, row["value"])


def plot_group(ax, rows, title):
    if not rows:
        ax.axis("off")
        return
    x = np.arange(len(rows))
    y = np.array([r["relative_final"] for r in rows], dtype=np.float64)
    yerr = np.array([r["final_std"] / max(abs(r["base_final"]), 1e-9) for r in rows], dtype=np.float64)
    colors = ["#4C78A8" if r["group"] == "reward" else "#F58518" if r["group"] == "hyper" else "#54A24B" for r in rows]
    ax.bar(x, y, yerr=yerr, color=colors, alpha=0.86, edgecolor="black", linewidth=0.45, capsize=3)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(r) for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative final reward")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot reward/hyperparameter sensitivity summary.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="reward_sensitivity")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    rows = load_cases(input_dir / "cases.csv")
    summaries = []
    for case in unique_cases(rows):
        summary = summarize_case(input_dir, case)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No reward curves found under %s" % input_dir)

    base_rows = [r for r in summaries if r["case_id"] == "base"]
    if not base_rows:
        raise SystemExit("Baseline case is missing; cannot normalize sensitivity.")
    base_final = base_rows[0]["final_mean"]
    for row in summaries:
        row["base_final"] = base_final
        row["relative_final"] = row["final_mean"] / max(abs(base_final), 1e-9)

    summaries = sorted(
        summaries,
        key=lambda r: ({"base": 0, "reward": 1, "hyper": 2}.get(r["group"], 9), r["param"], str(r["value"])),
    )
    write_summary(input_dir / "sensitivity_summary.csv", summaries)

    reward_rows = [r for r in summaries if r["group"] == "reward"]
    hyper_rows = [r for r in summaries if r["group"] == "hyper"]

    if reward_rows and hyper_rows:
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), constrained_layout=True)
        plot_group(axes[0], reward_rows, "Reward-weight sensitivity")
        plot_group(axes[1], hyper_rows, "Training-hyperparameter sensitivity")
    elif reward_rows:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2), constrained_layout=True)
        plot_group(ax, reward_rows, "Reward-weight sensitivity")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2), constrained_layout=True)
        plot_group(ax, hyper_rows, "Training-hyperparameter sensitivity")

    png_path = input_dir / ("%s.png" % args.output_prefix)
    pdf_path = input_dir / ("%s.pdf" % args.output_prefix)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print("Summary saved to: %s" % (input_dir / "sensitivity_summary.csv"))
    print("Figure saved to: %s and %s" % (png_path, pdf_path))


if __name__ == "__main__":
    main()
