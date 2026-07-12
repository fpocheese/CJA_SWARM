#!/usr/bin/env python3
"""Plot the short RL convergence preview from train_single_algo.py."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="onpolicy/scripts/results/rl_convergence_preview")
    parser.add_argument("--algo", type=str, default="Advanced-MAPPO")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reward_file = results_dir / f"{args.algo}_seed{args.seed}_rewards.npy"
    rewards = np.load(reward_file)
    episodes = np.arange(1, len(rewards) + 1)
    smooth = moving_average(rewards, max(1, args.window))
    smooth_ep = np.arange(max(1, args.window), len(rewards) + 1) if len(rewards) >= args.window else episodes

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(episodes, rewards, color="#868e96", linewidth=1.0, alpha=0.65, label="Episode reward")
    ax.plot(smooth_ep, smooth, color="#1864ab", linewidth=1.8, label=f"Moving average (w={args.window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average episode reward")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(results_dir / f"rl_convergence_preview.{ext}", bbox_inches="tight")
    plt.close(fig)

    with open(results_dir / "rl_convergence_preview_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["episodes", len(rewards)])
        writer.writerow(["initial_reward", float(rewards[0])])
        writer.writerow(["final_reward", float(rewards[-1])])
        writer.writerow(["best_reward", float(np.max(rewards))])
        writer.writerow(["last_window_mean", float(np.mean(rewards[-min(len(rewards), args.window):]))])

    print(f"saved: {results_dir / 'rl_convergence_preview.pdf'}")


if __name__ == "__main__":
    main()
