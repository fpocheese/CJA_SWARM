#!/usr/bin/env python3
"""Plot a compact HIL result figure from a recorded trajectory npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OFF_COL = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#ff9896",
]
DEF_COL = [
    "#17becf", "#ff7f0e", "#2a363b", "#e3b505",
    "#6a3d9a", "#b15928", "#1b9e77", "#7570b3",
    "#a6761d", "#66a61e",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="HIL 6v6 result")
    args = parser.parse_args()

    traj = np.load(args.trajectory, allow_pickle=True)
    with open(args.summary) as fh:
        summary_doc = json.load(fh)
    summary = summary_doc["summaries"][0] if "summaries" in summary_doc else summary_doc

    t = np.asarray(traj["time"], dtype=float)
    off_x = np.asarray(traj["off_x"], dtype=float)
    off_y = np.asarray(traj["off_y"], dtype=float)
    off_z = np.asarray(traj["off_z"], dtype=float)
    off_v = np.asarray(traj["off_v"], dtype=float)
    off_d = np.asarray(traj["off_d_hvt"], dtype=float)
    def_x = np.asarray(traj["def_x"], dtype=float)
    def_y = np.asarray(traj["def_y"], dtype=float)
    def_z = np.asarray(traj["def_z"], dtype=float)
    actions = np.asarray(traj["actor_actions"], dtype=float)
    hvt = (float(traj["hvt_x"]), float(traj["hvt_y"]), float(traj["hvt_z"]))
    n_off = off_x.shape[0]
    n_def = def_x.shape[0]
    hitter = int(summary.get("hit_indices", [summary.get("best_agent", 0)])[0]) if summary.get("hit_indices") else int(summary.get("best_agent", 0))
    hit_time = float(summary.get("final_time_s", t[-1])) if summary.get("success") else None

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(12.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])

    for i in range(n_off):
        color = OFF_COL[i % len(OFF_COL)]
        lw = 2.2 if i == hitter else 1.1
        ax3d.plot(off_x[i], off_y[i], off_z[i], color=color, lw=lw,
                  label=f"A{i}" + (" hit" if i == hitter and summary.get("success") else ""))
        ax3d.scatter(off_x[i, 0], off_y[i, 0], off_z[i, 0], marker="o", s=24,
                     facecolor="white", edgecolor=color)
        ax3d.scatter(off_x[i, -1], off_y[i, -1], off_z[i, -1], marker="D", s=30,
                     color=color)
        ax_d.plot(t, off_d[i], color=color, lw=lw, label=f"A{i}")
        ax_v.plot(t, off_v[i], color=color, lw=lw, label=f"A{i}")

    for j in range(n_def):
        color = DEF_COL[j % len(DEF_COL)]
        ax3d.plot(def_x[j], def_y[j], def_z[j], color=color, lw=0.9,
                  ls="--", alpha=0.7, label=f"D{j}")

    ax3d.scatter([hvt[0]], [hvt[1]], [hvt[2]], marker="*", s=180,
                 color="#ffbf00", edgecolor="k", label="HVT")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D trajectories")
    ax3d.view_init(elev=22, azim=-52)
    ax3d.legend(loc="upper left", fontsize=7, ncol=2)

    if hit_time is not None:
        for ax in (ax_d, ax_v):
            ax.axvline(hit_time, color="k", lw=1.1, ls=":", label="hit")
    ax_d.set_title("Distance to HVT")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_ylabel("Distance (m)")
    ax_d.legend(fontsize=7, ncol=3)

    ax_v.set_title("Offensive speed")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Speed (m/s)")
    ax_v.legend(fontsize=7, ncol=3)

    status = "success" if summary.get("success") else summary.get("done_reason", "unknown")
    fig.suptitle(
        f"{args.title}: seed={summary.get('seed')} status={status}, "
        f"hit={summary.get('hit_indices', [])}, best={summary.get('best_min_dist_m', summary.get('best_hvt_distance_m', float('nan'))):.2f} m",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
