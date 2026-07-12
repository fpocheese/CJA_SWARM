#!/usr/bin/env python
"""Build comparison figures for the v69 adversary scenario suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.eval_v69_collect import (
    add_2d_arrows,
    annotate_time_markers,
    breach_time,
    closest_defender_rows,
    farr,
    load_rows,
    row_at_time,
    rows_in_window,
    save_fig,
    set_ieee_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_summary(eval_dir: Path) -> dict:
    with (eval_dir / "summary.json").open() as f:
        data = json.load(f)
    data["_eval_dir"] = str(eval_dir)
    return data


def choose_episode(summary: dict) -> dict:
    episodes = summary.get("episodes", [])
    for ep in episodes:
        if ep.get("success"):
            return ep
    return min(episodes, key=lambda ep: ep.get("best_min_dist_m", float("inf"))) if episodes else {}


def suite_rows(summaries: list[dict], labels: list[str]) -> list[dict]:
    rows = []
    for idx, summary in enumerate(summaries):
        label = labels[idx] if idx < len(labels) else summary.get("scenario_label", summary.get("scenario_case", f"case{idx+1}"))
        episodes = summary.get("episodes", [])
        hit_times = [float(ep.get("final_step", 0)) * 0.01 for ep in episodes if ep.get("success")]
        rows.append({
            "label": label,
            "scenario_case": summary.get("scenario_case", ""),
            "episodes": len(episodes),
            "success_episodes": int(summary.get("success_episodes", 0)),
            "success_rate": float(summary.get("hit_rate", 0.0)),
            "total_hits": int(summary.get("total_hits", 0)),
            "best_min_dist_m": float(summary.get("best_min_dist_m", 0.0)),
            "mean_hit_time_s": float(np.mean(hit_times)) if hit_times else float("nan"),
            "model_dir": summary.get("model_dir", ""),
            "eval_dir": summary.get("_eval_dir", ""),
        })
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(out_dir: Path, rows: list[dict]):
    set_ieee_style()
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    colors = ["#0072B2", "#D55E00", "#009E73"][:len(labels)]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.05), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(x, [row["success_rate"] for row in rows], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("(a) Episode-level success", pad=2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, [row["total_hits"] for row in rows], color=colors)
    ax.set_ylabel("Hits")
    ax.set_title("(b) Total HVT hits", pad=2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, [row["best_min_dist_m"] for row in rows], color=colors)
    ax.axhline(5.0, color="k", linestyle=":", linewidth=0.85, label="5 m hit radius")
    ax.set_ylabel("Best miss distance (m)")
    ax.set_title("(c) Best terminal miss distance", pad=2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend(frameon=False, loc="best", fontsize=6.2)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    hit_times = [row["mean_hit_time_s"] for row in rows]
    ax.bar(x, [0.0 if np.isnan(v) else v for v in hit_times], color=colors)
    ax.set_ylabel("Mean hit time (s)")
    ax.set_title("(d) Time to successful strike", pad=2)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "fig09_scenario_comparison")


def plot_breach_snapshots(out_dir: Path, summaries: list[dict], labels: list[str]):
    set_ieee_style()
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 2.35), constrained_layout=True, squeeze=False)
    for idx, summary in enumerate(summaries):
        ax = axes[0, idx]
        eval_dir = Path(summary["_eval_dir"])
        telemetry = load_rows(eval_dir / "telemetry.csv")
        ep = choose_episode(summary)
        if not ep:
            ax.set_axis_off()
            continue
        seed = int(ep["seed"])
        hit_agent = int(ep["hit_indices"][0]) if ep.get("hit_indices") else int(np.argmin(ep.get("min_dist_per_agent_m", [0])))
        tele_seed = [r for r in telemetry if int(r["seed"]) == seed]
        rows_hit = [r for r in tele_seed if r["team"] == "offense" and int(r["agent"]) == hit_agent]
        if not rows_hit:
            ax.set_axis_off()
            continue
        t_b = breach_time(rows_hit)
        def_id, rows_def = closest_defender_rows(tele_seed, rows_hit, t_b)
        hit_time = float(rows_hit[-1]["time_s"])
        hit_window = rows_in_window(rows_hit, t_b, 7.0, max(8.0, min(16.0, hit_time - t_b + 1.5)))
        def_window = rows_in_window(rows_def, t_b, 7.0, max(8.0, min(16.0, hit_time - t_b + 1.5))) if rows_def else []
        x_hit, y_hit = farr(hit_window, "x_m"), farr(hit_window, "y_m")
        ax.plot(x_hit, y_hit, color="#0072B2", lw=1.7, label=f"A{hit_agent}")
        add_2d_arrows(ax, x_hit, y_hit, "#0072B2", count=3)
        if def_window:
            x_def, y_def = farr(def_window, "x_m"), farr(def_window, "y_m")
            ax.plot(x_def, y_def, "--", color="#D55E00", lw=1.2, label=f"D{def_id}")
            add_2d_arrows(ax, x_def, y_def, "#D55E00", count=3)
        ax.scatter([1200], [0], marker="*", s=52, color="k", zorder=7)
        times = sorted(set(round(v, 1) for v in [max(float(hit_window[0]["time_s"]), t_b - 5), t_b, min(float(hit_window[-1]["time_s"]), t_b + 4), min(float(hit_window[-1]["time_s"]), hit_time)]))
        annotate_time_markers(ax, hit_window, times, "#0072B2", "o", "")
        if def_window:
            annotate_time_markers(ax, def_window, times, "#D55E00", "s", "")
        all_x = list(x_hit) + ([float(r["x_m"]) for r in def_window] if def_window else []) + [1200.0]
        all_y = list(y_hit) + ([float(r["y_m"]) for r in def_window] if def_window else []) + [0.0]
        pad = 46.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        label = labels[idx] if idx < len(labels) else summary.get("scenario_case", f"Case {idx+1}")
        status = "hit" if ep.get("success") else "best"
        ax.set_title(f"({chr(97+idx)}) {label}: {status}", pad=2)
        ax.set_xlabel("X (m)")
        if idx == 0:
            ax.set_ylabel("Y (m)")
        ax.legend(frameon=False, loc="best", fontsize=6.0)
    save_fig(fig, out_dir, "fig10_scenario_breach_snapshots")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dirs = [Path(item).resolve() for item in args.eval_dirs]
    summaries = [load_summary(item) for item in eval_dirs]
    rows = suite_rows(summaries, args.labels)
    write_csv(out_dir / "scenario_suite.csv", rows)
    with (out_dir / "scenario_suite_summary.json").open("w") as f:
        json.dump({"cases": rows}, f, indent=2, ensure_ascii=False)
    plot_comparison(out_dir, rows)
    plot_breach_snapshots(out_dir, summaries, args.labels)
    print(json.dumps({"out_dir": str(out_dir), "cases": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
