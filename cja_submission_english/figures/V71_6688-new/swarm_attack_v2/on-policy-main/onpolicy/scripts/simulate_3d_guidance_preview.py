#!/usr/bin/env python3
"""Preview 3D cooperative interception scenarios.

This script is intentionally independent from the 2D ART-MAPPO training loop.
The existing trained policies and MPE environment are 2D, so this preview first
validates the 3D engagement geometry, target maneuvers, and paper-style figures.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


G = 9.81
N_DEF = 20
N_ATT = 8
DT = 0.05
HORIZON = 75.0
HIT_RADIUS = 20.0


ASSIGNMENT = np.array([
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3,
], dtype=int)


COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        if fallback is None:
            return np.zeros_like(v)
        return fallback.copy()
    return v / n


def clip_norm(v, max_norm):
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-9:
        return v
    return v * (max_norm / n)


def init_scene(seed):
    rng = np.random.default_rng(seed)
    protected = np.array([0.0, 0.0, 0.0])

    att_angles = np.linspace(0.0, 2 * np.pi, N_ATT, endpoint=False) + rng.normal(0.0, 0.08, N_ATT)
    att_r = rng.uniform(1200.0, 1500.0, N_ATT)
    att_z = rng.uniform(650.0, 850.0, N_ATT)
    att_pos = np.column_stack((att_r * np.cos(att_angles), att_r * np.sin(att_angles), att_z))
    att_speed = rng.uniform(24.0, 30.0, N_ATT)
    att_vel = np.zeros_like(att_pos)
    for j in range(N_ATT):
        direction = unit(protected - att_pos[j])
        att_vel[j] = att_speed[j] * direction

    def_angles = np.linspace(0.0, 2 * np.pi, N_DEF, endpoint=False) + rng.normal(0.0, 0.12, N_DEF)
    def_r = rng.uniform(20.0, 95.0, N_DEF)
    def_pos = np.column_stack((def_r * np.cos(def_angles), def_r * np.sin(def_angles), np.zeros(N_DEF)))
    def_speed = rng.uniform(16.0, 22.0, N_DEF)
    def_vel = np.zeros_like(def_pos)
    for i in range(N_DEF):
        j = ASSIGNMENT[i]
        climb_bias = np.array([0.0, 0.0, 0.22])
        direction = unit(att_pos[j] - def_pos[i] + 150.0 * climb_bias)
        def_vel[i] = def_speed[i] * direction

    return protected, att_pos, att_vel, def_pos, def_vel


def update_attackers(case_name, t, protected, att_pos, att_vel, phases):
    new_pos = att_pos.copy()
    new_vel = att_vel.copy()
    for j in range(N_ATT):
        speed = np.linalg.norm(new_vel[j])
        to_goal = unit(protected - new_pos[j], fallback=np.array([0.0, 0.0, -1.0]))
        horizontal_perp = unit(np.cross(np.array([0.0, 0.0, 1.0]), to_goal),
                               fallback=np.array([1.0, 0.0, 0.0]))

        homing = 0.55 * G * (to_goal - unit(new_vel[j], fallback=to_goal))
        maneuver = np.zeros(3)
        if case_name == "case1":
            if t < 15.0:
                sign = 1.0 if (j % 2 == 0) else -1.0
                maneuver = sign * 0.38 * G * horizontal_perp
                maneuver += 0.10 * G * math.sin(0.65 * t + phases[j]) * np.array([0.0, 0.0, 1.0])
            else:
                maneuver = 0.10 * G * math.sin(0.25 * t + phases[j]) * horizontal_perp
        else:
            maneuver = 0.45 * G * math.sin(0.85 * t + phases[j]) * horizontal_perp
            maneuver += 0.16 * G * math.sin(1.25 * t + 0.7 * phases[j]) * np.array([0.0, 0.0, 1.0])

        acc = clip_norm(homing + maneuver, 0.65 * G)
        vel = new_vel[j] + acc * DT
        vel = unit(vel, fallback=to_goal) * np.clip(np.linalg.norm(vel), 20.0, 32.0)
        new_vel[j] = vel
        new_pos[j] = new_pos[j] + vel * DT
        if new_pos[j, 2] < 0.0:
            new_pos[j, 2] = 0.0
            new_vel[j, 2] = min(0.0, new_vel[j, 2])
    return new_pos, new_vel


def update_defenders(t, att_pos, att_vel, def_pos, def_vel, done, target_done_times):
    new_pos = def_pos.copy()
    new_vel = def_vel.copy()
    controls = np.zeros((N_DEF, 3))  # nx, ny, nz in g
    tgo = np.full(N_DEF, np.nan)

    group_tgo = {}
    for target in range(N_ATT):
        members = np.where((ASSIGNMENT == target) & (~done))[0]
        vals = []
        for i in members:
            rel = att_pos[target] - def_pos[i]
            dist = np.linalg.norm(rel)
            speed = max(np.linalg.norm(def_vel[i]), 1.0)
            vals.append(dist / speed)
        group_tgo[target] = float(np.mean(vals)) if vals else 0.0

    for i in range(N_DEF):
        if done[i]:
            controls[i] = 0.0
            continue

        j = ASSIGNMENT[i]
        rel = att_pos[j] - new_pos[i]
        dist = np.linalg.norm(rel)
        speed = max(np.linalg.norm(new_vel[i]), 1.0)
        tgo[i] = dist / speed

        lead_time = np.clip(dist / max(speed + np.linalg.norm(att_vel[j]), 1.0), 0.0, 4.5)
        aim_point = att_pos[j] + 0.85 * lead_time * att_vel[j]
        los = unit(aim_point - new_pos[i], fallback=unit(rel))

        mean_tgo = max(group_tgo[j], 1.0)
        speed_cmd = 45.0 + 1.8 * (tgo[i] - mean_tgo)
        if dist < 350.0:
            speed_cmd += 4.0
        speed_cmd = float(np.clip(speed_cmd, 22.0, 62.0))

        desired_vel = speed_cmd * los
        acc_cmd = (desired_vel - new_vel[i]) / 1.4

        vhat = unit(new_vel[i], fallback=los)
        axial = float(np.dot(acc_cmd, vhat))
        normal = acc_cmd - axial * vhat
        axial = float(np.clip(axial, -0.10 * G, 1.0 * G))
        normal = clip_norm(normal, 1.0 * G)
        acc = axial * vhat + normal

        # Split normal command into horizontal lateral and vertical parts for paper-style plots.
        horizontal_vel = np.array([new_vel[i, 0], new_vel[i, 1], 0.0])
        hhat = unit(horizontal_vel, fallback=np.array([1.0, 0.0, 0.0]))
        lhat = np.array([-hhat[1], hhat[0], 0.0])
        controls[i, 0] = axial / G
        controls[i, 1] = np.dot(acc, lhat) / G
        controls[i, 2] = acc[2] / G

        new_vel[i] = new_vel[i] + acc * DT
        spd = np.linalg.norm(new_vel[i])
        if spd > 65.0:
            new_vel[i] *= 65.0 / spd
        if spd < 12.0:
            new_vel[i] = unit(new_vel[i], fallback=los) * 12.0
        new_pos[i] = new_pos[i] + new_vel[i] * DT
        if new_pos[i, 2] < 0.0:
            new_pos[i, 2] = 0.0
            new_vel[i, 2] = max(0.0, new_vel[i, 2])

        true_dist = np.linalg.norm(att_pos[j] - new_pos[i])
        if true_dist <= HIT_RADIUS:
            done[i] = True
            target_done_times[i] = t
            new_vel[i] *= 0.15
    return new_pos, new_vel, controls, tgo, done, target_done_times


def simulate_case(case_name, seed):
    protected, att_pos, att_vel, def_pos, def_vel = init_scene(seed)
    rng = np.random.default_rng(seed + 123)
    phases = rng.uniform(0.0, 2 * np.pi, N_ATT)
    n_steps = int(HORIZON / DT) + 1
    times = np.linspace(0.0, HORIZON, n_steps)

    att_hist = np.zeros((n_steps, N_ATT, 3))
    def_hist = np.zeros((n_steps, N_DEF, 3))
    speed_hist = np.zeros((n_steps, N_DEF))
    ctrl_hist = np.zeros((n_steps, N_DEF, 3))
    tgo_hist = np.full((n_steps, N_DEF), np.nan)
    done = np.zeros(N_DEF, dtype=bool)
    hit_times = np.full(N_DEF, np.nan)

    for k, t in enumerate(times):
        att_hist[k] = att_pos
        def_hist[k] = def_pos
        speed_hist[k] = np.linalg.norm(def_vel, axis=1)

        dist_now = np.linalg.norm(att_pos[ASSIGNMENT] - def_pos, axis=1)
        tgo_hist[k] = np.where(done, np.nan, dist_now / np.maximum(speed_hist[k], 1.0))
        if k == n_steps - 1 or np.all(done):
            break

        att_pos, att_vel = update_attackers(case_name, t, protected, att_pos, att_vel, phases)
        def_pos, def_vel, controls, tgo, done, hit_times = update_defenders(
            t, att_pos, att_vel, def_pos, def_vel, done, hit_times)
        ctrl_hist[k + 1] = controls

    used = k + 1
    return {
        "case": case_name,
        "times": times[:used],
        "att": att_hist[:used],
        "def": def_hist[:used],
        "speed": speed_hist[:used],
        "controls": ctrl_hist[:used],
        "tgo": tgo_hist[:used],
        "hit_times": hit_times,
        "success": done.copy(),
    }


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def savefig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(result, outdir):
    case = result["case"]
    att = result["att"]
    deff = result["def"]

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    for j in range(N_ATT):
        ax.plot(att[:, j, 0], att[:, j, 1], att[:, j, 2],
                color="crimson", linestyle="--", linewidth=1.3, alpha=0.9)
        ax.scatter(att[0, j, 0], att[0, j, 1], att[0, j, 2],
                   marker="x", color="crimson", s=20)
        ax.text(att[0, j, 0], att[0, j, 1], att[0, j, 2] + 30, f"A{j+1}", color="crimson", fontsize=7)

    for i in range(N_DEF):
        c = COLORS[ASSIGNMENT[i]]
        ax.plot(deff[:, i, 0], deff[:, i, 1], deff[:, i, 2], color=c, linewidth=0.9, alpha=0.8)
        ax.scatter(deff[0, i, 0], deff[0, i, 1], deff[0, i, 2], marker="^", color=c, s=10)

    ax.scatter([0], [0], [0], marker="*", color="black", s=70, label="Protected asset")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Altitude z (m)")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1, 1, 0.45))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    savefig(fig, outdir, f"{case}_3d_trajectory")


def plot_kinematics(result, outdir):
    case = result["case"]
    t = result["times"]
    ctrl = result["controls"]
    speed = result["speed"]

    labels = [
        ("$n_y$ lateral (g)", ctrl[:, :, 1], (-1.1, 1.1)),
        ("$n_z$ vertical (g)", ctrl[:, :, 2], (-1.1, 1.1)),
        ("Velocity $V_D$ (m/s)", speed, (0, 70)),
        ("$n_x$ axial (g)", ctrl[:, :, 0], (-0.2, 1.1)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.7), sharex=True)
    axes = axes.ravel()
    for ax, (ylabel, data, ylim) in zip(axes, labels):
        for i in range(N_DEF):
            ax.plot(t, data[:, i], color=COLORS[ASSIGNMENT[i]], linewidth=0.65, alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].set_xlabel("Time (s)")
    axes[3].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, outdir, f"{case}_3d_kinematics")


def plot_tgo(result, outdir):
    case = result["case"]
    t = result["times"]
    tgo = result["tgo"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i in range(N_DEF):
        ax.plot(t, tgo[:, i], color=COLORS[ASSIGNMENT[i]], linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$t_{go}$ (s)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, outdir, f"{case}_3d_tgo")


def plot_time_sync(result, outdir):
    case = result["case"]
    hit = result["hit_times"]
    deltas = []
    labels = []
    for j in range(N_ATT):
        members = np.where(ASSIGNMENT == j)[0]
        times = hit[members]
        if np.all(np.isfinite(times)):
            deltas.append(float(np.max(times) - np.min(times)))
        else:
            deltas.append(np.nan)
        labels.append(f"A{j+1}")

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(N_ATT)
    vals = np.nan_to_num(deltas, nan=HORIZON)
    colors = [COLORS[j] for j in range(N_ATT)]
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, alpha=0.82)
    for idx, val in enumerate(deltas):
        txt = "fail" if not np.isfinite(val) else f"{val:.2f}"
        ax.text(idx, vals[idx] + 0.08, txt, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Terminal $\Delta t$ (s)")
    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, outdir, f"{case}_3d_time_sync")
    return deltas


def write_summary(results, sync_rows, outdir):
    path = outdir / "summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "success_count", "mean_hit_time", "max_delta_t", "mean_min_miss"])
        for result, deltas in zip(results, sync_rows):
            hit = result["hit_times"]
            success = result["success"]
            miss_history = np.linalg.norm(result["att"][:, ASSIGNMENT, :] - result["def"], axis=2)
            min_miss = np.min(miss_history, axis=0)
            finite_deltas = [d for d in deltas if np.isfinite(d)]
            writer.writerow([
                result["case"],
                int(np.sum(success)),
                float(np.nanmean(hit)) if np.any(np.isfinite(hit)) else np.nan,
                float(np.max(finite_deltas)) if finite_deltas else np.nan,
                float(np.mean(min_miss)),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="onpolicy/scripts/results/3d_guidance_preview")
    parser.add_argument("--seed", type=int, default=2606)
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = [
        simulate_case("case1", args.seed),
        simulate_case("case2", args.seed + 9),
    ]
    sync_rows = []
    for result in results:
        plot_trajectory(result, outdir)
        plot_kinematics(result, outdir)
        plot_tgo(result, outdir)
        sync_rows.append(plot_time_sync(result, outdir))

    write_summary(results, sync_rows, outdir)
    print(f"Saved 3D preview figures to: {outdir}")
    for result, deltas in zip(results, sync_rows):
        finite = [d for d in deltas if np.isfinite(d)]
        print(
            f"{result['case']}: success={int(np.sum(result['success']))}/{N_DEF}, "
            f"mean_hit={np.nanmean(result['hit_times']):.2f}s, "
            f"max_delta_t={np.max(finite) if finite else np.nan:.2f}s")


if __name__ == "__main__":
    main()
