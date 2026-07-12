#!/usr/bin/env python3
"""Standalone 2-D reproductions of the two source-paper examples.

The generated figures are sanity checks for the source algorithms themselves,
separate from the V71 3-D FOV-penetration comparison cases.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("paper_reproductions")
OUT.mkdir(exist_ok=True)


def wrap_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def garcia_yij(e, p, v_e, v_p):
    alpha = v_e / v_p
    d = np.linalg.norm(e - p)
    return (e[1] - alpha**2 * p[1] - alpha * d) / max(1.0 - alpha**2, 1e-9)


def garcia_aimpoint(e, p, v_e, v_p):
    alpha = v_e / v_p
    d = np.linalg.norm(e - p)
    den = max(1.0 - alpha**2, 1e-9)
    return np.array([
        (e[0] - alpha**2 * p[0]) / den,
        (e[1] - alpha**2 * p[1] - alpha * d) / den,
    ])


def reproduce_garcia_bddg():
    # Representative 3 pursuer / 3 evader BDDG case matching the paper's
    # closed-form assignment and Apollonius guidance mechanism.
    evaders = np.array([[2.0, 8.5], [9.0, 7.2], [16.0, 9.0]], dtype=float)
    pursuers = np.array([[1.0, 1.0], [10.0, 2.0], [18.0, 1.0]], dtype=float)
    v_e = np.array([0.65, 0.65, 0.65])
    v_p = np.array([1.0, 1.0, 1.0])
    n = len(evaders)

    rows = []
    best = None
    for perm in itertools.permutations(range(n)):
        payoff = 0.0
        yvals = []
        for j, i in enumerate(perm):
            y = garcia_yij(evaders[j], pursuers[i], v_e[j], v_p[i])
            yvals.append(y)
            payoff += y
        rows.append({"assignment": list(perm), "ys": payoff, "yij": yvals})
        if best is None or payoff > best["ys"]:
            best = rows[-1]

    e = evaders.copy()
    p = pursuers.copy()
    dt = 0.02
    traj_e = [e.copy()]
    traj_p = [p.copy()]
    capture_times = {}
    for k in range(1200):
        for j, i in enumerate(best["assignment"]):
            if j in capture_times:
                continue
            aim = garcia_aimpoint(e[j], p[i], v_e[j], v_p[i])
            de = aim - e[j]
            dp = aim - p[i]
            if np.linalg.norm(de) > 1e-8:
                e[j] += v_e[j] * dt * de / np.linalg.norm(de)
            if np.linalg.norm(dp) > 1e-8:
                p[i] += v_p[i] * dt * dp / np.linalg.norm(dp)
            if np.linalg.norm(e[j] - p[i]) < 0.02:
                capture_times[j] = k * dt
        traj_e.append(e.copy())
        traj_p.append(p.copy())
        if len(capture_times) == n:
            break

    traj_e = np.asarray(traj_e)
    traj_p = np.asarray(traj_p)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.axhline(0, color="k", lw=1.1, label="border")
    for j in range(n):
        ax.plot(traj_e[:, j, 0], traj_e[:, j, 1], lw=1.6, label=fr"$E_{j+1}$")
        ax.scatter(traj_e[0, j, 0], traj_e[0, j, 1], marker="o", facecolor="white", edgecolor="C0")
    for i in range(n):
        ax.plot(traj_p[:, i, 0], traj_p[:, i, 1], "--", lw=1.3, label=fr"$P_{i+1}$")
        ax.scatter(traj_p[0, i, 0], traj_p[0, i, 1], marker="s", facecolor="white", edgecolor="C1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Garcia et al. BDDG closed-form assignment reproduction")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "garcia_bddg_reproduction.pdf")
    plt.close(fig)

    with open(OUT / "garcia_bddg_summary.json", "w") as f:
        json.dump({"assignments": rows, "best": best, "capture_times": capture_times}, f, indent=2)


def wei_yang_initial_positions():
    # Example 1 table values from Wei & Yang 2018.
    lamb = np.array([1.0000, 2.0001, 3.0000, 4.0000])
    R = np.array([16.1200, 16.5853, 15.8100, 15.5000])
    target = np.array([-8.9561, 5.9793])
    attackers = target + np.column_stack([R * np.cos(lamb), R * np.sin(lamb)])
    return attackers, target


def reproduce_wei_yang():
    # Deterministic feedback approximation of the paper's Example 1 geometry:
    # attackers that can observe target form an encirclement; target reacts
    # only inside rc, reproducing rr > rc late-awareness capture behavior.
    attackers, target = wei_yang_initial_positions()
    VA = 1.406
    VT = 0.7
    Rc = 1.1
    rc = 3.05
    rr = 5.061
    dt = 0.002
    tf = 18.0
    a = attackers.copy()
    t = target.copy()
    tr_a = [a.copy()]
    tr_t = [t.copy()]
    min_r = []
    hit_time = None
    for k in range(int(tf / dt)):
        rel = a - t
        R = np.linalg.norm(rel, axis=1)
        min_r.append(float(R.min()))
        if R.min() <= Rc:
            hit_time = k * dt
            break
        centroid = a.mean(axis=0)
        detected = R < rc
        if detected.any():
            flee = t - a[detected].mean(axis=0)
        else:
            flee = t - centroid
        if np.linalg.norm(flee) < 1e-9:
            flee = np.array([1.0, 0.0])
        t += VT * dt * flee / np.linalg.norm(flee)

        center_dir = t - a
        for i in range(4):
            # attackers outside rr keep moving to reduce range; inside rr they
            # add ring-closing terms so the target is driven toward the group.
            toward = center_dir[i] / max(np.linalg.norm(center_dir[i]), 1e-9)
            if R[i] < rr:
                angle = math.atan2(toward[1], toward[0]) + 0.45 * math.sin(i * math.pi / 2)
                ring = np.array([math.cos(angle), math.sin(angle)])
                direction = 0.86 * toward + 0.14 * ring
            else:
                direction = toward
            a[i] += VA * dt * direction / max(np.linalg.norm(direction), 1e-9)
        tr_a.append(a.copy())
        tr_t.append(t.copy())

    tr_a = np.asarray(tr_a)
    tr_t = np.asarray(tr_t)
    time = np.arange(len(min_r)) * dt
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    for i in range(4):
        ax.plot(tr_a[:, i, 0], tr_a[:, i, 1], lw=1.2, label=fr"$A_{i+1}$")
        ax.scatter(tr_a[0, i, 0], tr_a[0, i, 1], facecolor="white", edgecolor=f"C{i}", s=24)
    ax.plot(tr_t[:, 0], tr_t[:, 1], "k", lw=1.8, label="target")
    ax.scatter(tr_t[0, 0], tr_t[0, 1], marker="*", color="k", s=60)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    axes[1].plot(time, min_r, color="#b22222")
    axes[1].axhline(Rc, color="k", ls="--", lw=0.9, label=r"$R_c$")
    axes[1].axhline(rr, color="0.5", ls=":", lw=0.9, label=r"$r_r$")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("min range (km)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.suptitle("Wei-Yang 2018 Example-1 geometry reproduction")
    fig.tight_layout()
    fig.savefig(OUT / "wei_yang_example1_reproduction.pdf")
    plt.close(fig)
    with open(OUT / "wei_yang_example1_summary.json", "w") as f:
        json.dump({"hit_time_s": hit_time, "paper_tf_s": 15.3537, "min_range_km": min_r[-1]}, f, indent=2)


if __name__ == "__main__":
    reproduce_garcia_bddg()
    reproduce_wei_yang()
    print(f"wrote {OUT}")
