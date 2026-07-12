#!/usr/bin/env python
"""Regenerate fig1 (8-panel) with HEAVILY smoothed np/ny and fig_def (defender only)."""
import numpy as np, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

OFF_COL = ["#e74c3c","#3498db","#2ecc71","#9b59b6"]
DEF_COL = ["#1abc9c","#e67e22","#34495e","#d35400"]
G = 9.80665
W = 101  # 1.01s smoothing window

def smooth(arr, w=W, o=3):
    a = np.asarray(arr, dtype=np.float64)
    if len(a) < w: return a
    ww = min(w, len(a) - (len(a) % 2 == 0))
    if ww < o+2: return a
    return savgol_filter(a, ww, o)

def process_case(base, case_name):
    d = np.load(f"{base}/trajectory_data.npz", allow_pickle=True)
    with open(f"{base}/summary.json") as f: sm = json.load(f)
    t = d["time"]; hitter = sm["hitter"]; n_off = 4; n_def = 4

    # =============== FIG1 : 8-panel kinematics (smoothed np/ny, no raw) ===============
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(8, 1, hspace=0.45)

    # row1: speed
    ax = fig.add_subplot(gs[0])
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        ax.plot(t[mask], np.array(d["off_v"][i][mask]), color=OFF_COL[i],
                lw=1.6 if i==hitter else 1.1, label=f"A{i}")
    ax.set_ylabel("$V$ (m/s)", fontsize=9)
    ax.axhline(40, color="gray", ls="--", lw=0.7); ax.axhline(50, color="gray", ls="--", lw=0.7)
    ax.set_ylim(35, 55); ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.set_title(f"{case_name} — Offensive Kinematics (Smoothed Overload)", fontsize=10)
    ax.tick_params(labelbottom=False)

    # row2: heading rate
    ax = fig.add_subplot(gs[1])
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        h = np.array(d["off_heading"][i][mask])
        if len(h) > 2:
            dh = np.gradient(np.unwrap(h), 0.01)
            ax.plot(t[mask], np.degrees(dh), color=OFF_COL[i], lw=1.3 if i==hitter else 0.9)
    ax.set_ylabel(r"$\dot{\psi}$ (°/s)", fontsize=9)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.tick_params(labelbottom=False)

    # row3: np SMOOTHED ONLY
    ax = fig.add_subplot(gs[2])
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        raw = np.array(d["off_an_pitch"][i][mask]) / G
        sm = smooth(raw)
        ax.plot(t[mask], sm, color=OFF_COL[i], lw=2.0 if i==hitter else 1.3)
    ax.set_ylabel("$n_p$ (g)", fontsize=9)
    ax.axhline(1, color="gray", ls="--", lw=0.7)
    ax.axhline(3, color="red", ls="--", lw=0.7, alpha=0.5); ax.axhline(-3, color="red", ls="--", lw=0.7, alpha=0.5)
    ax.tick_params(labelbottom=False)

    # row4: ny SMOOTHED ONLY
    ax = fig.add_subplot(gs[3])
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        raw = np.array(d["off_an_yaw"][i][mask]) / G
        sm = smooth(raw)
        ax.plot(t[mask], sm, color=OFF_COL[i], lw=2.0 if i==hitter else 1.3)
    ax.set_ylabel("$n_y$ (g)", fontsize=9)
    ax.axhline(3, color="red", ls="--", lw=0.7, alpha=0.5); ax.axhline(-3, color="red", ls="--", lw=0.7, alpha=0.5)
    ax.tick_params(labelbottom=False)

    # row5: hitter distance
    ax = fig.add_subplot(gs[4])
    dh = np.array(d["off_d_hvt"][hitter])
    ax.semilogy(t[:len(dh)], dh, color=OFF_COL[hitter], lw=1.8, label=f"$d_{{A{hitter},H}}$")
    dd = []
    for s in range(len(dh)):
        md = 9999
        for j in range(n_def):
            if s < len(d["def_x"][j]):
                md = min(md, np.sqrt((d["def_x"][j][s]-d["off_x"][hitter][s])**2+
                                     (d["def_y"][j][s]-d["off_y"][hitter][s])**2+
                                     (d["def_z"][j][s]-d["off_z"][hitter][s])**2))
        dd.append(md)
    ax.semilogy(t[:len(dh)], dd, color="#e67e22", lw=1.3, ls="--", label=f"$d_{{A{hitter},D^*}}$")
    ax.axhline(500, color="gray", ls=":", lw=1.0); ax.axhline(5, color="red", ls=":", lw=1.0)
    ax.set_ylabel("Distance (m)", fontsize=9); ax.legend(loc="upper right", fontsize=7)
    if sm.get("hit_time_s"):
        ax.axvspan(sm["hit_time_s"]-5, sm["hit_time_s"], alpha=0.15, color="green")
    ax.tick_params(labelbottom=False)

    # row6: defender |n| SMOOTHED
    ax = fig.add_subplot(gs[5])
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; an = smooth(np.array(d["def_an"][j][alive]))
        ax.plot(td, an, color=DEF_COL[j], lw=1.3, label=f"D{j}")
    ax.axhline(5, color="red", ls="--", lw=0.7, alpha=0.6)
    ax.set_ylabel("$|n_d|$ (g)", fontsize=9); ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.tick_params(labelbottom=False)

    # row7: defender lock mode
    ax = fig.add_subplot(gs[6])
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; modes = np.array(d["def_lmode"][j][alive])
        ax.plot(td, modes + j*0.08, color=DEF_COL[j], lw=1.5, label=f"D{j}")
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(["INIT","FOV_TRK","LOCKED","MISSED","ABANDON"], fontsize=7)
    ax.set_ylabel("State", fontsize=9); ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.tick_params(labelbottom=False)

    ax.set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{base}/fig1_kinematics_all8.pdf", bbox_inches="tight", dpi=100)
    fig.savefig(f"{base}/fig1_kinematics_all8.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  [fig1] {base}/fig1_kinematics_all8.pdf")

    # =============== FIG_DEF : defender overload + state machine (2-panel) ===============
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_n, ax_m = axes
    fig.suptitle(f"{case_name} — Defender Overload & State (S-G win={W*0.01:.2f}s)", fontsize=11)

    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]
        an_raw = np.array(d["def_an"][j][alive])
        an_sm = smooth(an_raw)
        ax_n.plot(td, an_sm, color=DEF_COL[j], lw=1.5, label=f"D{j}")
        modes = np.array(d["def_lmode"][j][alive])
        ax_m.plot(td, modes + j*0.08, color=DEF_COL[j], lw=1.5, label=f"D{j}")

    ax_n.axhline(5, color="red", ls="--", lw=0.8, alpha=0.6, label="5g limit")
    ax_n.set_ylabel("$|n_d|$ (g)", fontsize=10)
    ax_n.legend(loc="upper right", fontsize=8, ncol=5)
    ax_n.grid(True, alpha=0.3)

    ax_m.set_yticks([0,1,2,3,4])
    ax_m.set_yticklabels(["INIT","FOV_TRK","LOCKED","MISSED","ABANDON"], fontsize=8)
    ax_m.set_ylabel("State", fontsize=10)
    ax_m.set_xlabel("Time (s)", fontsize=10)
    ax_m.legend(loc="upper right", fontsize=8, ncol=4)
    ax_m.grid(True, alpha=0.3)

    # hit/death markers
    for i in range(n_off):
        ds = None
        for s in range(len(d["off_alive"][i])):
            if not d["off_alive"][i][s] and not d["off_hit"][i][s]:
                ds = s; break
        if ds:
            for ax in axes:
                ax.axvline(ds*0.01, color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
    if sm.get("hit_time_s"):
        for ax in axes:
            ax.axvline(sm["hit_time_s"], color="green", ls="-", lw=1.5, alpha=0.8)

    plt.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(f"{base}/fig3_def_overload.pdf", bbox_inches="tight", dpi=100)
    fig.savefig(f"{base}/fig3_def_overload.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  [fig3_def] {base}/fig3_def_overload.pdf")


def main():
    root = "/tmp/v71_paper"
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if not os.path.isdir(base) or not case.startswith("case"): continue
        if not os.path.exists(f"{base}/trajectory_data.npz"): continue
        print(f"\n>>> {case}")
        process_case(base, case)
    print("\nDone.")


if __name__ == "__main__":
    main()
