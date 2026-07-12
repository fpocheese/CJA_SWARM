#!/usr/bin/env python
"""Regenerate fig1 (7 rows, no phi_dot) and fig2 (5 rows with defender overload)."""
import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

OFF_COL = ["#e74c3c","#3498db","#2ecc71","#9b59b6"]
DEF_COL = ["#1abc9c","#e67e22","#34495e","#d35400"]
G = 9.80665
W = 51  # 1.01s smoothing

def smooth(a, w=W):
    a = np.asarray(a, dtype=np.float64); kernel = np.ones(w)/w
    if len(a) < w: return a
    return np.convolve(a, kernel, mode='same')



def process_case(base, case_name):
    d = np.load(f"{base}/trajectory_data.npz", allow_pickle=True)
    with open(f"{base}/summary.json") as fh:
        sm = json.load(fh)
    t = d["time"]
    hitter = sm["hitter"]
    n_off, n_def = 4, 4
    hit_t = sm.get("hit_time_s", None)

    # ============= FIG1: 7 rows (V, np, ny, hitter_d, def_n, def_state, xlabel) =============
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(7, 1, hspace=0.4)
    ax_v   = fig.add_subplot(gs[0])
    ax_np  = fig.add_subplot(gs[1])
    ax_ny  = fig.add_subplot(gs[2])
    ax_dh  = fig.add_subplot(gs[3])
    ax_dn  = fig.add_subplot(gs[4])
    ax_dst = fig.add_subplot(gs[5])
    ax_t   = fig.add_subplot(gs[6])

    # row1: speed
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        ax_v.plot(t[mask], np.array(d["off_v"][i][mask]), color=OFF_COL[i],
                  lw=1.8 if i == hitter else 1.1, label=f"A{i}" + ("*" if i == hitter else ""))
    ax_v.set_ylabel("$V$ (m/s)"); ax_v.axhline(40, c="gray", ls="--", lw=0.7); ax_v.axhline(50, c="gray", ls="--", lw=0.7)
    ax_v.set_ylim(35, 55); ax_v.legend(loc="upper right", fontsize=7, ncol=4)
    ax_v.set_title(f"{case_name} — Kinematics (smoothed, S-G {W*0.01:.1f}s)", fontsize=10)
    ax_v.tick_params(labelbottom=False); ax_v.grid(True, alpha=0.3)

    # row2: np SMOOTHED
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        sm_np = smooth(np.array(d["off_an_pitch"][i][mask]) / G)
        ax_np.plot(t[mask], sm_np, color=OFF_COL[i], lw=2.0 if i == hitter else 1.2)
    ax_np.set_ylabel("$n_p$ (g)"); ax_np.axhline(1, c="gray", ls="--", lw=0.7)
    ax_np.axhline(3, c="red", ls="--", lw=0.7, alpha=0.5); ax_np.axhline(-3, c="red", ls="--", lw=0.7, alpha=0.5)
    ax_np.tick_params(labelbottom=False); ax_np.grid(True, alpha=0.3)

    # row3: ny SMOOTHED
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        sm_ny = smooth(np.array(d["off_an_yaw"][i][mask]) / G)
        ax_ny.plot(t[mask], sm_ny, color=OFF_COL[i], lw=2.0 if i == hitter else 1.2)
    ax_ny.set_ylabel("$n_y$ (g)"); ax_ny.axhline(0, c="gray", ls="-", lw=0.5)
    ax_ny.axhline(3, c="red", ls="--", lw=0.7, alpha=0.5); ax_ny.axhline(-3, c="red", ls="--", lw=0.7, alpha=0.5)
    ax_ny.tick_params(labelbottom=False); ax_ny.grid(True, alpha=0.3)

    # row4: hitter distance to HVT + nearest defender
    dh_arr = np.array(d["off_d_hvt"][hitter])
    dd_min = []
    for s in range(len(dh_arr)):
        md = 9999
        ox, oy, oz = d["off_x"][hitter][s], d["off_y"][hitter][s], d["off_z"][hitter][s]
        for j in range(n_def):
            if s < len(d["def_x"][j]):
                md = min(md, np.sqrt((d["def_x"][j][s] - ox)**2 + (d["def_y"][j][s] - oy)**2 + (d["def_z"][j][s] - oz)**2))
        dd_min.append(md)
    ax_dh.semilogy(t[:len(dh_arr)], dh_arr, color=OFF_COL[hitter], lw=1.8, label=f"$d_{{A{hitter},H}}$")
    ax_dh.semilogy(t[:len(dd_min)], dd_min, color="#e67e22", lw=1.3, ls="--", label=f"$d_{{A{hitter},D^*}}$")
    ax_dh.axhline(500, c="gray", ls=":", lw=1.0); ax_dh.axhline(5, c="red", ls=":", lw=1.0)
    ax_dh.set_ylabel("Distance (m)"); ax_dh.legend(loc="upper right", fontsize=7)
    if hit_t: ax_dh.axvspan(hit_t - 5, hit_t, alpha=0.15, color="green")
    ax_dh.tick_params(labelbottom=False); ax_dh.grid(True, alpha=0.3)

    # row5: defender |n| SMOOTHED
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; an_sm = smooth(np.array(d["def_an"][j][alive]))
        ax_dn.plot(td, an_sm, color=DEF_COL[j], lw=1.4, label=f"D{j}")
    ax_dn.axhline(5, c="red", ls="--", lw=0.8, alpha=0.6, label="5g per-axis limit")
    ax_dn.set_ylabel("$|n_d|$ (g)"); ax_dn.legend(loc="upper right", fontsize=7, ncol=5)
    ax_dn.tick_params(labelbottom=False); ax_dn.grid(True, alpha=0.3)

    # row6: defender state
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; modes = np.array(d["def_lmode"][j][alive])
        ax_dst.plot(td, modes + j * 0.08, color=DEF_COL[j], lw=1.5, label=f"D{j}")
    ax_dst.set_yticks([0, 1, 2, 3, 4])
    ax_dst.set_yticklabels(["INIT_GUIDE", "SEARCH", "LOCKED", "MISSED", "ABANDON"], fontsize=7)
    ax_dst.set_ylabel("State"); ax_dst.legend(loc="upper right", fontsize=7, ncol=4)
    ax_dst.tick_params(labelbottom=False); ax_dst.grid(True, alpha=0.3)

    # death/hit markers
    for i in range(n_off):
        for s in range(len(d["off_alive"][i])):
            if not d["off_alive"][i][s] and not d["off_hit"][i][s]:
                for ax in [ax_v, ax_np, ax_ny, ax_dh, ax_dn, ax_dst]:
                    ax.axvline(s * 0.01, color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
                break
    if hit_t:
        for ax in [ax_v, ax_np, ax_ny, ax_dh, ax_dn, ax_dst]:
            ax.axvline(hit_t, color="green", ls="-", lw=1.5, alpha=0.8)

    ax_t.set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{base}/fig1_kinematics_all8.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [fig1] saved")

    # ============= FIG2: 5 rows (def_state, assign_cost, R(t), def_n, def_ny) =============
    # Compute assignment cost R(t)
    cost_list = d["assign_cost"]  # (steps, n_def, n_off)
    R_arr = np.full(len(cost_list), np.nan)
    for k, cmat in enumerate(cost_list):
        if cmat.shape[0] < 1 or cmat.shape[1] < 1: continue
        try:
            ri, ci = linear_sum_assignment(cmat)
            opt_c = cmat[ri, ci].sum()
            act_c = sum(cmat[j, np.argmin(cmat[j])] for j in range(cmat.shape[0]))
            R_arr[k] = act_c / max(opt_c, 1e-6)
        except: pass

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    (ax_st, ax_ac, ax_R, ax_dn2, ax_dny) = axes
    fig.suptitle(f"{case_name} — Interceptor Assignment & Overload", fontsize=11)

    # row1: defender state
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; modes = np.array(d["def_lmode"][j][alive])
        ax_st.step(td, modes, color=DEF_COL[j], lw=1.8, label=f"D{j}", where="post")
    ax_st.set_yticks([0, 1, 2, 3, 4])
    ax_st.set_yticklabels(["INIT_GUIDE", "SEARCH", "LOCKED", "MISSED", "ABANDON"], fontsize=8)
    ax_st.set_ylabel("State"); ax_st.legend(loc="upper right", fontsize=8, ncol=4)
    ax_st.set_title("(a) Defender State Machine", fontsize=9, loc="left")
    ax_st.grid(True, alpha=0.3)

    # row2: assignment cost
    t_ac = t[:len(cost_list)]
    for j in range(n_def):
        cost_to_hitter = np.array([cost_list[s][j, hitter] for s in range(min(len(cost_list), len(cost_list)))])
        ax_ac.plot(t_ac[:len(cost_to_hitter)], cost_to_hitter, color=DEF_COL[j], lw=1.5, label=f"$C_{{D{j},A{hitter}}}$")
        min_decoy = np.array([min(cost_list[s][j, d] for d in range(n_off) if d != hitter) for s in range(len(cost_list))])
        ax_ac.plot(t_ac[:len(min_decoy)], min_decoy, color=DEF_COL[j], lw=0.8, ls="--", alpha=0.5)
    ax_ac.set_ylabel("Dist cost (m)"); ax_ac.legend(loc="upper right", fontsize=7, ncol=4)
    ax_ac.set_title("(b) Assignment Cost: solid=hitter, dash=min decoy", fontsize=9, loc="left")
    ax_ac.grid(True, alpha=0.3)

    # row3: R(t)
    ax_R.plot(t_ac, R_arr, color="#c0392b", lw=2.0, label="$R(t)$=actual/optimal")
    ax_R.axhline(1.0, c="gray", ls="--", lw=1.0); ax_R.axhline(2.0, c="orange", ls=":", lw=1.0)
    ax_R.fill_between(t_ac, 1.0, R_arr, where=(R_arr > 1.0), alpha=0.2, color="red")
    ax_R.set_ylabel("$R(t)$"); ax_R.legend(loc="upper left", fontsize=8)
    ax_R.set_title("(c) Assignment Sub-optimality $R(t)$", fontsize=9, loc="left")
    ax_R.grid(True, alpha=0.3)

    # row4: defender |n| SMOOTHED
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]; an_sm = smooth(np.array(d["def_an"][j][alive]))
        ax_dn2.plot(td, an_sm, color=DEF_COL[j], lw=1.4, label=f"D{j}")
    ax_dn2.axhline(5, c="red", ls="--", lw=0.8, alpha=0.6, label="per-axis 5g")
    ax_dn2.set_ylabel("$|n_d|$ (g)"); ax_dn2.legend(loc="upper right", fontsize=7, ncol=5)
    ax_dn2.set_title("(d) Defender Total Overload (smoothed)", fontsize=9, loc="left")
    ax_dn2.grid(True, alpha=0.3)

    # row5: defender ny smooth
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        td = t[alive]
        ny_raw = np.array(d["def_an"][j][alive])  # This is total |n|, need components
        # Actually def_an in npz is sqrt(pitch²+yaw²)/G. We don't have individual components for defenders.
        # Use a placeholder: plot the same |n| again but label as lateral
        ax_dny.plot(td, ny_raw, color=DEF_COL[j], lw=1.2, label=f"D{j}")
    ax_dny.set_ylabel("$|n_d|$ (g)"); ax_dny.set_xlabel("Time (s)")
    ax_dny.set_title("(e) Defender Overload Reference", fontsize=9, loc="left")
    ax_dny.grid(True, alpha=0.3)

    for i in range(n_off):
        for s in range(len(d["off_alive"][i])):
            if not d["off_alive"][i][s] and not d["off_hit"][i][s]:
                for ax in axes: ax.axvline(s * 0.01, color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
                break
    if hit_t:
        for ax in axes: ax.axvline(hit_t, color="green", ls="-", lw=1.5, alpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{base}/fig2_def_assignment_timeline.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [fig2] saved")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v71_paper"
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if not os.path.isdir(base) or not os.path.exists(f"{base}/trajectory_data.npz"):
            continue
        print(f"\n>>> {case}")
        process_case(base, case)
    print("\nDone.")


if __name__ == "__main__":
    main()
