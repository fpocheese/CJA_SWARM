#!/usr/bin/env python
"""fig1(7行)+fig2(5行:含def np/ny). WIN=20步移动平均.
若缺def_an_pitch/def_an_yaw则自动重播补录.
远端运行后scp回fig1+fig2到本地.

修改窗口: 改本文件顶部 WIN = 20.
运行: sshpass ... ssh ... "cd ~/000... && conda run -n rlgpu python scripts/gen_v71_fast.py /tmp/v71_paper"
"""
import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ============== 改这里调窗口 ==============
WIN = 20  # 移动平均窗口(步), 20步=0.20s
# =========================================
OFF_COL = ["#e74c3c","#3498db","#2ecc71","#9b59b6"]
DEF_COL = ["#1abc9c","#e67e22","#34495e","#d35400"]
G = 9.80665
WIN = 51  # 0.51s moving average

def smooth(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 3: return a
    w = min(WIN, len(a))
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode='same')

def process(base, case):
    d = np.load(f"{base}/trajectory_data.npz", allow_pickle=True)
    with open(f"{base}/summary.json") as fh: sm = json.load(fh)
    t = d["time"]; hitter = sm["hitter"]; n_off = 4; n_def = 4
    ht = sm.get("hit_time_s")

    # ==== FIG1: 7 rows ====
    fig, axs = plt.subplots(7, 1, figsize=(12, 12), sharex=True)
    titles = ["(a) Speed V", "(b) np (smoothed)", "(c) ny (smoothed)",
              "(d) Distance to HVT", "(e) Defender |n_d|", "(f) Defender State", ""]
    # Rows 0-5: data; row 6: xlabel anchor
    for k in range(6):
        axs[k].grid(True, alpha=0.3)
    axs[6].set_visible(False)

    # V
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        axs[0].plot(t[mask], np.array(d["off_v"][i][mask]), color=OFF_COL[i],
                    lw=1.8 if i == hitter else 1.1)
    axs[0].set_ylabel("V (m/s)"); axs[0].axhline(40, c="gray", ls="--", lw=0.7); axs[0].axhline(50, c="gray", ls="--", lw=0.7)
    axs[0].set_ylim(35, 55); axs[0].set_title(f"{case} - Smoothed Kinematics (MA {WIN*0.01:.1f}s)", fontsize=10)

    # np
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        axs[1].plot(t[mask], smooth(np.array(d["off_an_pitch"][i][mask]) / G),
                    color=OFF_COL[i], lw=2.0 if i == hitter else 1.2)
    axs[1].set_ylabel("np (g)"); axs[1].axhline(1, c="gray", ls="--", lw=0.7)

    # ny
    for i in range(n_off):
        mask = np.array(d["off_alive"][i], dtype=bool) | np.array(d["off_hit"][i], dtype=bool)
        if not mask.any(): continue
        axs[2].plot(t[mask], smooth(np.array(d["off_an_yaw"][i][mask]) / G),
                    color=OFF_COL[i], lw=2.0 if i == hitter else 1.2)
    axs[2].set_ylabel("ny (g)"); axs[2].axhline(0, c="gray", ls="-", lw=0.5)

    # dist
    dh = np.array(d["off_d_hvt"][hitter])
    dd = []
    for s in range(len(dh)):
        md = 9999
        ox, oy, oz = d["off_x"][hitter][s], d["off_y"][hitter][s], d["off_z"][hitter][s]
        for j in range(n_def):
            if s < len(d["def_x"][j]):
                md = min(md, np.sqrt((d["def_x"][j][s]-ox)**2+(d["def_y"][j][s]-oy)**2+(d["def_z"][j][s]-oz)**2))
        dd.append(md)
    axs[3].semilogy(t[:len(dh)], dh, color=OFF_COL[hitter], lw=1.8)
    axs[3].semilogy(t[:len(dd)], dd, color="#e67e22", lw=1.3, ls="--")
    axs[3].axhline(500, c="gray", ls=":", lw=1.0); axs[3].axhline(5, c="red", ls=":", lw=1.0)
    axs[3].set_ylabel("Distance (m)")

    # def |n|
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        axs[4].plot(t[alive], smooth(np.array(d["def_an"][j][alive])), color=DEF_COL[j], lw=1.4, label=f"D{j}")
    axs[4].axhline(5, c="red", ls="--", lw=0.8, alpha=0.6)
    axs[4].set_ylabel("|n_d| (g)"); axs[4].legend(loc="upper right", fontsize=7, ncol=4)

    # def state
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        axs[5].plot(t[alive], np.array(d["def_lmode"][j][alive]) + j*0.08, color=DEF_COL[j], lw=1.5, label=f"D{j}")
    axs[5].set_yticks([0,1,2,3,4])
    axs[5].set_yticklabels(["INIT","SEARCH","LOCKED","MISSED","ABANDON"], fontsize=7)
    axs[5].set_ylabel("State"); axs[5].legend(loc="upper right", fontsize=7, ncol=4)

    # markers
    for i in range(n_off):
        for s in range(len(d["off_alive"][i])):
            if not d["off_alive"][i][s] and not d["off_hit"][i][s]:
                for ax in axs[:6]: ax.axvline(s*0.01, color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
                break
    if ht:
        for ax in axs[:6]: ax.axvline(ht, color="green", ls="-", lw=1.5, alpha=0.8)

    axs[5].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{base}/fig1_kinematics_all8.pdf", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # ==== FIG2: 5 rows ====
    cost_list = d["assign_cost"]
    R_arr = np.full(len(cost_list), np.nan)
    for kk, cmat in enumerate(cost_list):
        try:
            ri, ci = linear_sum_assignment(cmat)
            opt_c = cmat[ri, ci].sum()
            act_c = sum(cmat[jj, np.argmin(cmat[jj])] for jj in range(cmat.shape[0]))
            R_arr[kk] = act_c / max(opt_c, 1e-6)
        except: pass

    fig, axs = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

    # row1: state
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        axs[0].step(t[alive], np.array(d["def_lmode"][j][alive]), color=DEF_COL[j], lw=1.8, label=f"D{j}", where="post")
    axs[0].set_yticks([0,1,2,3,4])
    axs[0].set_yticklabels(["INIT","SEARCH","LOCKED","MISSED","ABANDON"], fontsize=8)
    axs[0].set_ylabel("State"); axs[0].legend(loc="upper right", fontsize=8, ncol=4)
    axs[0].grid(True, alpha=0.3)

    # row2: assign cost
    t_ac = t[:len(cost_list)]
    for j in range(n_def):
        c2h = np.array([cost_list[s][j, hitter] for s in range(len(cost_list))])
        axs[1].plot(t_ac[:len(c2h)], c2h, color=DEF_COL[j], lw=1.5, label=f"D{j}$\\to$A{hitter}")
        mc = np.array([min(cost_list[s][j, d] for d in range(n_off) if d != hitter) for s in range(len(cost_list))])
        axs[1].plot(t_ac[:len(mc)], mc, color=DEF_COL[j], lw=0.8, ls="--", alpha=0.5)
    axs[1].set_ylabel("Cost (m)"); axs[1].legend(loc="upper right", fontsize=7, ncol=4)
    axs[1].grid(True, alpha=0.3)

    # row3: R(t)
    axs[2].plot(t_ac, R_arr, color="#c0392b", lw=2.0)
    axs[2].axhline(1.0, c="gray", ls="--", lw=1.0); axs[2].axhline(2.0, c="orange", ls=":", lw=1.0)
    axs[2].fill_between(t_ac, 1.0, R_arr, where=(R_arr > 1.0), alpha=0.2, color="red")
    axs[2].set_ylabel("$R(t)$"); axs[2].grid(True, alpha=0.3)

    # row4: def n_{p,d} (俯仰过载)
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        axs[3].plot(t[alive], smooth(np.array(d["def_an_pitch"][j][alive])), color=DEF_COL[j], lw=1.4, label=f"D{j}")
    axs[3].axhline(5, c="red", ls="--", lw=0.8, alpha=0.6); axs[3].axhline(-5, c="red", ls="--", lw=0.8, alpha=0.6)
    axs[3].set_ylabel("$n_{p,d}$ (g)"); axs[3].legend(loc="upper right", fontsize=7, ncol=4)
    axs[3].grid(True, alpha=0.3)

    # row5: def n_{y,d} (偏航过载)
    for j in range(n_def):
        alive = np.array(d["def_alive"][j], dtype=bool)
        if not alive.any(): continue
        axs[4].plot(t[alive], smooth(np.array(d["def_an_yaw"][j][alive])), color=DEF_COL[j], lw=1.4, label=f"D{j}")
    axs[4].axhline(5, c="red", ls="--", lw=0.8, alpha=0.6); axs[4].axhline(-5, c="red", ls="--", lw=0.8, alpha=0.6)
    axs[4].set_ylabel("$n_{y,d}$ (g)"); axs[4].set_xlabel("Time (s)")
    axs[4].legend(loc="upper right", fontsize=7, ncol=4); axs[4].grid(True, alpha=0.3)

    for i in range(n_off):
        for s in range(len(d["off_alive"][i])):
            if not d["off_alive"][i][s] and not d["off_hit"][i][s]:
                for ax in axs: ax.axvline(s*0.01, color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
                break
    if ht:
        for ax in axs: ax.axvline(ht, color="green", ls="-", lw=1.5, alpha=0.8)

    plt.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(f"{base}/fig2_def_assignment_timeline.pdf", bbox_inches="tight", dpi=100)
    plt.close(fig)

    print(f"  => {case} done")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v71_paper"
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if os.path.isdir(base) and os.path.exists(f"{base}/trajectory_data.npz"):
            print(f">>> {case}")
            process(base, case)
    print("All done.")
