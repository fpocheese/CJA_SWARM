#!/usr/bin/env python
"""V71 paper figure generator — IEEE-journal style (array-format data).

Produces per case:
  fig1_offensive.pdf   Offensive kinematics (V, n_p, n_y, distance bundle)
  fig2_defensive.pdf   Defender Gantt + assignment cost + R(t) + pitch/yaw overload
  fig3_game.pdf        Game priors (Phi, N_eff, roles, P_pen, lock, Gamma/Xi, P_hit/E_esc)
  fig4_traj3d.pdf      3-D engagement trajectory with start/end markers and arrows
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.35,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.6",
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

WIN = 20
DT = 0.01
OFF_COL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
DEF_COL = ["#17becf", "#ff7f0e", "#2a363b", "#e3b505"]
G = 9.80665


def smooth(a, w=WIN):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 3:
        return a
    w = min(w, len(a))
    return np.convolve(a, np.ones(w)/w, mode="same")


def to_float(arr):
    return np.asarray(arr, dtype=np.float64)


def death_step(off_alive, off_hit, i):
    alive = to_float(off_alive[i])
    hit = to_float(off_hit[i])
    if np.any(hit == 1):
        return None
    dead = np.where(alive == 0)[0]
    return int(dead[0]) if dead.size else None


def draw_markers(axes, t, ht, death_steps):
    if not isinstance(axes, (list, np.ndarray, tuple)):
        axes = [axes]
    for i, ds in death_steps.items():
        if ds is None:
            continue
        for ax in axes:
            ax.axvline(t[ds], color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
    if ht is not None:
        for ax in axes:
            ax.axvline(ht, color="#145214", ls="-", lw=1.2, alpha=0.85)


def compute_def_np_ny(d, j):
    """Compute defender pitch/yaw overload from position numerical differentiation."""
    x = to_float(d["def_x"][j])
    y = to_float(d["def_y"][j])
    z = to_float(d["def_z"][j])
    alive = to_float(d["def_alive"][j])

    vx = np.gradient(x, DT)
    vy = np.gradient(y, DT)
    vz = np.gradient(z, DT)
    V = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-9

    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)
    az = np.gradient(vz, DT)

    # unit tangent
    tx, ty, tz = vx/V, vy/V, vz/V
    # tangential acceleration (along velocity)
    a_tang = ax*tx + ay*ty + az*tz
    # normal acceleration vector
    anx = ax - a_tang*tx
    any_ = ay - a_tang*ty
    anz = az - a_tang*tz

    # pitch direction: in the vertical plane containing velocity
    # approximate: project normal accel onto vertical vs horizontal
    # pitch overload ~ vertical component of normal accel
    # yaw overload ~ horizontal component of normal accel
    Vhor = np.sqrt(vx**2 + vy**2) + 1e-9
    # pitch normal: perpendicular to velocity in vertical plane
    # = (-vx*vz/Vhor/V, -vy*vz/Vhor/V, Vhor/V)
    pnx = -tx*tz / (Vhor/V + 1e-9)
    pny = -ty*tz / (Vhor/V + 1e-9)
    pnz = Vhor / V

    # yaw normal: perpendicular to velocity in horizontal plane
    # = (-vy/Vhor, vx/Vhor, 0)
    ynx = -vy / Vhor
    yny = vx / Vhor

    n_pitch = (anx*pnx + any_*pny + anz*pnz) / G
    n_yaw = (anx*ynx + any_*yny) / G

    # mask dead steps
    n_pitch[alive == 0] = np.nan
    n_yaw[alive == 0] = np.nan
    return n_pitch, n_yaw


# ═══════════════ FIG 1: offensive kinematics ═══════════════
def fig1_offensive(base, d, sm):
    t = to_float(d["time"])
    n_off = d["off_x"].shape[0]
    n_def = d["def_x"].shape[0]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d["off_alive"], d["off_hit"], i) for i in range(n_off)}

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = GridSpec(4, 1, hspace=0.38, left=0.09, right=0.985, top=0.985, bottom=0.06)
    axV, axNp, axNy, axR = [fig.add_subplot(gs[k]) for k in range(4)]

    for i in range(n_off):
        alive = to_float(d["off_alive"][i])
        hit = to_float(d["off_hit"][i])
        mask = (alive == 1) | (hit == 1)
        if not mask.any():
            continue
        lw = 1.8 if i == hitter else 1.0
        lbl = rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else "")
        axV.plot(t[mask], to_float(d["off_v"][i])[mask], color=OFF_COL[i], lw=lw, label=lbl)
        axNp.plot(t[mask], smooth(to_float(d["off_an_pitch"][i]) / G)[mask], color=OFF_COL[i], lw=lw)
        axNy.plot(t[mask], smooth(to_float(d["off_an_yaw"][i]) / G)[mask], color=OFF_COL[i], lw=lw)

    axV.set_ylabel(r"$V_i$ (m s$^{-1}$)")
    axV.legend(loc="upper right", ncol=4, handlelength=1.4)
    axV.text(0.5, -0.22, "(a)", transform=axV.transAxes, fontweight='bold', fontsize=9, ha='center')

    axNp.axhline(1, color="0.55", ls="--", lw=0.6)
    axNp.set_ylabel(r"$n_{p,i}$ (g)")
    axNp.text(0.5, -0.22, "(b)", transform=axNp.transAxes, fontweight='bold', fontsize=9, ha='center')

    axNy.axhline(0, color="0.55", ls="-", lw=0.5)
    axNy.set_ylabel(r"$n_{y,i}$ (g)")
    axNy.text(0.5, -0.22, "(c)", transform=axNy.transAxes, fontweight='bold', fontsize=9, ha='center')

    dh = to_float(d["off_d_hvt"][hitter])
    dnear = np.full(len(t), np.nan)
    ox = to_float(d["off_x"][hitter])
    oy = to_float(d["off_y"][hitter])
    oz = to_float(d["off_z"][hitter])
    for j in range(n_def):
        dd = np.sqrt((to_float(d["def_x"][j])-ox)**2 +
                     (to_float(d["def_y"][j])-oy)**2 +
                     (to_float(d["def_z"][j])-oz)**2)
        dd[to_float(d["def_alive"][j]) == 0] = 1e9
        dnear = np.fmin(dnear, dd)
    dnear[dnear > 1e8] = np.nan

    axR.semilogy(t, dh, color=OFF_COL[hitter], lw=1.8, label=rf"$\rho_{{{hitter}H}}$")
    axR.semilogy(t, dnear, color="#ff7f0e", lw=1.25, ls="--", label=rf"$\min_{{j}}\rho_{{{hitter}j}}$")
    axR.axhline(5, color="#b22222", ls=":", lw=0.8)
    axR.text(t[-1]*0.97, 6.5, r"$\rho^{\mathrm{kill}}{=}5$m", fontsize=7, color="#b22222", ha="right")
    axR.set_ylabel(r"Distance (m)")
    axR.set_xlabel(r"Time $t$ (s)")
    axR.legend(loc="upper right", ncol=1, handlelength=1.6)
    axR.text(0.5, -0.22, "(d)", transform=axR.transAxes, fontweight='bold', fontsize=9, ha='center')

    for ax in (axV, axNp, axNy, axR):
        ax.set_xlim(t[0], t[-1])
    draw_markers([axV, axNp, axNy, axR], t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig1_offensive.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig1_offensive.pdf")


# ═══════════════ FIG 2: defensive ═══════════════
def fig2_defensive(base, d, sm, gd):
    t = to_float(d["time"])
    n_off = d["off_x"].shape[0]
    n_def = d["def_x"].shape[0]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d["off_alive"], d["off_hit"], i) for i in range(n_off)}

    fig = plt.figure(figsize=(7.8, 11.0))
    gs = GridSpec(5, 1, hspace=0.38, left=0.09, right=0.985, top=0.985, bottom=0.048,
                  height_ratios=[1.0, 0.85, 0.85, 0.85, 0.85])
    axG, axC, axR, axNp, axNy = [fig.add_subplot(gs[k]) for k in range(5)]

    # (a) Gantt
    dl = to_float(d["def_ltgt"])  # (n_def, T)
    target_colors = {0: OFF_COL[0], 1: OFF_COL[1], 2: OFF_COL[2], 3: OFF_COL[3], -1: "#dcdcdc"}
    axG.set_xlim(t[0], t[-1])
    axG.set_ylim(0, 1)
    for j in range(n_def):
        tgts = dl[j].astype(int)
        N = len(tgts)
        seg_start = 0
        for k in range(1, N + 1):
            if k == N or tgts[k] != tgts[seg_start]:
                tgt = int(tgts[seg_start])
                c = target_colors.get(tgt, "#dcdcdc")
                axG.axvspan(t[seg_start], t[k-1],
                            ymin=j/4 + 0.025, ymax=(j+1)/4 - 0.025,
                            alpha=0.85, color=c, lw=0)
                mid_t = 0.5 * (t[seg_start] + t[k-1])
                if (t[k-1] - t[seg_start]) > 1.5:
                    x_ax = (mid_t - t[0]) / (t[-1] - t[0])
                    lbl = rf"$A_{{{tgt}}}$" if tgt >= 0 else r"$\varnothing$"
                    axG.text(x_ax, (j + 0.5)/4, lbl,
                             ha="center", va="center", fontsize=8, fontweight="bold",
                             color="white" if tgt >= 0 else "#444",
                             transform=axG.transAxes)
                seg_start = k
        axG.axhline(j/4, color="white", lw=1.2)
    axG.axhline(1, color="white", lw=1.2)
    axG.set_yticks([0.125, 0.375, 0.625, 0.875])
    axG.set_yticklabels([rf"$D_{{{k}}}$" for k in range(n_def)])
    axG.set_ylabel("Defender")
    axG.grid(False)
    axG.text(0.5, -0.22, "(a)", transform=axG.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (b) assignment cost to hitter
    cm = to_float(d["assign_cost"])  # (T, n_def, n_off)
    T_ = cm.shape[0]
    for j in range(n_def):
        axC.plot(t[:T_], cm[:T_, j, hitter], color=DEF_COL[j], lw=1.2,
                 label=rf"$c_{{{j}\to{hitter}}}$")
    axC.set_ylabel(r"$c_{j\to A^{\star}}$ (m)")
    axC.legend(loc="upper right", ncol=4, handlelength=1.4)
    axC.text(0.5, -0.22, "(b)", transform=axC.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (c) R(t) — fixed dtype and handling of unassigned defenders
    R_t = np.full(T_, np.nan)
    def_alive_f = to_float(d["def_alive"])
    for s in range(T_):
        m = cm[s]
        ri, ci = linear_sum_assignment(m)
        opt = m[ri, ci].sum()
        tot = 0.0; n = 0
        for j in range(n_def):
            tgt = int(dl[j, s])
            if def_alive_f[j, s] == 1 and tgt >= 0:
                tot += m[j, tgt]; n += 1
        if n == 0 or opt < 1e-6:
            continue
        R_t[s] = tot / (opt * n / 4.0)
    axR.plot(t[:T_], R_t, color="#c0392b", lw=1.6)
    axR.axhline(1.0, color="0.3", ls="--", lw=0.9)
    valid_R = R_t[~np.isnan(R_t)]
    if valid_R.size > 0:
        axR.fill_between(t[:T_], 1.0, R_t, where=(~np.isnan(R_t)) & (R_t > 1.0),
                         alpha=0.18, color="#c0392b")
    axR.set_ylabel(r"$R(t)$")
    axR.text(0.5, -0.22, "(c)", transform=axR.transAxes, fontweight='bold', fontsize=9, ha='center')
    if valid_R.size > 0:
        axR.set_ylim(bottom=max(0.5, valid_R.min()-0.1))

    # (d) defender pitch overload
    for j in range(n_def):
        alive = to_float(d["def_alive"][j]) == 1
        if not alive.any():
            continue
        n_pitch, n_yaw = compute_def_np_ny(d, j)
        n_pitch_s = smooth(n_pitch)
        n_yaw_s = smooth(n_yaw)
        axNp.plot(t[alive], n_pitch_s[alive], color=DEF_COL[j], lw=1.25,
                  label=rf"$D_{{{j}}}$")
        axNy.plot(t[alive], n_yaw_s[alive], color=DEF_COL[j], lw=1.25,
                  label=rf"$D_{{{j}}}$")
    axNp.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNp.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNp.set_ylabel(r"$n_{p,D_{j}}$ (g)")
    axNp.legend(loc="upper right", ncol=4, handlelength=1.4)
    axNp.text(0.5, -0.22, "(d)", transform=axNp.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (e) defender yaw overload
    axNy.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNy.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNy.set_ylabel(r"$n_{y,D_{j}}$ (g)")
    axNy.set_xlabel(r"Time $t$ (s)")
    axNy.legend(loc="upper right", ncol=4, handlelength=1.4)
    axNy.text(0.5, -0.22, "(e)", transform=axNy.transAxes, fontweight='bold', fontsize=9, ha='center')

    for ax in (axG, axC, axR, axNp, axNy):
        ax.set_xlim(t[0], t[-1])
    draw_markers([axG, axC, axR, axNp, axNy], t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig2_defensive.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig2_defensive.pdf")


# ═══════════════ FIG 3: game priors ═══════════════
def fig3_game(base, d, sm, gd):
    t = to_float(d["time"])
    n_off = d["off_x"].shape[0]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d["off_alive"], d["off_hit"], i) for i in range(n_off)}

    fig = plt.figure(figsize=(7.8, 10.8))
    gs = GridSpec(6, 1, hspace=0.42, left=0.1, right=0.95, top=0.985, bottom=0.045)
    axPhi, axRole, axPen, axLock, axGX, axPH = [fig.add_subplot(gs[k]) for k in range(6)]

    # (a) Phi_decoy + N_eff
    phi = to_float(gd["decoy_Phi"])
    axPhi.plot(t[:len(phi)], phi, color="#0072B2", lw=1.8, label=r"$\Phi_{\mathrm{decoy}}(t)$")
    axPhi.set_ylabel(r"$\Phi_{\mathrm{decoy}}$", color="#0072B2")
    axPhi.tick_params(axis="y", labelcolor="#0072B2")
    neff = to_float(gd["pen_N_eff"])
    a2 = axPhi.twinx()
    a2.plot(t[:len(neff)], neff, color="#D55E00", lw=1.6, ls="--", label=r"$N_{\mathrm{eff}}(t)$")
    a2.set_ylabel(r"$N_{\mathrm{eff}}$", color="#D55E00")
    a2.tick_params(axis="y", labelcolor="#D55E00")
    a2.grid(False)
    lines1, labels1 = axPhi.get_legend_handles_labels()
    lines2, labels2 = a2.get_legend_handles_labels()
    axPhi.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, handlelength=1.6)
    axPhi.text(0.5, -0.22, "(a)", transform=axPhi.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (b) role probabilities
    rd = to_float(gd["decoy_role_decoy"])
    rp = to_float(gd["decoy_role_pen"])
    for i in range(n_off):
        lw = 1.8 if i == hitter else 1.0
        axRole.plot(t, rd[i], color=OFF_COL[i], lw=lw,
                    label=rf"$\pi^{{D}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        axRole.plot(t, rp[i], color=OFF_COL[i], lw=0.9, ls="--", alpha=0.75)
    axRole.set_ylabel(r"$\pi^{r}_{i}(t)$")
    axRole.set_ylim(-0.03, 1.05)
    axRole.legend(loc="upper right", ncol=4, handlelength=1.6)
    axRole.annotate(r"solid: $\pi^{D}$   dashed: $\pi^{P}$",
                    xy=(0.12, 1.02), xycoords="axes fraction", fontsize=7.5, ha="left", va="bottom")
    axRole.text(0.5, -0.22, "(b)", transform=axRole.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (c) P_pen
    pp = to_float(gd["pen_P_pen"])
    for i in range(n_off):
        axPen.plot(t, pp[i], color=OFF_COL[i], lw=1.6 if i == hitter else 1.0,
                   label=rf"$P^{{\mathrm{{pen}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axPen.set_ylabel(r"$P^{\mathrm{pen}}_{i}(t)$")
    axPen.set_ylim(-0.03, 1.05)
    axPen.legend(loc="upper right", ncol=4, handlelength=1.4)
    axPen.text(0.5, -0.22, "(c)", transform=axPen.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (d) lock pressure
    lp = to_float(gd["decoy_lock_pressure"])
    for i in range(n_off):
        axLock.plot(t, lp[i], color=OFF_COL[i], lw=1.6 if i == hitter else 1.0,
                    label=rf"$\lambda_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axLock.set_ylabel(r"$\lambda_{i}(t)$")
    axLock.legend(loc="upper right", ncol=4, handlelength=1.4)
    axLock.text(0.5, -0.22, "(d)", transform=axLock.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (e) Gamma_mean / Xi_mean
    gm = to_float(gd["esc_Gamma_mean"])
    xi = to_float(gd["esc_Xi_mean"])
    axGX.plot(t[:len(gm)], gm, color="#009E73", lw=1.6, label=r"$\bar{\Gamma}(t)$")
    axGX.plot(t[:len(xi)], xi, color="#CC79A7", lw=1.4, ls="--", label=r"$\bar{\Xi}(t)$")
    axGX.axhline(0, color="0.4", ls=":", lw=0.7)
    axGX.set_ylabel(r"$\bar{\Gamma},\ \bar{\Xi}$")
    axGX.legend(loc="upper right", ncol=2, handlelength=1.8)
    axGX.text(0.5, -0.22, "(e)", transform=axGX.transAxes, fontweight='bold', fontsize=9, ha='center')

    # (f) P_hit + E_esc
    ph = to_float(gd["hvt_P_hit"])
    ee = to_float(gd["esc_E_esc"])
    for i in range(n_off):
        axPH.plot(t, ph[i], color=OFF_COL[i], lw=1.6 if i == hitter else 0.8,
                  alpha=0.9 if i == hitter else 0.55,
                  label=rf"$P^{{\mathrm{{hit}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        axPH.plot(t, ee[i], color=OFF_COL[i], lw=1.1 if i == hitter else 0.6, ls=":",
                  alpha=0.8 if i == hitter else 0.5)
    axPH.set_ylabel(r"$P^{\mathrm{hit}}_{i},\ E^{\mathrm{esc}}_{i}$")
    axPH.set_xlabel(r"Time $t$ (s)")
    axPH.legend(loc="upper left", ncol=4, handlelength=1.4)
    axPH.annotate(r"solid: $P^{\mathrm{hit}}$   dotted: $E^{\mathrm{esc}}$",
                  xy=(0.60, 1.02), xycoords="axes fraction", fontsize=7.5, ha="left", va="bottom")
    axPH.text(0.5, -0.22, "(f)", transform=axPH.transAxes, fontweight='bold', fontsize=9, ha='center')

    for ax in (axPhi, axRole, axPen, axLock, axGX, axPH):
        ax.set_xlim(t[0], t[-1])
    draw_markers([axPhi, axRole, axPen, axLock, axGX, axPH], t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig3_game.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig3_game.pdf")


# ═══════════════ FIG 4: 3-D trajectory (improved) ═══════════════
def fig4_traj3d(base, d, sm):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    t = to_float(d["time"])
    n_off = d["off_x"].shape[0]
    n_def = d["def_x"].shape[0]
    hitter = int(sm["hitter"])
    death_steps_dict = {i: death_step(d["off_alive"], d["off_hit"], i) for i in range(n_off)}

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    def add_arrow_3d(ax, xs, ys, zs, color, idx):
        """Add arrow annotation at a midpoint of the trajectory."""
        if len(xs) < 20:
            return
        mid = len(xs) // 2
        span = min(20, mid)
        dx = xs[mid+span] - xs[mid-span]
        dy = ys[mid+span] - ys[mid-span]
        dz = zs[mid+span] - zs[mid-span]
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-9
        scale = 80
        ax.quiver(xs[mid], ys[mid], zs[mid],
                  dx/norm*scale, dy/norm*scale, dz/norm*scale,
                  color=color, arrow_length_ratio=0.35, lw=2.2)

    # Plot offenders
    for i in range(n_off):
        alive = to_float(d["off_alive"][i])
        hit = to_float(d["off_hit"][i])
        mask = (alive == 1) | (hit == 1)
        if not mask.any():
            continue
        xs = to_float(d["off_x"][i])[mask]
        ys = to_float(d["off_y"][i])[mask]
        zs = to_float(d["off_z"][i])[mask]
        lw = 2.2 if i == hitter else 1.2
        lbl = rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else "")
        ax.plot(xs, ys, zs, color=OFF_COL[i], lw=lw, label=lbl, zorder=5 if i == hitter else 3)
        # Start marker (circle)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=OFF_COL[i], marker='o', s=40, zorder=6, edgecolors='k', linewidths=0.5)
        # End marker
        ds = death_steps_dict.get(i)
        if ds is None and np.any(hit == 1):
            # hitter — star at end
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=OFF_COL[i], marker='*', s=120, zorder=6, edgecolors='k', linewidths=0.5)
        elif ds is not None:
            # killed — X at end
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=OFF_COL[i], marker='X', s=60, zorder=6, edgecolors='k', linewidths=0.5)
        # Direction arrow at midpoint
        add_arrow_3d(ax, xs, ys, zs, OFF_COL[i], i)

    # Plot defenders
    for j in range(n_def):
        alive = to_float(d["def_alive"][j]) == 1
        if not alive.any():
            continue
        xs = to_float(d["def_x"][j])[alive]
        ys = to_float(d["def_y"][j])[alive]
        zs = to_float(d["def_z"][j])[alive]
        ax.plot(xs, ys, zs, color=DEF_COL[j], lw=1.0, ls="--", label=rf"$D_{{{j}}}$", alpha=0.8)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=DEF_COL[j], marker='s', s=30, zorder=4, edgecolors='k', linewidths=0.4)
        add_arrow_3d(ax, xs, ys, zs, DEF_COL[j], j)

    # HVT
    hvt_x, hvt_y, hvt_z = float(d["hvt_x"]), float(d["hvt_y"]), float(d["hvt_z"])
    ax.scatter([hvt_x], [hvt_y], [hvt_z], marker="*", color="#ffbf00", edgecolor="k",
               s=200, zorder=10, label="HVT")

    ax.set_xlabel(r"$x$ (m)", labelpad=6)
    ax.set_ylabel(r"$y$ (m)", labelpad=6)
    ax.set_zlabel(r"$z$ (m)", labelpad=4)
    ax.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.9)
    ax.view_init(elev=25, azim=-55)
    fig.savefig(os.path.join(base, "fig4_traj3d.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig4_traj3d.pdf")


# ─────────────────────────── main ───────────────────────────
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"root={root}")
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if not os.path.isdir(base) or not os.path.exists(os.path.join(base, "trajectory_data.npz")):
            continue
        print(f"\n>>> {case}")
        d = dict(np.load(os.path.join(base, "trajectory_data.npz"), allow_pickle=True))
        with open(os.path.join(base, "summary.json")) as fh:
            sm = json.load(fh)
        gd = {}
        gp = os.path.join(base, "game_data.npz")
        if os.path.exists(gp):
            gd = dict(np.load(gp, allow_pickle=True))
        fig1_offensive(base, d, sm)
        fig2_defensive(base, d, sm, gd)
        fig3_game(base, d, sm, gd)
        fig4_traj3d(base, d, sm)
    print("\nAll done.")


if __name__ == "__main__":
    main()
