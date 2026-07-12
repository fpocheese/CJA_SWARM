#!/usr/bin/env python
"""V71 paper figure generator — IEEE-journal style.

Produces four figure families per case (CaseA/B/C):
  fig1_offensive.pdf         Offensive kinematics (V, n_p, n_y, d(A_i,H) / d(A_i,D*))
  fig2_defensive.pdf         Defender Gantt + assignment cost + R(t) + interceptor load
  fig3_game.pdf              Game priors: Phi_decoy, N_eff, role probs, lock, Gamma/Xi, P_pen/P_hit/E_esc
  fig4_traj3d.pdf            3-D engagement trajectory (kept from prior version)
  fig5_traj3d_3views.pdf     Multi-view trajectory (kept)

All variable names used in titles / legends match the LaTeX source:
  q_ij, omega_LOS, V_c, Gamma_ij, Xi_ij, Z_tilde, Phi_decoy, U^D, pi^r,
  E_esc, P_pen, P_hit, N_eff, N_loss, rho_ij.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment

# ────────────────────────── IEEE-style rcParams ──────────────────────────
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

WIN = 20  # smoothing window (steps) = 0.2 s at 100 Hz
OFF_COL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]   # A0..A3
DEF_COL = ["#17becf", "#ff7f0e", "#2a363b", "#e3b505"]   # D0..D3
ROLE_COL = {"D": "#d62728", "P": "#2ca02c", "S": "#7f7f7f"}
G = 9.80665


# ─────────────────────────── helpers ────────────────────────────
def smooth(a, w=WIN):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 3:
        return a
    w = min(w, len(a))
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="same")


def arr(v):
    return np.asarray(v, dtype=np.float64)


def as_int_array(v):
    return np.asarray(v, dtype=int)


def death_step(d, i):
    alive = as_int_array([d["off_alive"][i][s] for s in range(len(d["time"]))])
    hit = as_int_array([d["off_hit"][i][s] for s in range(len(d["time"]))])
    dead = np.where(alive == 0)[0]
    hit_idx = np.where(hit == 1)[0]
    if hit_idx.size:
        return None   # hitter, no "death"
    return int(dead[0]) if dead.size else None


def draw_death_and_hit(axes, d, t, ht, death_steps):
    axes = axes if isinstance(axes, (list, np.ndarray, tuple)) else [axes]
    for i, ds in death_steps.items():
        if ds is None:
            continue
        for ax in axes:
            ax.axvline(t[ds], color=OFF_COL[i], ls=":", lw=0.9, alpha=0.6)
    if ht is not None:
        for ax in axes:
            ax.axvline(ht, color="#145214", ls="-", lw=1.2, alpha=0.85)


def fov_style_label_annotation(ax, txt, xy, xytext, color="k"):
    ax.annotate(txt, xy=xy, xytext=xytext,
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.6))


# ═══════════════ FIG 1: offensive kinematics ═══════════════
def fig1_offensive(base, d, sm):
    """Offensive kinematics panel: V_i, n_{p,i}, n_{y,i}, and distance bundle."""
    t = d["time"]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")

    death_steps = {i: death_step(d, i) for i in range(4)}

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = GridSpec(4, 1, hspace=0.22, left=0.09, right=0.985, top=0.985, bottom=0.06)
    axV, axNp, axNy, axR = [fig.add_subplot(gs[k]) for k in range(4)]

    # (a) airspeed V_i
    for i in range(4):
        alive_or_hit = (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
                       (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)
        if not alive_or_hit.any():
            continue
        vv = arr([d["off_v"][i][s] for s in range(len(t))])
        axV.plot(t[alive_or_hit], vv[alive_or_hit],
                 color=OFF_COL[i], lw=1.8 if i == hitter else 1.0,
                 label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axV.axhline(40, color="0.55", ls="--", lw=0.6)
    axV.axhline(50, color="0.55", ls="--", lw=0.6)
    axV.set_ylim(35, 55)
    axV.set_ylabel(r"$V_i$ (m s$^{-1}$)")
    axV.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axV.text(0.005, 0.94, "(a)", transform=axV.transAxes, fontweight='bold', fontsize=9)

    # (b) pitch overload n_p
    for i in range(4):
        alive_or_hit = (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
                       (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)
        if not alive_or_hit.any():
            continue
        np_arr = smooth(arr([d["off_an_pitch"][i][s] for s in range(len(t))]) / G)
        axNp.plot(t[alive_or_hit], np_arr[alive_or_hit],
                  color=OFF_COL[i], lw=1.8 if i == hitter else 1.0)
    axNp.axhline(1, color="0.55", ls="--", lw=0.6)
    axNp.set_ylabel(r"$n_{p,i}$ (g)")
    axNp.text(0.005, 0.94, "(b)", transform=axNp.transAxes, fontweight='bold', fontsize=9)

    # (c) yaw overload n_y
    for i in range(4):
        alive_or_hit = (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
                       (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)
        if not alive_or_hit.any():
            continue
        ny_arr = smooth(arr([d["off_an_yaw"][i][s] for s in range(len(t))]) / G)
        axNy.plot(t[alive_or_hit], ny_arr[alive_or_hit],
                  color=OFF_COL[i], lw=1.8 if i == hitter else 1.0)
    axNy.axhline(0, color="0.55", ls="-", lw=0.5)
    axNy.set_ylabel(r"$n_{y,i}$ (g)")
    axNy.text(0.005, 0.94, "(c)", transform=axNy.transAxes, fontweight='bold', fontsize=9)

    # (d) distance bundle — hitter to HVT and to nearest interceptor
    dh = arr([d["off_d_hvt"][hitter][s] for s in range(len(t))])
    dnear = np.full(len(t), np.nan)
    for s in range(len(t)):
        ox = d["off_x"][hitter][s]; oy = d["off_y"][hitter][s]; oz = d["off_z"][hitter][s]
        md = 1e9
        for j in range(4):
            if as_int_array([d["def_alive"][j][s]])[0] == 1:
                dd = np.sqrt((d["def_x"][j][s]-ox)**2 +
                             (d["def_y"][j][s]-oy)**2 +
                             (d["def_z"][j][s]-oz)**2)
                md = min(md, dd)
        dnear[s] = md if md < 1e9 else np.nan
    axR.semilogy(t, dh, color=OFF_COL[hitter], lw=1.8,
                 label=rf"$\rho_{{{hitter}H}}$  ($A_{{{hitter}}}\to H$)")
    axR.semilogy(t, dnear, color="#ff7f0e", lw=1.25, ls="--",
                 label=rf"$\min_{{j}}\rho_{{{hitter}j}}$  ($A_{{{hitter}}}\to$ nearest $D$)")
    axR.axhline(500, color="0.55", ls=":", lw=0.8)
    axR.axhline(5, color="#b22222", ls=":", lw=0.8)
    axR.text(t[-1]*0.97, 6.5, r"$\rho^{\mathrm{kill}}{=}5$m", fontsize=7,
             color="#b22222", ha="right")
    axR.text(t[-1]*0.97, 650, r"$500$m", fontsize=7, color="0.4", ha="right")
    axR.set_ylabel(r"Distance (m)")
    axR.set_xlabel(r"Time $t$ (s)")
    axR.legend(loc="upper right", ncol=1, handlelength=1.6, borderaxespad=0.3)
    axR.text(0.005, 0.94, "(d)", transform=axR.transAxes, fontweight='bold', fontsize=9)

    for ax in (axV, axNp, axNy, axR):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axV, axNp, axNy, axR], d, t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig1_offensive.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig1_offensive.pdf")


# ═══════════════ FIG 2: defensive assignment / overload ═══════════════
def fig2_defensive(base, d, sm, gd):
    t = d["time"]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d, i) for i in range(4)}

    fig = plt.figure(figsize=(7.8, 8.8))
    gs = GridSpec(4, 1, hspace=0.28, left=0.09, right=0.985, top=0.985, bottom=0.055,
                  height_ratios=[1.0, 0.9, 0.9, 1.0])
    axG, axC, axR, axO = [fig.add_subplot(gs[k]) for k in range(4)]

    # ─── (a) Gantt ───
    dl = gd.get("def_ltgt", d.get("def_ltgt"))
    target_colors = {0: OFF_COL[0], 1: OFF_COL[1], 2: OFF_COL[2], 3: OFF_COL[3], -1: "#dcdcdc"}
    # set x-limits first so that tight-bbox computation for labels is bounded
    axG.set_xlim(t[0], t[-1])
    axG.set_ylim(0, 1)
    for j in range(4):
        tgts = as_int_array([dl[j][s] for s in range(len(t))])
        N = len(tgts)
        seg_start = 0
        for k in range(1, N + 1):
            if k == N or tgts[k] != tgts[seg_start]:
                tgt = int(tgts[seg_start])
                c = target_colors.get(tgt, "#dcdcdc")
                axG.axvspan(t[seg_start], t[k-1],
                            ymin=j/4 + 0.025, ymax=(j+1)/4 - 0.025,
                            alpha=0.85, color=c, lw=0)
                # in-span label: use axes-fraction for both x and y to avoid tight-bbox blow-up
                mid_t = 0.5 * (t[seg_start] + t[k-1])
                if (t[k-1] - t[seg_start]) > 1.5:
                    # convert mid_t from data-units to axes-fraction
                    x_ax = (mid_t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else 0.5
                    lbl = rf"$A_{{{tgt}}}$" if tgt >= 0 else r"$\varnothing$"
                    axG.text(x_ax, (j + 0.5)/4, lbl,
                             ha="center", va="center",
                             fontsize=8, fontweight="bold",
                             color="white" if tgt >= 0 else "#444",
                             transform=axG.transAxes)
                seg_start = k
        axG.axhline(j/4, color="white", lw=1.2)
    axG.axhline(1, color="white", lw=1.2)
    axG.set_yticks([0.125, 0.375, 0.625, 0.875])
    axG.set_yticklabels([rf"$D_{{{k}}}$" for k in range(4)])
    axG.set_ylabel("Defender")
    axG.grid(False)
    axG.text(0.005, 0.93, "(a)", transform=axG.transAxes, fontweight='bold', fontsize=9)

    # ─── (b) assignment cost for hitter ───
    cm = d["assign_cost"]  # (T, 4, 4)
    for j in range(4):
        c2h = arr([cm[s][j, hitter] for s in range(len(cm))])
        axC.plot(t[:len(c2h)], c2h, color=DEF_COL[j], lw=1.2,
                 label=rf"$c_{{{j}\to{hitter}}}$")
    axC.set_ylabel(r"$c_{j\to A^{\star}}$ (m)")
    axC.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axC.text(0.005, 0.93, "(b)", transform=axC.transAxes, fontweight='bold', fontsize=9)

    # ─── (c) R(t) — ratio of actual (greedy / currently-tracked) to Hungarian-optimal ───
    T_ = len(cm)
    R_t = np.full(T_, np.nan)
    def_alive = np.stack([as_int_array([d["def_alive"][j][s] for s in range(T_)]) for j in range(4)])
    ltgt = np.stack([as_int_array([dl[j][s] for s in range(T_)]) for j in range(4)])
    for s in range(T_):
        m = cm[s]
        try:
            ri, ci = linear_sum_assignment(m)
            opt = m[ri, ci].sum()
        except Exception:
            continue
        # actual (realized) assignment cost with currently tracked targets
        tot = 0.0; n = 0
        for j in range(4):
            if def_alive[j, s] and ltgt[j, s] >= 0:
                tot += m[j, ltgt[j, s]]; n += 1
        if n == 0 or opt < 1e-6:
            continue
        # normalize actual to comparable "per-assignment" optimum
        R_t[s] = tot / (opt * n / 4.0)
    axR.plot(t[:T_], R_t, color="#c0392b", lw=1.6)
    axR.axhline(1.0, color="0.3", ls="--", lw=0.9)
    axR.axhline(2.0, color="#e67e22", ls=":", lw=0.9)
    axR.fill_between(t[:T_], 1.0, R_t, where=(R_t > 1.0), alpha=0.18, color="#c0392b")
    axR.set_ylabel(r"$R(t)$")
    axR.text(0.005, 0.93, "(c)", transform=axR.transAxes, fontweight='bold', fontsize=9)
    axR.set_ylim(bottom=0.9)

    # ─── (d) defender overload n_p (solid) / n_y (dashed) ───
    for j in range(4):
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        npp = smooth(arr([d["def_an_pitch"][j][s] for s in range(len(t))]))
        nyy = smooth(arr([d["def_an_yaw"][j][s] for s in range(len(t))]))
        axO.plot(t[alive], npp[alive], color=DEF_COL[j], lw=1.25,
                 label=rf"$n_{{p,D_{{{j}}}}}$")
        axO.plot(t[alive], nyy[alive], color=DEF_COL[j], lw=0.9, ls="--", alpha=0.65)
    axO.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axO.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axO.set_ylabel(r"$n_{p,D_{j}}\ /\ n_{y,D_{j}}$ (g)")
    axO.set_xlabel(r"Time $t$ (s)")
    axO.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axO.text(0.005, 0.93, "(d)", transform=axO.transAxes, fontweight='bold', fontsize=9)

    for ax in (axG, axC, axR, axO):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axG, axC, axR, axO], d, t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig2_defensive.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig2_defensive.pdf")


# ═══════════════ FIG 3: game priors ═══════════════
def fig3_game(base, d, sm, gd):
    t = d["time"]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d, i) for i in range(4)}

    fig = plt.figure(figsize=(7.8, 10.8))
    gs = GridSpec(6, 1, hspace=0.32, left=0.1, right=0.95, top=0.985, bottom=0.045)
    axPhi, axRole, axPen, axLock, axGX, axPH = [fig.add_subplot(gs[k]) for k in range(6)]

    # (a) Phi_decoy (left) + N_eff (right)
    if "decoy_Phi" in gd:
        phi = arr(gd["decoy_Phi"])
        axPhi.plot(t[:len(phi)], phi, color="#0072B2", lw=1.8,
                   label=r"$\Phi_{\mathrm{decoy}}(t)$")
        axPhi.set_ylabel(r"$\Phi_{\mathrm{decoy}}$", color="#0072B2")
        axPhi.tick_params(axis="y", labelcolor="#0072B2")
    if "pen_N_eff" in gd:
        neff = arr(gd["pen_N_eff"])
        a2 = axPhi.twinx()
        a2.plot(t[:len(neff)], neff, color="#D55E00", lw=1.6, ls="--",
                label=r"$N_{\mathrm{eff}}(t)$")
        a2.set_ylabel(r"$N_{\mathrm{eff}}$", color="#D55E00")
        a2.tick_params(axis="y", labelcolor="#D55E00")
        a2.grid(False)
        # combined legend
        lines1, labels1 = axPhi.get_legend_handles_labels()
        lines2, labels2 = a2.get_legend_handles_labels()
        axPhi.legend(lines1 + lines2, labels1 + labels2,
                     loc="upper left", ncol=2, handlelength=1.6)
    axPhi.text(0.005, 0.93, "(a)", transform=axPhi.transAxes, fontweight='bold', fontsize=9)

    # (b) role probabilities for all attackers (decoy solid, pen dashed, stealth dotted)
    rd_all = gd.get("decoy_role_decoy", [])
    rp_all = gd.get("decoy_role_pen", [])
    rs_all = gd.get("decoy_role_stealth", [])
    for i in range(4):
        if i < len(rd_all):
            rd = arr([rd_all[i][s] for s in range(len(t))])
            lw = 1.8 if i == hitter else 1.0
            axRole.plot(t, rd, color=OFF_COL[i], lw=lw,
                        label=rf"$\pi^{{D}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        if i < len(rp_all):
            rp = arr([rp_all[i][s] for s in range(len(t))])
            axRole.plot(t, rp, color=OFF_COL[i], lw=0.9, ls="--", alpha=0.75)
    axRole.set_ylabel(r"$\pi^{r}_{i}(t)$")
    axRole.set_ylim(-0.03, 1.05)
    axRole.legend(loc="upper right", ncol=4, handlelength=1.6, borderaxespad=0.3)
    axRole.text(0.005, 0.93, "(b)", transform=axRole.transAxes, fontweight='bold', fontsize=9)
    # annotate linestyle key
    axRole.annotate(r"solid: $\pi^{D}$   dashed: $\pi^{P}$",
                    xy=(0.12, 1.02), xycoords="axes fraction",
                    fontsize=7.5, ha="left", va="bottom")

    # (c) P_pen
    pp_all = gd.get("pen_P_pen", [])
    for i in range(4):
        if i < len(pp_all):
            pp = arr([pp_all[i][s] for s in range(len(t))])
            axPen.plot(t, pp, color=OFF_COL[i], lw=1.6 if i == hitter else 1.0,
                       label=rf"$P^{{\mathrm{{pen}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axPen.set_ylabel(r"$P^{\mathrm{pen}}_{i}(t)$")
    axPen.set_ylim(-0.03, 1.05)
    axPen.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axPen.text(0.005, 0.93, "(c)", transform=axPen.transAxes, fontweight='bold', fontsize=9)

    # (d) lock pressure λ_i
    lp_all = gd.get("decoy_lock_pressure", [])
    for i in range(4):
        if i < len(lp_all):
            lp = arr([lp_all[i][s] for s in range(len(t))])
            axLock.plot(t, lp, color=OFF_COL[i], lw=1.6 if i == hitter else 1.0,
                        label=rf"$\lambda_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axLock.set_ylabel(r"$\lambda_{i}(t)$")
    axLock.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axLock.text(0.005, 0.93, "(d)", transform=axLock.transAxes, fontweight='bold', fontsize=9)

    # (e) Gamma_mean / Xi_mean
    if "esc_Gamma_mean" in gd:
        gm = arr(gd["esc_Gamma_mean"])
        axGX.plot(t[:len(gm)], gm, color="#009E73", lw=1.6,
                  label=r"$\bar{\Gamma}(t)\equiv\langle\Gamma_{ij}\rangle$")
    if "esc_Xi_mean" in gd:
        xi = arr(gd["esc_Xi_mean"])
        axGX.plot(t[:len(xi)], xi, color="#CC79A7", lw=1.4, ls="--",
                  label=r"$\bar{\Xi}(t)\equiv\langle\Xi_{ij}\rangle$")
    axGX.axhline(0, color="0.4", ls=":", lw=0.7)
    axGX.set_ylabel(r"$\bar{\Gamma},\ \bar{\Xi}$")
    axGX.legend(loc="upper right", ncol=2, handlelength=1.8, borderaxespad=0.3)
    axGX.text(0.005, 0.93, "(e)", transform=axGX.transAxes, fontweight='bold', fontsize=9)

    # (f) P_hit + E_esc of hitter (highlight), others as thin curves
    ph_all = gd.get("hvt_P_hit", [])
    ee_all = gd.get("esc_E_esc", [])
    for i in range(4):
        if i < len(ph_all):
            ph = arr([ph_all[i][s] for s in range(len(t))])
            axPH.plot(t, ph, color=OFF_COL[i], lw=1.6 if i == hitter else 0.8, alpha=0.9 if i == hitter else 0.55,
                      label=rf"$P^{{\mathrm{{hit}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        if i < len(ee_all):
            ee = arr([ee_all[i][s] for s in range(len(t))])
            axPH.plot(t, ee, color=OFF_COL[i], lw=1.1 if i == hitter else 0.6, ls=":",
                      alpha=0.8 if i == hitter else 0.5)
    axPH.set_ylabel(r"$P^{\mathrm{hit}}_{i},\ E^{\mathrm{esc}}_{i}$")
    axPH.set_xlabel(r"Time $t$ (s)")
    axPH.legend(loc="upper left", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axPH.annotate(r"solid: $P^{\mathrm{hit}}$   dotted: $E^{\mathrm{esc}}$",
                  xy=(0.60, 1.02), xycoords="axes fraction",
                  fontsize=7.5, ha="left", va="bottom")
    axPH.text(0.005, 0.93, "(f)", transform=axPH.transAxes, fontweight='bold', fontsize=9)

    for ax in (axPhi, axRole, axPen, axLock, axGX, axPH):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axPhi, axRole, axPen, axLock, axGX, axPH], d, t, ht, death_steps)

    fig.savefig(os.path.join(base, "fig3_game.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig3_game.pdf")


# ═══════════════ helper to draw 3-D engagement if missing ═══════════════
def fig4_traj3d_if_missing(base, d, sm):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    out = os.path.join(base, "fig4_traj3d.pdf")
    if os.path.exists(out):
        return
    hitter = int(sm["hitter"]); t = d["time"]
    fig = plt.figure(figsize=(6.5, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(4):
        xs = arr([d["off_x"][i][s] for s in range(len(t))])
        ys = arr([d["off_y"][i][s] for s in range(len(t))])
        zs = arr([d["off_z"][i][s] for s in range(len(t))])
        alive = as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1
        hit = as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1
        mask = alive | hit
        if not mask.any():
            continue
        ax.plot(xs[mask], ys[mask], zs[mask], color=OFF_COL[i],
                lw=2.0 if i == hitter else 1.0,
                label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    for j in range(4):
        xs = arr([d["def_x"][j][s] for s in range(len(t))])
        ys = arr([d["def_y"][j][s] for s in range(len(t))])
        zs = arr([d["def_z"][j][s] for s in range(len(t))])
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        ax.plot(xs[alive], ys[alive], zs[alive], color=DEF_COL[j], lw=1.0, ls="--",
                label=rf"$D_{{{j}}}$")
    ax.scatter([float(d["hvt_x"])], [float(d["hvt_y"])], [float(d["hvt_z"])],
               marker="*", color="#ffbf00", edgecolor="k", s=140, label="HVT")
    ax.set_xlabel(r"$x$ (m)"); ax.set_ylabel(r"$y$ (m)"); ax.set_zlabel(r"$z$ (m)")
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_traj3d.pdf (reconstructed)")


# ─────────────────────────── main driver ───────────────────────────
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"WIN={WIN} steps ({WIN*0.01:.2f}s), root={root}")
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if not os.path.isdir(base) or not os.path.exists(os.path.join(base, "trajectory_data.npz")):
            continue
        print(f"\n>>> {case}")
        d = np.load(os.path.join(base, "trajectory_data.npz"), allow_pickle=True)
        with open(os.path.join(base, "summary.json")) as fh:
            sm = json.load(fh)
        gd = {}
        gp = os.path.join(base, "game_data.npz")
        if os.path.exists(gp):
            gd = dict(np.load(gp, allow_pickle=True))
        fig1_offensive(base, d, sm)
        fig2_defensive(base, d, sm, gd)
        fig3_game(base, d, sm, gd)
        fig4_traj3d_if_missing(base, d, sm)
    print("\nAll done.")


if __name__ == "__main__":
    main()
