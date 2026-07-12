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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from PIL import Image

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

SMOOTH_WIN = 51  # Savitzky-Golay window: 0.51 s at 100 Hz
OFF_COL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]   # A0..A3
DEF_COL = ["#17becf", "#ff7f0e", "#2a363b", "#e3b505"]   # D0..D3
ROLE_COL = {"D": "#d62728", "P": "#2ca02c", "S": "#7f7f7f"}
G = 9.80665
FOV_HALF_ANGLE_RAD = np.deg2rad(30.0)
DETECTION_RANGE_M = 2000.0
BREACH_DISTANCE_THRESHOLD_M = 500.0
BREACH_WINDOW_BEFORE_S = 4.0
BREACH_WINDOW_AFTER_S = 4.0


# ─────────────────────────── helpers ────────────────────────────
def smooth(a, w=SMOOTH_WIN, poly=3):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 3:
        return a
    w = min(w, len(a) if len(a) % 2 == 1 else len(a) - 1)
    if w <= poly + 2:
        return a
    if w % 2 == 0:
        w -= 1
    return savgol_filter(a, window_length=w, polyorder=poly, mode="interp")


def smooth_masked(a, mask, w=SMOOTH_WIN, poly=3):
    """Smooth only contiguous visible segments to avoid dead-state edge bleed."""
    a = np.asarray(a, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = np.full_like(a, np.nan, dtype=np.float64)
    start = None
    for k, ok in enumerate(np.r_[mask, False]):
        if ok and start is None:
            start = k
        elif (not ok) and start is not None:
            out[start:k] = smooth(a[start:k], w=w, poly=poly)
            start = None
    return out


def arr(v):
    return np.asarray(v, dtype=np.float64)


def as_int_array(v):
    return np.asarray(v, dtype=int)


def panel_label(ax, label, y=-0.24):
    ax.text(0.5, y, label, transform=ax.transAxes,
            ha="center", va="top", fontweight="bold", fontsize=9,
            clip_on=False)


def _save_pdf_rasterized(fig, path, dpi=300):
    """Save a figure as a rasterized single-page PDF by first exporting PNG
    and converting it to PDF. This avoids some PDF viewer rendering bugs
    caused by layered vector content.
    """
    png_path = str(path) + ".png"
    fig.savefig(png_path, dpi=dpi)
    try:
        im = Image.open(png_path).convert("RGB")
        im.save(path, "PDF", resolution=dpi)
    finally:
        try:
            os.remove(png_path)
        except Exception:
            pass


def _save_png_and_pdf(fig, stem_path, dpi=300):
    """Save a PNG plus a rasterized PDF that is generated from the same PNG."""
    png_path = str(stem_path) + ".png"
    pdf_path = str(stem_path) + ".pdf"
    fig.savefig(png_path, dpi=dpi)
    im = Image.open(png_path).convert("RGB")
    im.save(pdf_path, "PDF", resolution=dpi)


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


def min_fov_series(d, t, attacker_i, n_def):
    """Return Fig. 5 off-axis angle, nearest-defender range, and FOV hit mask."""
    att_x = arr([d["off_x"][attacker_i][s] for s in range(len(t))])
    att_y = arr([d["off_y"][attacker_i][s] for s in range(len(t))])
    att_z = arr([d["off_z"][attacker_i][s] for s in range(len(t))])
    def_x = np.stack([arr([d["def_x"][j][s] for s in range(len(t))]) for j in range(n_def)])
    def_y = np.stack([arr([d["def_y"][j][s] for s in range(len(t))]) for j in range(n_def)])
    def_z = np.stack([arr([d["def_z"][j][s] for s in range(len(t))]) for j in range(n_def)])
    def_heading = np.stack([arr([d["def_heading"][j][s] for s in range(len(t))]) for j in range(n_def)])
    def_gamma = np.stack([arr([d["def_gamma"][j][s] for s in range(len(t))]) for j in range(n_def)])
    def_alive = np.stack([as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
                          for j in range(n_def)])

    theta_min = np.full(len(t), np.nan, dtype=np.float64)
    rho_min = np.full(len(t), np.nan, dtype=np.float64)
    detected = np.zeros(len(t), dtype=bool)
    for s in range(len(t)):
        a = np.array([att_x[s], att_y[s], att_z[s]], dtype=np.float64)
        best_theta = np.inf
        best_rho = np.inf
        for j in range(n_def):
            if not def_alive[j, s]:
                continue
            dpos = np.array([def_x[j, s], def_y[j, s], def_z[j, s]], dtype=np.float64)
            rel = a - dpos
            rho = float(np.linalg.norm(rel))
            if rho < 1e-9:
                continue
            rel_hat = rel / rho
            cg = np.cos(def_gamma[j, s])
            body = np.array([cg * np.cos(def_heading[j, s]),
                             cg * np.sin(def_heading[j, s]),
                             np.sin(def_gamma[j, s])], dtype=np.float64)
            theta = float(np.arccos(np.clip(np.dot(body, rel_hat), -1.0, 1.0)))
            best_theta = min(best_theta, theta)
            best_rho = min(best_rho, rho)
            if rho <= DETECTION_RANGE_M and theta <= FOV_HALF_ANGLE_RAD:
                detected[s] = True
        if np.isfinite(best_theta):
            theta_min[s] = best_theta
        if np.isfinite(best_rho):
            rho_min[s] = best_rho
    return theta_min, rho_min, detected


def select_breach_attackers(d, sm, max_count=2):
    """Prefer attackers that actually reached the HVT; fall back to closest attackers."""
    t = d["time"]
    n_off = int(sm.get("n_offensive", len(d["off_x"])))
    n_off = max(1, min(n_off, len(d["off_x"])))

    hit_order = []
    for i in range(n_off):
        hit = as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1
        hit_idx = np.where(hit)[0]
        if hit_idx.size:
            hit_order.append((int(hit_idx[0]), i))
    hit_order.sort()
    selected = [i for _, i in hit_order[:max_count]]

    if len(selected) < max_count:
        closeness = []
        for i in range(n_off):
            if i in selected:
                continue
            dist = arr([d["off_d_hvt"][i][s] for s in range(len(t))])
            finite = np.isfinite(dist)
            if finite.any():
                closeness.append((float(np.nanmin(dist[finite])), i))
        closeness.sort()
        selected.extend([i for _, i in closeness[:max_count - len(selected)]])
    return selected[:max_count]


def breach_step_from_fig5(theta_min, rho_min, detected):
    close = np.isfinite(rho_min) & (rho_min <= BREACH_DISTANCE_THRESHOLD_M)
    in_fov = np.isfinite(theta_min) & (theta_min <= FOV_HALF_ANGLE_RAD)
    cross = np.where(in_fov[:-1] & ~in_fov[1:] & (close[:-1] | close[1:]))[0] + 1
    if cross.size:
        return int(cross[-1])
    det_cross = np.where(detected[:-1] & ~detected[1:] & (close[:-1] | close[1:]))[0] + 1
    if det_cross.size:
        return int(det_cross[-1])
    if close.any():
        idx = np.where(close)[0]
        return int(idx[np.nanargmin(rho_min[idx])])
    finite = np.isfinite(rho_min)
    if finite.any():
        idx = np.where(finite)[0]
        return int(idx[np.nanargmin(rho_min[idx])])
    return 0


def breach_window_mask(t, step):
    center = float(t[step])
    mask = (t >= center - BREACH_WINDOW_BEFORE_S) & (t <= center + BREACH_WINDOW_AFTER_S)
    if np.count_nonzero(mask) >= 4:
        return mask
    lo = max(0, step - 40)
    hi = min(len(t), step + 41)
    out = np.zeros(len(t), dtype=bool)
    out[lo:hi] = True
    return out


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

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(4, 1, hspace=0.46, left=0.09, right=0.985, top=0.985, bottom=0.075)
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
    panel_label(axV, "(a)")

    # (b) pitch overload n_p
    for i in range(4):
        alive_or_hit = (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
                       (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)
        if not alive_or_hit.any():
            continue
        np_raw = arr([d["off_an_pitch"][i][s] for s in range(len(t))]) / G
        np_arr = smooth_masked(np_raw, alive_or_hit)
        axNp.plot(t[alive_or_hit], np_arr[alive_or_hit],
                  color=OFF_COL[i], lw=1.8 if i == hitter else 1.0)
    axNp.axhline(1, color="0.55", ls="--", lw=0.6)
    axNp.set_ylabel(r"$n_{p,i}$ (g)")
    panel_label(axNp, "(b)")

    # (c) yaw overload n_y
    for i in range(4):
        alive_or_hit = (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
                       (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)
        if not alive_or_hit.any():
            continue
        ny_raw = arr([d["off_an_yaw"][i][s] for s in range(len(t))]) / G
        ny_arr = smooth_masked(ny_raw, alive_or_hit)
        axNy.plot(t[alive_or_hit], ny_arr[alive_or_hit],
                  color=OFF_COL[i], lw=1.8 if i == hitter else 1.0)
    axNy.axhline(0, color="0.55", ls="-", lw=0.5)
    axNy.set_ylabel(r"$n_{y,i}$ (g)")
    panel_label(axNy, "(c)")

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
    axR.set_xlabel(r"Time $t$ (s)", labelpad=14)
    axR.legend(loc="upper right", ncol=1, handlelength=1.6, borderaxespad=0.3)
    panel_label(axR, "(d)", y=-0.18)

    for ax in (axV, axNp, axNy, axR):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axV, axNp, axNy, axR], d, t, ht, death_steps)

    _save_pdf_rasterized(fig, os.path.join(base, "fig1_offensive.pdf"))
    plt.close(fig)
    print("  fig1_offensive.pdf")


# ═══════════════ FIG 2: defensive assignment / overload ═══════════════
def fig2_defensive(base, d, sm, gd):
    t = d["time"]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    dl_raw = d.get("def_ltgt", gd.get("def_ltgt"))
    n_def = int(sm.get("n_defensive", len(dl_raw)))
    n_def = max(1, min(n_def, len(dl_raw)))
    death_steps = {i: death_step(d, i) for i in range(n_def)}

    fig = plt.figure(figsize=(7.8, 10.2))
    gs = GridSpec(5, 1, hspace=0.48, left=0.09, right=0.985, top=0.985, bottom=0.07,
                  height_ratios=[1.0, 0.88, 0.88, 0.86, 0.86])
    axG, axC, axR, axNpD, axNyD = [fig.add_subplot(gs[k]) for k in range(5)]

    # ─── (a) Gantt ───
    dl = np.stack([as_int_array(dl_raw[j]) for j in range(n_def)])
    target_colors = {i: OFF_COL[i % len(OFF_COL)] for i in range(max(4, n_def))}
    target_colors[-1] = "#dcdcdc"
    # set x-limits first so that tight-bbox computation for labels is bounded
    axG.set_xlim(t[0], t[-1])
    axG.set_ylim(0, 1)
    for j in range(n_def):
        tgts = as_int_array([dl[j][s] for s in range(len(t))])
        N = len(tgts)
        seg_start = 0
        for k in range(1, N + 1):
            if k == N or tgts[k] != tgts[seg_start]:
                tgt = int(tgts[seg_start])
                c = target_colors.get(tgt, "#dcdcdc")
                axG.axvspan(t[seg_start], t[k-1],
                            ymin=j/n_def + 0.025/n_def, ymax=(j+1)/n_def - 0.025/n_def,
                            alpha=0.85, color=c, lw=0)
                # in-span label: use axes-fraction for both x and y to avoid tight-bbox blow-up
                mid_t = 0.5 * (t[seg_start] + t[k-1])
                if (t[k-1] - t[seg_start]) > 1.5:
                    # convert mid_t from data-units to axes-fraction
                    x_ax = (mid_t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else 0.5
                    lbl = rf"$A_{{{tgt}}}$" if tgt >= 0 else r"$\varnothing$"
                    axG.text(x_ax, (j + 0.5)/n_def, lbl,
                             ha="center", va="center",
                             fontsize=8, fontweight="bold",
                             color="white" if tgt >= 0 else "#444",
                             transform=axG.transAxes)
                seg_start = k
        axG.axhline(j/n_def, color="white", lw=1.2)
    axG.axhline(1, color="white", lw=1.2)
    axG.set_yticks([(j + 0.5)/n_def for j in range(n_def)])
    axG.set_yticklabels([rf"$D_{{{k}}}$" for k in range(n_def)])
    axG.set_ylabel("Defender")
    axG.grid(False)
    panel_label(axG, "(a)")

    # ─── (b) assignment cost for hitter ───
    cm = np.asarray(d["assign_cost"], dtype=np.float64)
    for j in range(n_def):
        c2h = arr([cm[s][j, hitter] for s in range(len(cm))])
        axC.plot(t[:len(c2h)], c2h, color=DEF_COL[j % len(DEF_COL)], lw=1.2,
                 label=rf"$c_{{{j}\to{hitter}}}$")
    axC.set_ylabel(r"$c_{j\to A^{\star}}$ (m)")
    axC.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    panel_label(axC, "(b)")

    # ─── (c) R(t) — realized def_ltgt cost / optimized assignment cost.
    # C=assign_cost is the simulator's integrated assignment metric.  The
    # numerator follows the actually recorded def_ltgt targets; duplicate
    # commitments and uncovered alive attackers are penalized because they are
    # precisely the coordination failures induced by the offensive decoys.
    T_ = len(cm)
    R_t = np.full(T_, np.nan)
    def_alive = np.stack([as_int_array([d["def_alive"][j][s] for s in range(T_)]) for j in range(n_def)])
    ltgt = np.stack([as_int_array([dl[j][s] for s in range(T_)]) for j in range(n_def)])
    for s in range(T_):
        m = cm[s]
        active_def = [j for j in range(n_def) if def_alive[j, s]]
        active_att = [
            i for i in range(int(sm.get("n_offensive", len(d["off_alive"]))))
            if int(d["off_alive"][i][s]) == 1 or int(d["off_hit"][i][s]) == 1
        ]
        if not active_def or not active_att:
            continue
        sub = m[np.ix_(active_def, active_att)]
        ri, ci = linear_sum_assignment(sub)
        opt = sub[ri, ci].sum()
        if opt < 1e-6:
            continue

        counts = {}
        for j in active_def:
            tgt = int(ltgt[j, s])
            if tgt >= 0:
                counts[tgt] = counts.get(tgt, 0) + 1

        actual = 0.0
        covered = set()
        for j in active_def:
            tgt = int(ltgt[j, s])
            if tgt < 0:
                continue
            duplicate_penalty = max(0, counts.get(tgt, 0) - 1)
            actual += m[j, tgt] * (1.0 + duplicate_penalty)
            if tgt in active_att:
                covered.add(tgt)
        for tgt in active_att:
            if tgt not in covered:
                actual += min(m[j, tgt] for j in active_def)
        R_t[s] = max(actual / opt, 1.0)
    axR.plot(t[:T_], R_t, color="#c0392b", lw=1.6,
             label=r"$R(t)$")
    axR.axhline(1.0, color="0.3", ls="--", lw=0.9)
    axR.axhline(2.0, color="#e67e22", ls=":", lw=0.9)
    axR.fill_between(t[:T_], 1.0, R_t, where=(R_t > 1.0), alpha=0.18, color="#c0392b")
    axR.set_ylabel(r"$R(t)$")
    axR.legend(loc="upper right", ncol=1, handlelength=1.4, borderaxespad=0.3)
    panel_label(axR, "(c)")
    axR.set_ylim(bottom=0.9)

    # ─── (d,e) defender overload n_p / n_y in g ───
    for j in range(n_def):
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        npp_raw = arr([d["def_an_pitch"][j][s] for s in range(len(t))]) / G
        nyy_raw = arr([d["def_an_yaw"][j][s] for s in range(len(t))]) / G
        npp = smooth_masked(npp_raw, alive)
        nyy = smooth_masked(nyy_raw, alive)
        axNpD.plot(t[alive], npp[alive], color=DEF_COL[j % len(DEF_COL)], lw=1.25,
                 label=rf"$n_{{p,D_{{{j}}}}}$")
        axNyD.plot(t[alive], nyy[alive], color=DEF_COL[j % len(DEF_COL)], lw=1.25,
                   label=rf"$n_{{y,D_{{{j}}}}}$")
    for ax, lbl, panel in ((axNpD, r"$n_{p,D_j}$ (g)", "(d)"),
                           (axNyD, r"$n_{y,D_j}$ (g)", "(e)")):
        ax.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
        ax.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
        ax.set_ylabel(lbl)
        ax.set_ylim(-5.6, 5.6)
        ax.legend(loc="upper right", ncol=max(1, n_def), handlelength=1.4, borderaxespad=0.3)
        panel_label(ax, panel)
    axNyD.set_xlabel(r"Time $t$ (s)", labelpad=14)

    for ax in (axG, axC, axR, axNpD, axNyD):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axG, axC, axR, axNpD, axNyD], d, t, ht, death_steps)

    _save_pdf_rasterized(fig, os.path.join(base, "fig2_defensive.pdf"))
    plt.close(fig)
    print("  fig2_defensive.pdf")


# ═══════════════ FIG 3: game priors ═══════════════
def fig3_game(base, d, sm, gd):
    t = d["time"]
    hitter = int(sm["hitter"])
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d, i) for i in range(4)}

    fig = plt.figure(figsize=(7.8, 11.2))
    gs = GridSpec(6, 1, hspace=0.48, left=0.1, right=0.95, top=0.985, bottom=0.07)
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
    panel_label(axPhi, "(a)")

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
            axRole.plot(t, rp, color=OFF_COL[i], lw=0.9, ls="--", alpha=0.75,
                        label=rf"$\pi^{{P}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        if i < len(rs_all):
            rs = arr([rs_all[i][s] for s in range(len(t))])
            axRole.plot(t, rs, color=OFF_COL[i], lw=0.8, ls=":", alpha=0.72,
                        label=rf"$\pi^{{S}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axRole.set_ylabel(r"$\pi^{r}_{i}(t)$")
    axRole.set_ylim(-0.03, 1.05)
    axRole.legend(loc="upper right", ncol=4, handlelength=1.6, borderaxespad=0.3)
    panel_label(axRole, "(b)")
    # annotate linestyle key
    axRole.annotate(r"solid: $\pi^{D}$   dashed: $\pi^{P}$   dotted: $\pi^{S}$",
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
    panel_label(axPen, "(c)")

    # (d) lock pressure λ_i
    lp_all = gd.get("decoy_lock_pressure", [])
    for i in range(4):
        if i < len(lp_all):
            lp = arr([lp_all[i][s] for s in range(len(t))])
            axLock.plot(t, lp, color=OFF_COL[i], lw=1.6 if i == hitter else 1.0,
                        label=rf"$\lambda_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axLock.set_ylabel(r"$\lambda_{i}(t)$")
    axLock.legend(loc="upper right", ncol=4, handlelength=1.4, borderaxespad=0.3)
    panel_label(axLock, "(d)")

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
    panel_label(axGX, "(e)")

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
                      alpha=0.8 if i == hitter else 0.5,
                      label=rf"$E^{{\mathrm{{esc}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axPH.set_ylabel(r"$P^{\mathrm{hit}}_{i},\ E^{\mathrm{esc}}_{i}$")
    axPH.set_xlabel(r"Time $t$ (s)", labelpad=14)
    axPH.legend(loc="upper left", ncol=4, handlelength=1.4, borderaxespad=0.3)
    axPH.annotate(r"solid: $P^{\mathrm{hit}}$   dotted: $E^{\mathrm{esc}}$",
                  xy=(0.60, 1.02), xycoords="axes fraction",
                  fontsize=7.5, ha="left", va="bottom")
    panel_label(axPH, "(f)", y=-0.18)

    for ax in (axPhi, axRole, axPen, axLock, axGX, axPH):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axPhi, axRole, axPen, axLock, axGX, axPH], d, t, ht, death_steps)

    _save_pdf_rasterized(fig, os.path.join(base, "fig3_game.pdf"))
    plt.close(fig)
    print("  fig3_game.pdf")


# ═══════════════ helper to draw 3-D engagement ═══════════════
class Arrow3D(FancyArrowPatch):
    """A clean projected 3-D arrow for trajectory direction markers."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def _project(self):
        from mpl_toolkits.mplot3d import proj3d
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        self._project()
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        return self._project()


class HandlerDirectionArrow(HandlerPatch):
    """Draw one clean arrow in the legend instead of a line+marker composite."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        arrow = FancyArrowPatch((xdescent, y), (xdescent + width, y),
                                arrowstyle="-|>",
                                mutation_scale=0.82 * fontsize,
                                lw=0.9,
                                color=orig_handle.get_edgecolor(),
                                transform=trans)
        return [arrow]


def arrow_segment_by_fraction(xs, ys, zs, frac=0.6, span=0.055):
    xyz = np.column_stack([xs, ys, zs]).astype(float)
    if len(xyz) < 4:
        return None
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] <= 1e-9:
        return None
    s0 = np.clip((frac - span / 2) * s[-1], 0.0, s[-1])
    s1 = np.clip((frac + span / 2) * s[-1], 0.0, s[-1])
    if s1 <= s0:
        return None
    p0 = np.array([np.interp(s0, s, xyz[:, k]) for k in range(3)])
    p1 = np.array([np.interp(s1, s, xyz[:, k]) for k in range(3)])
    if np.linalg.norm(p1 - p0) <= 1e-9:
        return None
    return p0, p1


def add_flow_arrows_3d(ax, xs, ys, zs, color, fractions=(0.62,), lw=0.9,
                       mutation_scale=10, alpha=0.92):
    for frac in fractions:
        seg = arrow_segment_by_fraction(xs, ys, zs, frac=frac)
        if seg is None:
            continue
        p0, p1 = seg
        arrow = Arrow3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        mutation_scale=mutation_scale, lw=lw,
                        arrowstyle="-|>", color=color, alpha=alpha)
        arrow.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.75),
                                pe.Normal()])
        ax.add_artist(arrow)


def set_axes_equal_3d(ax):
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xr = abs(xlim[1] - xlim[0]); yr = abs(ylim[1] - ylim[0]); zr = abs(zlim[1] - zlim[0])
    r = 0.5 * max(xr, yr, zr)
    xm = np.mean(xlim); ym = np.mean(ylim); zm = np.mean(zlim)
    ax.set_xlim3d([xm - r, xm + r])
    ax.set_ylim3d([ym - r, ym + r])
    ax.set_zlim3d([zm - r, zm + r])


def set_tight_3d_limits(ax, xs, ys, zs, pad=0.055):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    spans = []
    for vals in (xs, ys, zs):
        lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))
        span = max(hi - lo, 1.0)
        spans.append((lo, hi, span))
    (x0, x1, xr), (y0, y1, yr), (z0, z1, zr) = spans
    ax.set_xlim(x0 - pad * xr, x1 + pad * xr)
    ax.set_ylim(y0 - pad * yr, y1 + pad * yr)
    ax.set_zlim(z0 - pad * zr, z1 + pad * zr)
    try:
        ax.set_box_aspect((xr, yr, max(zr, 0.38 * max(xr, yr))))
    except Exception:
        pass


def style_3d_axes(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.9))
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.45


def fig4_traj3d(base, d, sm):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    out = os.path.join(base, "fig4_traj3d.pdf")
    hitter = int(sm["hitter"]); t = d["time"]
    fig = plt.figure(figsize=(7.15, 6.15))
    ax = fig.add_subplot(111, projection="3d")
    style_3d_axes(ax)
    all_x, all_y, all_z = [], [], []
    for i in range(4):
        xs = arr([d["off_x"][i][s] for s in range(len(t))])
        ys = arr([d["off_y"][i][s] for s in range(len(t))])
        zs = arr([d["off_z"][i][s] for s in range(len(t))])
        alive = as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1
        hit = as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1
        mask = alive | hit
        if not mask.any():
            continue
        xsm, ysm, zsm = xs[mask], ys[mask], zs[mask]
        all_x.extend(xsm); all_y.extend(ysm); all_z.extend(zsm)
        lw = 2.15 if i == hitter else 1.18
        alpha = 0.98 if i == hitter else 0.78
        line, = ax.plot(xsm, ysm, zsm, color=OFF_COL[i], lw=lw, alpha=alpha,
                        label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        line.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.75),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=OFF_COL[i], marker="o", linewidths=1.1,
                   s=42 if i == hitter else 30, depthshade=False, zorder=7)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=OFF_COL[i],
                   edgecolors="white", marker="D", linewidths=0.8,
                   s=50 if i == hitter else 36, depthshade=False, zorder=8)
        frac = 0.62 if i == hitter else (0.43 + 0.07 * i)
        add_flow_arrows_3d(ax, xsm, ysm, zsm, OFF_COL[i],
                           fractions=(frac,),
                           lw=0.90 if i == hitter else 0.58,
                           mutation_scale=9.5 if i == hitter else 7.0,
                           alpha=0.94 if i == hitter else 0.68)
    for j in range(4):
        xs = arr([d["def_x"][j][s] for s in range(len(t))])
        ys = arr([d["def_y"][j][s] for s in range(len(t))])
        zs = arr([d["def_z"][j][s] for s in range(len(t))])
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        xsm, ysm, zsm = xs[alive], ys[alive], zs[alive]
        all_x.extend(xsm); all_y.extend(ysm); all_z.extend(zsm)
        line, = ax.plot(xsm, ysm, zsm, color=DEF_COL[j], lw=0.95, ls=(0, (4, 2)),
                        alpha=0.70, label=rf"$D_{{{j}}}$")
        line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="white", alpha=0.55),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=DEF_COL[j], marker="o", linewidths=0.9,
                   s=24, depthshade=False, zorder=6)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=DEF_COL[j],
                   edgecolors="white", marker="D", linewidths=0.7,
                   s=30, depthshade=False, zorder=7)
        add_flow_arrows_3d(ax, xsm, ysm, zsm, DEF_COL[j],
                           fractions=(0.46 + 0.055 * j,),
                           lw=0.52, mutation_scale=6.8, alpha=0.58)
    all_x.append(float(d["hvt_x"])); all_y.append(float(d["hvt_y"])); all_z.append(float(d["hvt_z"]))
    ax.scatter([float(d["hvt_x"])], [float(d["hvt_y"])], [float(d["hvt_z"])],
               marker="*", color="#ffbf00", edgecolor="k", linewidths=0.8,
               s=170, depthshade=False, label="HVT")
    ax.text(float(d["hvt_x"]), float(d["hvt_y"]), float(d["hvt_z"]) + 80,
            "HVT", fontsize=8.5, ha="center", va="bottom")
    ax.set_xlabel(r"$x$ (m)", labelpad=7)
    ax.set_ylabel(r"$y$ (m)", labelpad=7)
    ax.set_zlabel(r"$z$ (m)", labelpad=7)
    ax.view_init(elev=23, azim=-54)
    set_tight_3d_limits(ax, all_x, all_y, all_z)
    handles, labels = ax.get_legend_handles_labels()
    direction_handle = FancyArrowPatch((0, 0), (1, 0),
                                       arrowstyle="-|>", color="0.25", lw=0.9)
    handles += [
        Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white",
               markeredgecolor="0.25", lw=0, markersize=5.0, label="start"),
        Line2D([0], [0], marker="D", color="0.25", markerfacecolor="0.25",
               markeredgecolor="white", lw=0, markersize=5.0, label="terminal"),
        direction_handle,
    ]
    labels += ["start", "terminal", "direction"]
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.91),
              bbox_transform=ax.transAxes, ncol=3, fontsize=7.2,
              handlelength=1.6, columnspacing=0.9, borderaxespad=0.25,
              framealpha=0.92, handler_map={FancyArrowPatch: HandlerDirectionArrow()})
    _save_pdf_rasterized(fig, out)
    plt.close(fig)
    print("  fig4_traj3d.pdf")


def fig5_penetration_angle(base, d, sm):
    t = d["time"]
    hitter = int(sm["hitter"])
    n_off = int(sm.get("n_offensive", len(d["off_x"])))
    n_def = int(sm.get("n_defensive", len(d["def_x"])))
    n_off = max(1, min(n_off, len(d["off_x"])))
    n_def = max(1, min(n_def, len(d["def_x"])))
    ht = sm.get("hit_time_s")
    death_steps = {i: death_step(d, i) for i in range(n_off)}

    fig, (axA, axR) = plt.subplots(2, 1, figsize=(7.6, 6.8), sharex=True,
                                   gridspec_kw={"hspace": 0.36, "height_ratios": [1.0, 0.92]})

    for i in range(n_off):
        theta_min, rho_min, detected = min_fov_series(d, t, i, n_def)
        color = OFF_COL[i % len(OFF_COL)]
        lw = 2.0 if i == hitter else 1.1
        alpha = 0.98 if i == hitter else 0.72
        deg = np.degrees(theta_min)
        axA.plot(t, deg, color=color, lw=lw, alpha=alpha,
                 label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        axA.fill_between(t, 0, deg, where=detected & np.isfinite(deg),
                         color=color, alpha=0.08)
        axR.plot(t, rho_min, color=color, lw=lw, alpha=alpha,
                 label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))

    axA.axhline(np.degrees(FOV_HALF_ANGLE_RAD), color="#b22222", ls="--", lw=1.0,
                label=rf"$\alpha_{{\mathrm{{FOV}}}}={np.degrees(FOV_HALF_ANGLE_RAD):.0f}^\circ$")
    axA.set_ylabel(r"Off-axis angle $\alpha$ (deg)")
    axA.set_ylim(bottom=0)
    axA.legend(loc="upper right", ncol=2, handlelength=1.6, borderaxespad=0.3)
    panel_label(axA, "(a)")

    axR.axhline(DETECTION_RANGE_M, color="#b22222", ls="--", lw=1.0,
                label=rf"$R_{{\mathrm{{det}}}}={DETECTION_RANGE_M:.0f}$m")
    axR.set_ylabel(r"Nearest defender distance $\rho_{\min}$ (m)")
    axR.set_xlabel(r"Time $t$ (s)", labelpad=14)
    axR.legend(loc="upper right", ncol=2, handlelength=1.6, borderaxespad=0.3)
    panel_label(axR, "(b)")

    for ax in (axA, axR):
        ax.set_xlim(t[0], t[-1])

    draw_death_and_hit([axA, axR], d, t, ht, death_steps)

    _save_pdf_rasterized(fig, os.path.join(base, "fig5_penetration_angle.pdf"))
    plt.close(fig)
    print("  fig5_penetration_angle.pdf")


def _plot_breach_bottom_panel(ax, info, color, panel):
    t = info["t"]
    win = info["window"]
    step = info["step"]
    theta_deg = np.degrees(info["theta"])
    rho = info["rho"]
    tt = t[win]
    theta_win = theta_deg[win]
    rho_win = rho[win]

    axR = ax.twinx()
    line_alpha, = ax.plot(tt, theta_win, color=color, lw=1.45,
                          label=r"$\alpha$")
    line_rho, = axR.plot(tt, rho_win, color="#2a363b", lw=1.35,
                         label=r"$\rho_{\min}$")
    ax.axhline(np.degrees(FOV_HALF_ANGLE_RAD), color="#b22222", ls="--",
               lw=0.85, alpha=0.85)
    axR.axhline(BREACH_DISTANCE_THRESHOLD_M, color="#b22222", ls=":",
                lw=0.95, alpha=0.9)
    ax.axvline(t[step], color="0.25", ls="-", lw=0.85, alpha=0.75)

    ax.set_xlim(float(tt[0]), float(tt[-1]))
    ax.set_ylim(0, 180)
    finite_rho = rho_win[np.isfinite(rho_win)]
    rho_top = max(BREACH_DISTANCE_THRESHOLD_M * 1.15,
                  float(np.nanmax(finite_rho)) * 1.08 if finite_rho.size else BREACH_DISTANCE_THRESHOLD_M)
    axR.set_ylim(0, rho_top)

    ax.set_title(rf"$A_{{{info['attacker']}}}$ breach segment, $t_B={t[step]:.1f}$ s",
                 fontsize=8.4, pad=3.0)
    ax.set_xlabel(r"Time $t$ (s)", labelpad=8)
    ax.set_ylabel(r"$\alpha$ (deg)", color=color)
    axR.set_ylabel(r"$\rho_{\min}$ (m)", color="#2a363b")
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    axR.locator_params(axis="y", nbins=4)
    ax.tick_params(axis="both", labelsize=8.0, length=2.5, pad=2.0)
    axR.tick_params(axis="y", labelsize=8.0, length=2.5, pad=2.0, colors="#2a363b")
    ax.tick_params(axis="y", colors=color)
    ax.grid(True, alpha=0.30, lw=0.35)
    axR.grid(False)
    ax.text(0.02, 0.93, rf"$\alpha_{{\mathrm{{FOV}}}}={np.degrees(FOV_HALF_ANGLE_RAD):.0f}^\circ$",
            transform=ax.transAxes, color="#b22222", fontsize=7.2,
            ha="left", va="top")
    axR.text(0.98, 0.84, rf"$\rho={BREACH_DISTANCE_THRESHOLD_M:.0f}$ m",
             transform=axR.transAxes, color="#b22222", fontsize=7.2,
             ha="right", va="top")
    ax.legend([line_alpha, line_rho], [r"$\alpha$", r"$\rho_{\min}$"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, handlelength=1.4, fontsize=7.6, framealpha=0.88)
    panel_label(ax, panel, y=-0.42)
    return ax, axR


def fig6_traj3d_breach_insets(base, d, sm):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    t = d["time"]
    hitter = int(sm["hitter"])
    n_off = int(sm.get("n_offensive", len(d["off_x"])))
    n_def = int(sm.get("n_defensive", len(d["def_x"])))
    n_off = max(1, min(n_off, len(d["off_x"])))
    n_def = max(1, min(n_def, len(d["def_x"])))

    breach_infos = {}
    for i in select_breach_attackers(d, sm, max_count=2):
        theta_min, rho_min, detected = min_fov_series(d, t, i, n_def)
        step = breach_step_from_fig5(theta_min, rho_min, detected)
        breach_infos[i] = {
            "attacker": i,
            "theta": theta_min,
            "rho": rho_min,
            "detected": detected,
            "step": step,
            "window": breach_window_mask(t, step),
            "t": t,
            "xyz": (float(d["off_x"][i][step]),
                    float(d["off_y"][i][step]),
                    float(d["off_z"][i][step])),
        }

    fig = plt.figure(figsize=(7.6, 8.7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.1, 1.0],
                  hspace=0.34, wspace=0.42,
                  left=0.08, right=0.92, top=0.985, bottom=0.095)
    ax = fig.add_subplot(gs[0, :], projection="3d")
    style_3d_axes(ax)
    all_x, all_y, all_z = [], [], []

    for i in range(n_off):
        xs = arr([d["off_x"][i][s] for s in range(len(t))])
        ys = arr([d["off_y"][i][s] for s in range(len(t))])
        zs = arr([d["off_z"][i][s] for s in range(len(t))])
        alive = as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1
        hit = as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1
        mask = alive | hit
        if not mask.any():
            continue
        xsm, ysm, zsm = xs[mask], ys[mask], zs[mask]
        all_x.extend(xsm); all_y.extend(ysm); all_z.extend(zsm)
        lw = 2.15 if i == hitter else 1.18
        alpha = 0.98 if i == hitter else 0.78
        line, = ax.plot(xsm, ysm, zsm, color=OFF_COL[i % len(OFF_COL)], lw=lw, alpha=alpha,
                        label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        line.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.75),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=OFF_COL[i % len(OFF_COL)], marker="o", linewidths=1.1,
                   s=42 if i == hitter else 30, depthshade=False, zorder=7)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=OFF_COL[i % len(OFF_COL)],
                   edgecolors="white", marker="D", linewidths=0.8,
                   s=50 if i == hitter else 36, depthshade=False, zorder=8)
        if i in breach_infos:
            bx, by, bz = breach_infos[i]["xyz"]
            ax.scatter([bx], [by], [bz], facecolors="white",
                       edgecolors=OFF_COL[i % len(OFF_COL)], marker="s",
                       linewidths=1.35, s=54, depthshade=False, zorder=10)
            ax.text(bx, by, bz + 65, rf"$B_{{{i}}}$", fontsize=7.2,
                    color=OFF_COL[i % len(OFF_COL)], ha="center", va="bottom")
        frac = 0.62 if i == hitter else (0.43 + 0.07 * i)
        add_flow_arrows_3d(ax, xsm, ysm, zsm, OFF_COL[i % len(OFF_COL)],
                           fractions=(frac,),
                           lw=0.90 if i == hitter else 0.58,
                           mutation_scale=9.5 if i == hitter else 7.0,
                           alpha=0.94 if i == hitter else 0.68)

    for j in range(n_def):
        xs = arr([d["def_x"][j][s] for s in range(len(t))])
        ys = arr([d["def_y"][j][s] for s in range(len(t))])
        zs = arr([d["def_z"][j][s] for s in range(len(t))])
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        xsm, ysm, zsm = xs[alive], ys[alive], zs[alive]
        all_x.extend(xsm); all_y.extend(ysm); all_z.extend(zsm)
        line, = ax.plot(xsm, ysm, zsm, color=DEF_COL[j % len(DEF_COL)], lw=0.95,
                        ls=(0, (4, 2)), alpha=0.70, label=rf"$D_{{{j}}}$")
        line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="white", alpha=0.55),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=DEF_COL[j % len(DEF_COL)], marker="o", linewidths=0.9,
                   s=24, depthshade=False, zorder=6)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=DEF_COL[j % len(DEF_COL)],
                   edgecolors="white", marker="D", linewidths=0.7,
                   s=30, depthshade=False, zorder=7)
        add_flow_arrows_3d(ax, xsm, ysm, zsm, DEF_COL[j % len(DEF_COL)],
                           fractions=(0.46 + 0.055 * j,),
                           lw=0.52, mutation_scale=6.8, alpha=0.58)

    all_x.append(float(d["hvt_x"])); all_y.append(float(d["hvt_y"])); all_z.append(float(d["hvt_z"]))
    ax.scatter([float(d["hvt_x"])], [float(d["hvt_y"])], [float(d["hvt_z"])],
               marker="*", color="#ffbf00", edgecolor="k", linewidths=0.8,
               s=170, depthshade=False, label="HVT")
    ax.text(float(d["hvt_x"]), float(d["hvt_y"]), float(d["hvt_z"]) + 80,
            "HVT", fontsize=8.5, ha="center", va="bottom")

    ax.set_xlabel(r"$x$ (m)", labelpad=7)
    ax.set_ylabel(r"$y$ (m)", labelpad=7)
    ax.set_zlabel(r"$z$ (m)", labelpad=7)
    ax.view_init(elev=23, azim=-54)
    set_tight_3d_limits(ax, all_x, all_y, all_z)
    ax.text2D(0.5, -0.075, "(a)", transform=ax.transAxes,
              ha="center", va="top", fontweight="bold", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    direction_handle = FancyArrowPatch((0, 0), (1, 0),
                                       arrowstyle="-|>", color="0.25", lw=0.9)
    handles += [
        Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white",
               markeredgecolor="0.25", lw=0, markersize=5.0, label="start"),
        Line2D([0], [0], marker="D", color="0.25", markerfacecolor="0.25",
               markeredgecolor="white", lw=0, markersize=5.0, label="terminal"),
        Line2D([0], [0], marker="s", color="0.25", markerfacecolor="white",
               markeredgecolor="0.25", lw=0, markersize=5.0, label="breach"),
        direction_handle,
    ]
    labels += ["start", "terminal", "breach", "direction"]
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.91),
              bbox_transform=ax.transAxes, ncol=3, fontsize=7.0,
              handlelength=1.5, columnspacing=0.8, borderaxespad=0.25,
              framealpha=0.92, handler_map={FancyArrowPatch: HandlerDirectionArrow()})

    for idx, i in enumerate(breach_infos):
        bx = fig.add_subplot(gs[1, idx])
        _plot_breach_bottom_panel(bx, breach_infos[i],
                                  OFF_COL[i % len(OFF_COL)],
                                  f"({chr(ord('b') + idx)})")

    _save_png_and_pdf(fig, os.path.join(base, "fig6_traj3d_breach_insets"))
    plt.close(fig)
    print("  fig6_traj3d_breach_insets.pdf/.png")


# ─────────────────────────── main driver ───────────────────────────
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"SMOOTH_WIN={SMOOTH_WIN} steps ({SMOOTH_WIN*0.01:.2f}s), root={root}")
    for case in sorted(os.listdir(root)):
        base = os.path.join(root, case)
        if not os.path.isdir(base) or not os.path.exists(os.path.join(base, "trajectory_data.npz")):
            continue
        print(f"\n>>> {case}")
        # Materialize the npz once.  The plotting helpers index the same arrays
        # inside tight loops; keeping NpzFile lazy would repeatedly decompress
        # individual members and make the first figure pathologically slow.
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
        fig5_penetration_angle(base, d, sm)
        fig6_traj3d_breach_insets(base, d, sm)
    print("\nAll done.")


if __name__ == "__main__":
    main()
