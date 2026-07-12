#!/usr/bin/env python
"""V71 paper figure generator — IEEE-journal style.

Produces per-case paper panels:
  fig1a_speed.pdf, fig1b_pitch_overload.pdf, fig1c_yaw_overload.pdf, fig1d_distance.pdf
  fig2a_gantt.pdf, fig2b_assignment_cost.pdf, fig2c_ratio.pdf,
  fig2d_pitch_overload.pdf, fig2e_yaw_overload.pdf
  fig3a_role_prob.pdf, fig3b_lock_pressure.pdf,
  fig3c_phi_neff.pdf, fig3d_pen_prob.pdf, fig3e_gamma_xi.pdf, fig3f_hit_escape.pdf
  fig4_traj3d.pdf            3-D engagement trajectory

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

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
OVERLOAD_SMOOTH_WIN = 151  # smoother overload traces for paper figures (1.51 s at 100 Hz)
OFF_COL = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#ff9896", "#aec7e8", "#98df8a",
]
DEF_COL = [
    "#17becf", "#ff7f0e", "#2a363b", "#e3b505",
    "#6a3d9a", "#b15928", "#1b9e77", "#7570b3",
    "#a6761d", "#66a61e", "#e7298a", "#666666",
]
ROLE_COL = {"D": "#d62728", "P": "#2ca02c", "S": "#7f7f7f"}
ROLE_BAND_COL = {"D": "#D55E00", "P": "#009E73", "S": "#0072B2"}
G = 9.80665
TIME_LABEL = r"Time $t$ (s)"


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


def n_off(d):
    return int(np.asarray(d["off_x"]).shape[0])


def n_def(d):
    return int(np.asarray(d["def_x"]).shape[0])


def off_color(i):
    return OFF_COL[i % len(OFF_COL)]


def def_color(j):
    return DEF_COL[j % len(DEF_COL)]


def legend_cols(n, cap=4):
    return max(1, min(cap, int(n)))


def get_hitter(sm):
    if "hitter" in sm:
        return int(sm["hitter"])
    hit_indices = sm.get("hit_indices") or []
    if hit_indices:
        return int(hit_indices[0])
    return int(sm.get("best_agent", 0))


def get_hit_time(sm, d=None):
    if sm.get("hit_time_s") is not None:
        return sm.get("hit_time_s")
    hit_step = sm.get("hit_step", {})
    hitter = get_hitter(sm)
    if isinstance(hit_step, dict):
        step = hit_step.get(str(hitter), hit_step.get(hitter))
    else:
        step = hit_step
    if step is None:
        return None
    if d is not None and "time" in d:
        t = np.asarray(d["time"], dtype=float)
        idx = int(step) - 1
        if 0 <= idx < len(t):
            return float(t[idx])
    return float(step) * 0.01


def panel_label(ax, label, y=-0.24):
    ax.text(0.5, y, label, transform=ax.transAxes,
            ha="center", va="top", fontweight="bold", fontsize=9,
            clip_on=False)


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
            ax.axvline(t[ds], color=off_color(i), ls=":", lw=0.9, alpha=0.6)
    if ht is not None:
        for ax in axes:
            ax.axvline(ht, color="#145214", ls="-", lw=1.2, alpha=0.85)


def fov_style_label_annotation(ax, txt, xy, xytext, color="k"):
    ax.annotate(txt, xy=xy, xytext=xytext,
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.6))


def is_legacy_4v4_case(base, d):
    name = os.path.basename(os.path.normpath(base)).lower()
    return "4v4" in name or name.startswith("caseb") or (n_off(d) == 4 and n_def(d) == 4)


def save_tight(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


def new_time_axis(figsize=(4.25, 2.35)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.235)
    return fig, ax


def finish_time_axis(ax, t, xlabel=TIME_LABEL):
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(xlabel, labelpad=5)


def visible_attacker_mask(d, i, t):
    return (as_int_array([d["off_alive"][i][s] for s in range(len(t))]) == 1) | \
           (as_int_array([d["off_hit"][i][s] for s in range(len(t))]) == 1)


def lock_pressure_visible_mask(d, i, t, hit_time=None):
    """Keep only live, pre-terminal samples for lock-pressure curves."""
    t_arr = np.asarray(t, dtype=np.float64)
    alive = as_int_array([d["off_alive"][i][s] for s in range(len(t_arr))]) == 1
    hit = as_int_array([d["off_hit"][i][s] for s in range(len(t_arr))]) == 1
    mask = alive & (~hit)
    if hit_time is not None:
        mask &= t_arr < float(hit_time)
    return mask


def set_dynamic_y_range(ax, values, pad_frac=0.10, min_span=0.02, clamp_min=0.0):
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    lo = float(np.nanmin(y))
    hi = float(np.nanmax(y))
    span = max(hi - lo, min_span)
    center = 0.5 * (lo + hi)
    if hi - lo < min_span:
        lo = center - 0.5 * span
        hi = center + 0.5 * span
    pad = max(span * pad_frac, min_span * 0.25)
    ymin = lo - pad
    if clamp_min is not None:
        ymin = max(float(clamp_min), ymin)
    ax.set_ylim(ymin, hi + pad)


def terminal_time_mask(t, hit_time=None):
    t_arr = np.asarray(t, dtype=np.float64)
    mask = np.ones(len(t_arr), dtype=bool)
    if hit_time is not None:
        mask &= t_arr < float(hit_time)
    return mask


def game_series(gd, key):
    if key not in gd:
        return None
    return np.asarray(gd[key], dtype=np.float64)


def plot_attacker_game_series(ax, d, t, data, hitter, hit_time, ylabel, label_fmt,
                              linestyle="-", alpha=1.0, lw_main=1.6, lw_other=1.0,
                              label_lines=True):
    if data is None:
        return []
    values = []
    t_arr = np.asarray(t, dtype=np.float64)
    no = n_off(d)
    for i in range(no):
        if i >= data.shape[0]:
            continue
        y = np.asarray(data[i], dtype=np.float64)[:len(t_arr)]
        x = t_arr[:len(y)]
        visible = lock_pressure_visible_mask(d, i, x, hit_time) & np.isfinite(y)
        if not visible.any():
            continue
        values.extend(y[visible])
        label = None
        if label_lines:
            label = label_fmt(i) + (r"$^{\star}$" if i == hitter else "")
        ax.plot(x[visible], y[visible], color=off_color(i),
                lw=lw_main if i == hitter else lw_other, ls=linestyle,
                alpha=alpha if i == hitter else min(alpha, 0.72), label=label)
    ax.set_ylabel(ylabel)
    return values


def clip_limit(a, limit):
    return np.clip(np.asarray(a, dtype=np.float64), -float(limit), float(limit))


def annotate_limit_lines(ax, limit, label):
    ax.text(0.985, limit, label, transform=ax.get_yaxis_transform(),
            color="#b22222", fontsize=7.2, ha="right", va="bottom")
    ax.text(0.985, -limit, "-" + label, transform=ax.get_yaxis_transform(),
            color="#b22222", fontsize=7.2, ha="right", va="top")


def draw_mask_segments(ax, t, mask, y, color="k", lw=1.35, zorder=6):
    """Draw compact horizontal event segments on a heatmap row."""
    start = None
    for k, ok in enumerate(np.r_[mask, False]):
        if ok and start is None:
            start = k
        elif (not ok) and start is not None:
            ax.plot([t[start], t[k - 1]], [y, y], color=color, lw=lw,
                    solid_capstyle="butt", zorder=zorder)
            start = None


def normalize_timewise(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for s in range(x.shape[1]):
        col = x[:, s]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        lo = float(np.nanmin(col[ok]))
        hi = float(np.nanmax(col[ok]))
        if hi - lo < 1e-9:
            out[ok, s] = 0.0
        else:
            out[ok, s] = (col[ok] - lo) / (hi - lo)
    return out


def infer_role_preferences(d, sm, gd):
    """Infer soft D/P/S role preferences from closed-loop simulation evidence."""
    t = np.asarray(d["time"], dtype=np.float64)
    no = n_off(d)
    nd = n_def(d)
    hitter = get_hitter(sm)
    alive = np.stack([
        as_int_array([d["off_alive"][i][s] for s in range(len(t))])
        for i in range(no)
    ]).astype(bool)
    hit = np.stack([
        as_int_array([d["off_hit"][i][s] for s in range(len(t))])
        for i in range(no)
    ]).astype(bool)
    active = alive | hit

    lbc = np.asarray(d.get("off_lbc", np.zeros((no, len(t)))), dtype=np.float64)
    lp = np.asarray(gd.get("decoy_lock_pressure", np.zeros((no, len(t)))), dtype=np.float64)
    if lp.shape != (no, len(t)):
        lp = np.zeros((no, len(t)), dtype=np.float64)

    dl_raw = d.get("def_ltgt", gd.get("def_ltgt", np.full((nd, len(t)), -1)))
    dl = np.stack([as_int_array(dl_raw[j]) for j in range(nd)])
    def_alive = np.stack([
        as_int_array([d["def_alive"][j][s] for s in range(len(t))])
        for j in range(nd)
    ]).astype(bool)
    assign_frac = np.zeros((no, len(t)), dtype=np.float64)
    for s in range(len(t)):
        live_def = max(1, int(def_alive[:, s].sum()))
        for i in range(no):
            assign_frac[i, s] = np.logical_and(def_alive[:, s], dl[:, s] == i).sum() / live_def
    assign_any = np.clip(assign_frac * max(nd, 1), 0.0, 1.0)

    sacrifice = np.zeros((no, len(t)), dtype=np.float64)
    for i in range(no):
        ds = death_step(d, i)
        if ds is None:
            continue
        dt = np.maximum(t[ds] - t, 0.0)
        sacrifice[i, :ds + 1] = np.exp(-dt[:ds + 1] / 12.0)

    rho_h = np.stack([
        arr([d["off_d_hvt"][i][s] for s in range(len(t))])
        for i in range(no)
    ])
    rho0 = np.maximum(rho_h[:, [0]], 1.0)
    target_progress = np.clip((rho0 - rho_h) / np.maximum(rho0 - 5.0, 1.0), 0.0, 1.0)
    target_close = 1.0 - normalize_timewise(rho_h)

    pen = np.asarray(gd.get("pen_P_pen", np.zeros((no, len(t)))), dtype=np.float64)
    if pen.shape != (no, len(t)):
        pen = np.zeros((no, len(t)), dtype=np.float64)
    pen_rel = normalize_timewise(pen)

    lock_rel = normalize_timewise(lp)
    lbc_rel = np.clip(lbc / max(1.0, float(np.nanmax(lbc))), 0.0, 1.0)
    low_threat = np.clip(
        1.0 - 0.60 * lock_rel - 0.90 * lbc_rel - 0.70 * sacrifice - 0.25 * assign_any,
        0.0, 1.0
    )

    hit_bias = np.zeros((no, len(t)), dtype=np.float64)
    if 0 <= hitter < no:
        hit_bias[hitter] = 0.70 + 0.35 * target_progress[hitter]

    decoy_score = 0.35 * assign_any + 1.55 * lbc_rel + 0.95 * lock_rel + 1.30 * sacrifice
    attack_score = (
        1.45 * hit_bias
        + 0.85 * target_progress
        + 0.45 * target_close
        + 0.45 * low_threat
        + 0.25 * pen_rel
    )
    stealth_score = (
        1.15 * low_threat
        + 0.55 * (1.0 - assign_any)
        + 0.35 * (1.0 - target_progress)
        + 0.30 * (1.0 - target_close)
    )
    stealth_score[hitter] *= 0.62

    scores = np.stack([decoy_score, attack_score, stealth_score], axis=0)
    scores[:, ~active] = np.nan
    # Softmax over the three inferred role scores for each aircraft and time.
    temperature = 2.0
    finite_scores = np.where(np.isfinite(scores), scores, -1e9)
    finite_scores -= np.nanmax(finite_scores, axis=0, keepdims=True)
    ex = np.exp(temperature * finite_scores)
    ex[:, ~active] = np.nan
    denom = np.nansum(ex, axis=0, keepdims=True)
    probs = ex / np.maximum(denom, 1e-12)
    probs[:, ~active] = np.nan
    return probs


# ═══════════════ FIG 1: offensive kinematics ═══════════════
def fig1_offensive(base, d, sm):
    """Offensive kinematics as four separate paper panels."""
    t = d["time"]
    no = n_off(d)
    nd = n_def(d)
    hitter = get_hitter(sm)
    ht = get_hit_time(sm, d)

    death_steps = {i: death_step(d, i) for i in range(no)}

    # fig1a: airspeed V_i
    fig, axV = new_time_axis()
    for i in range(no):
        alive_or_hit = visible_attacker_mask(d, i, t)
        if not alive_or_hit.any():
            continue
        vv = arr([d["off_v"][i][s] for s in range(len(t))])
        axV.plot(t[alive_or_hit], vv[alive_or_hit],
                 color=off_color(i), lw=1.8 if i == hitter else 1.0,
                 label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axV.axhline(40, color="0.55", ls="--", lw=0.6)
    axV.axhline(50, color="0.55", ls="--", lw=0.6)
    axV.set_ylim(35, 55)
    axV.set_ylabel(r"$V_i$ (m s$^{-1}$)")
    axV.legend(loc="upper right", ncol=legend_cols(no), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axV, t)
    draw_death_and_hit(axV, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig1a_speed.pdf"))
    print("  fig1a_speed.pdf")

    # fig1b: pitch overload n_p
    fig, axNp = new_time_axis()
    for i in range(no):
        alive_or_hit = visible_attacker_mask(d, i, t)
        if not alive_or_hit.any():
            continue
        np_raw = arr([d["off_an_pitch"][i][s] for s in range(len(t))]) / G
        np_arr = clip_limit(smooth_masked(clip_limit(np_raw, 2.5), alive_or_hit, w=OVERLOAD_SMOOTH_WIN), 2.5)
        axNp.plot(t[alive_or_hit], np_arr[alive_or_hit],
                  color=off_color(i), lw=1.8 if i == hitter else 1.0)
    axNp.axhline(2.5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNp.axhline(-2.5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNp.set_ylabel(r"$n_{p,i}$ (g)")
    axNp.set_ylim(-3.0, 3.0)
    axNp.set_yticks(np.arange(-3.0, 3.1, 1.0))
    annotate_limit_lines(axNp, 2.5, "2.5g")
    finish_time_axis(axNp, t)
    draw_death_and_hit(axNp, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig1b_pitch_overload.pdf"))
    print("  fig1b_pitch_overload.pdf")

    # fig1c: yaw overload n_y
    fig, axNy = new_time_axis()
    for i in range(no):
        alive_or_hit = visible_attacker_mask(d, i, t)
        if not alive_or_hit.any():
            continue
        ny_raw = arr([d["off_an_yaw"][i][s] for s in range(len(t))]) / G
        ny_arr = clip_limit(smooth_masked(clip_limit(ny_raw, 2.5), alive_or_hit, w=OVERLOAD_SMOOTH_WIN), 2.5)
        axNy.plot(t[alive_or_hit], ny_arr[alive_or_hit],
                  color=off_color(i), lw=1.8 if i == hitter else 1.0)
    axNy.axhline(0, color="0.55", ls="-", lw=0.5)
    axNy.axhline(2.5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNy.axhline(-2.5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNy.set_ylabel(r"$n_{y,i}$ (g)")
    axNy.set_ylim(-3.0, 3.0)
    axNy.set_yticks(np.arange(-3.0, 3.1, 1.0))
    annotate_limit_lines(axNy, 2.5, "2.5g")
    finish_time_axis(axNy, t)
    draw_death_and_hit(axNy, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig1c_yaw_overload.pdf"))
    print("  fig1c_yaw_overload.pdf")

    # fig1d: distance bundle — hitter to HVT and to nearest interceptor
    fig, axR = new_time_axis()
    dh = arr([d["off_d_hvt"][hitter][s] for s in range(len(t))])
    dnear = np.full(len(t), np.nan)
    for s in range(len(t)):
        ox = d["off_x"][hitter][s]; oy = d["off_y"][hitter][s]; oz = d["off_z"][hitter][s]
        md = 1e9
        for j in range(nd):
            if as_int_array([d["def_alive"][j][s]])[0] == 1:
                dd = np.sqrt((d["def_x"][j][s]-ox)**2 +
                             (d["def_y"][j][s]-oy)**2 +
                             (d["def_z"][j][s]-oz)**2)
                md = min(md, dd)
        dnear[s] = md if md < 1e9 else np.nan
    axR.semilogy(t, dh, color=off_color(hitter), lw=1.8,
                 label=rf"$\rho_{{{hitter}H}}$  ($A_{{{hitter}}}\to H$)")
    axR.semilogy(t, dnear, color="#ff7f0e", lw=1.25, ls="--",
                 label=rf"$\min_{{j}}\rho_{{{hitter}j}}$  ($A_{{{hitter}}}\to$ nearest $D$)")
    axR.axhline(500, color="0.55", ls=":", lw=0.8)
    axR.axhline(5, color="#b22222", ls=":", lw=0.8)
    tx = t[0] + 0.97 * (t[-1] - t[0])
    axR.text(tx, 6.5, r"$\rho^{\mathrm{kill}}{=}5$m", fontsize=7,
             color="#b22222", ha="right")
    axR.text(tx, 650, r"$500$m", fontsize=7, color="0.4", ha="right")
    axR.set_ylabel(r"Distance (m)")
    axR.legend(loc="upper right", ncol=1, handlelength=1.6, borderaxespad=0.3)
    finish_time_axis(axR, t)
    draw_death_and_hit(axR, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig1d_distance.pdf"))
    print("  fig1d_distance.pdf")


# ═══════════════ FIG 2: defensive assignment / overload ═══════════════
def fig2_defensive(base, d, sm, gd):
    t = d["time"]
    no = n_off(d)
    nd = n_def(d)
    hitter = get_hitter(sm)
    ht = get_hit_time(sm, d)
    death_steps = {i: death_step(d, i) for i in range(no)}

    dl_raw = d.get("def_ltgt", gd.get("def_ltgt"))
    dl = np.stack([as_int_array(dl_raw[j]) for j in range(nd)])

    # fig2a: defender target Gantt
    fig, axG = new_time_axis(figsize=(5.35, 2.25))
    target_colors = {i: off_color(i) for i in range(no)}
    target_colors[-1] = "#dcdcdc"
    # set x-limits first so that tight-bbox computation for labels is bounded
    axG.set_xlim(t[0], t[-1])
    axG.set_ylim(0, 1)
    for j in range(nd):
        tgts = as_int_array([dl[j][s] for s in range(len(t))])
        N = len(tgts)
        seg_start = 0
        for k in range(1, N + 1):
            if k == N or tgts[k] != tgts[seg_start]:
                tgt = int(tgts[seg_start])
                c = target_colors.get(tgt, "#dcdcdc")
                axG.axvspan(t[seg_start], t[k-1],
                            ymin=j/max(nd, 1) + 0.025, ymax=(j+1)/max(nd, 1) - 0.025,
                            alpha=0.85, color=c, lw=0)
                # in-span label: use axes-fraction for both x and y to avoid tight-bbox blow-up
                mid_t = 0.5 * (t[seg_start] + t[k-1])
                if (t[k-1] - t[seg_start]) > 1.5:
                    # convert mid_t from data-units to axes-fraction
                    x_ax = (mid_t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else 0.5
                    lbl = rf"$A_{{{tgt}}}$" if tgt >= 0 else r"$\varnothing$"
                    axG.text(x_ax, (j + 0.5)/max(nd, 1), lbl,
                             ha="center", va="center",
                             fontsize=8, fontweight="bold",
                             color="white" if tgt >= 0 else "#444",
                             transform=axG.transAxes)
                seg_start = k
        axG.axhline(j/max(nd, 1), color="white", lw=1.2)
    axG.axhline(1, color="white", lw=1.2)
    axG.set_yticks([(j + 0.5) / max(nd, 1) for j in range(nd)])
    axG.set_yticklabels([rf"$D_{{{k}}}$" for k in range(nd)])
    axG.set_ylabel("Defender")
    axG.grid(False)
    finish_time_axis(axG, t)
    draw_death_and_hit(axG, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig2a_gantt.pdf"))
    print("  fig2a_gantt.pdf")

    # fig2b: assignment cost for the hitter
    fig, axC = new_time_axis()
    cm = np.asarray(d["assign_cost"], dtype=np.float64)  # (T, n_def, n_off)
    for j in range(nd):
        c2h = arr([cm[s][j, hitter] for s in range(len(cm))])
        axC.plot(t[:len(c2h)], c2h, color=def_color(j), lw=1.2,
                 label=rf"$c_{{{j}\to{hitter}}}$")
    axC.set_ylabel(r"$c_{j\to A^{\star}}$ (m)")
    axC.legend(loc="upper right", ncol=legend_cols(nd), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axC, t)
    draw_death_and_hit(axC, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig2b_assignment_cost.pdf"))
    print("  fig2b_assignment_cost.pdf")

    # fig2c: R(t) — realized def_ltgt cost / optimized assignment cost.
    # C=assign_cost is the simulator's integrated assignment metric.  The
    # numerator follows the actually recorded def_ltgt targets; duplicate
    # commitments and uncovered alive attackers are penalized because they are
    # precisely the coordination failures induced by the offensive decoys.
    fig, axR = new_time_axis()
    T_ = len(cm)
    R_t = np.full(T_, np.nan)
    def_alive = np.stack([as_int_array([d["def_alive"][j][s] for s in range(T_)]) for j in range(nd)])
    ltgt = np.stack([as_int_array([dl[j][s] for s in range(T_)]) for j in range(nd)])
    for s in range(T_):
        m = cm[s]
        active_def = [j for j in range(nd) if def_alive[j, s]]
        active_att = [
            i for i in range(no)
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
            if 0 <= tgt < no:
                counts[tgt] = counts.get(tgt, 0) + 1

        actual = 0.0
        covered = set()
        for j in active_def:
            tgt = int(ltgt[j, s])
            if tgt < 0 or tgt >= no:
                continue
            duplicate_penalty = max(0, counts.get(tgt, 0) - 1)
            actual += m[j, tgt] * (1.0 + duplicate_penalty)
            if tgt in active_att:
                covered.add(tgt)
        for tgt in active_att:
            if tgt not in covered:
                actual += min(m[j, tgt] for j in active_def)
        R_t[s] = max(actual / opt, 1.0)
    axR.plot(t[:T_], R_t, color="#c0392b", lw=1.6)
    axR.axhline(1.0, color="0.3", ls="--", lw=0.9)
    axR.axhline(2.0, color="#e67e22", ls=":", lw=0.9)
    axR.fill_between(t[:T_], 1.0, R_t, where=(R_t > 1.0), alpha=0.18, color="#c0392b")
    axR.set_ylabel(r"$\nu(t)$")
    axR.set_ylim(bottom=0.9)
    finish_time_axis(axR, t)
    draw_death_and_hit(axR, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig2c_ratio.pdf"))
    print("  fig2c_ratio.pdf")

    # fig2d/fig2e: defender overload n_p / n_y in g.  The legacy 4v4 data
    # stored defender accelerations in m/s^2, while 6v6/8v8 already store g.
    def_scale = G if is_legacy_4v4_case(base, d) else 1.0
    fig, axNpD = new_time_axis()
    for j in range(nd):
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        npp_raw = arr([d["def_an_pitch"][j][s] for s in range(len(t))]) / def_scale
        npp = clip_limit(smooth_masked(clip_limit(npp_raw, 5.0), alive, w=OVERLOAD_SMOOTH_WIN), 5.0)
        axNpD.plot(t[alive], npp[alive], color=def_color(j), lw=1.25,
                   label=rf"$n_{{p,D_{{{j}}}}}$")
    axNpD.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNpD.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNpD.set_ylabel(r"$n_{p,D_j}$ (g)")
    axNpD.set_ylim(-5.5, 5.5)
    axNpD.set_yticks(np.arange(-5.0, 5.1, 2.5))
    annotate_limit_lines(axNpD, 5.0, "5g")
    axNpD.legend(loc="upper right", ncol=legend_cols(nd), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axNpD, t)
    draw_death_and_hit(axNpD, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig2d_pitch_overload.pdf"))
    print("  fig2d_pitch_overload.pdf")

    fig, axNyD = new_time_axis()
    for j in range(nd):
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        nyy_raw = arr([d["def_an_yaw"][j][s] for s in range(len(t))]) / def_scale
        nyy = clip_limit(smooth_masked(clip_limit(nyy_raw, 5.0), alive, w=OVERLOAD_SMOOTH_WIN), 5.0)
        axNyD.plot(t[alive], nyy[alive], color=def_color(j), lw=1.25,
                   label=rf"$n_{{y,D_{{{j}}}}}$")
    axNyD.axhline(5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNyD.axhline(-5, color="#b22222", ls="--", lw=0.7, alpha=0.7)
    axNyD.set_ylabel(r"$n_{y,D_j}$ (g)")
    axNyD.set_ylim(-5.5, 5.5)
    axNyD.set_yticks(np.arange(-5.0, 5.1, 2.5))
    annotate_limit_lines(axNyD, 5.0, "5g")
    axNyD.legend(loc="upper right", ncol=legend_cols(nd), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axNyD, t)
    draw_death_and_hit(axNyD, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig2e_yaw_overload.pdf"))
    print("  fig2e_yaw_overload.pdf")


# ═══════════════ FIG 3: game priors ═══════════════
def fig3_game(base, d, sm, gd):
    t = d["time"]
    no = n_off(d)
    hitter = get_hitter(sm)
    ht = get_hit_time(sm, d)
    death_steps = {i: death_step(d, i) for i in range(no)}

    # fig3a: behavior-inferred continuous role preferences. Each heatmap row is
    # one attacker; brighter color means higher inferred probability.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(4.65, 2.95))
    fig.subplots_adjust(left=0.14, right=0.975, top=0.965, bottom=0.17, hspace=0.10)
    inferred = infer_role_preferences(d, sm, gd)
    lbc = np.asarray(d.get("off_lbc", np.zeros((no, len(t)))), dtype=np.float64)
    role_specs = [
        (axes[0], inferred[0], r"$\hat\pi_i^D(t)$"),
        (axes[1], inferred[1], r"$\hat\pi_i^P(t)$"),
        (axes[2], inferred[2], r"$\hat\pi_i^S(t)$"),
    ]
    common_vmax = 1.0
    images = []
    for ax, prob, ylabel in role_specs:
        im = ax.imshow(prob, aspect="auto", origin="lower", interpolation="nearest",
                       extent=[t[0], t[-1], -0.5, no - 0.5],
                       cmap="viridis", vmin=0.0, vmax=common_vmax)
        images.append(im)
        ax.set_ylabel(ylabel)
        ax.set_yticks(range(no))
        ylabels = [rf"$A_{{{i}}}$" + (r"$^\star$" if i == hitter else "") for i in range(no)]
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="y", labelsize=7.4, pad=1)
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.026, pad=0.012)
    cbar.ax.tick_params(labelsize=6.8, length=2, pad=1)
    cbar.set_label("Probability", fontsize=7.2, labelpad=2)
    for i in range(no):
        if i < lbc.shape[0]:
            draw_mask_segments(axes[0], t, lbc[i] > 0, i, color="k", lw=1.15)
    if ht is not None:
        axes[1].scatter([ht], [hitter], marker="*", s=70, color="#145214",
                        edgecolor="white", linewidth=0.45, zorder=8,
                        clip_on=False)
    for ax in axes:
        ax.set_xlim(t[0], t[-1])
        ax.grid(False)
    axes[-1].set_xlabel(TIME_LABEL, labelpad=5)
    draw_death_and_hit(axes, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3a_role_prob.pdf"))
    print("  fig3a_role_prob.pdf")

    # fig3b: lock pressure lambda_i.
    fig, axLock = new_time_axis(figsize=(4.45, 2.45))
    lp_all = gd.get("decoy_lock_pressure", [])
    visible_values = []
    for i in range(no):
        if i < len(lp_all):
            lp = arr([lp_all[i][s] for s in range(len(t))])
            visible = lock_pressure_visible_mask(d, i, t, ht)
            if not visible.any():
                continue
            visible_values.extend(lp[visible])
            axLock.plot(t[visible], lp[visible], color=off_color(i), lw=1.6 if i == hitter else 1.0,
                        label=rf"$\lambda_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
    axLock.set_ylabel(r"$\lambda_{i}(t)$")
    set_dynamic_y_range(axLock, visible_values)
    axLock.legend(loc="upper right", ncol=legend_cols(no), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axLock, t)
    draw_death_and_hit(axLock, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3b_lock_pressure.pdf"))
    print("  fig3b_lock_pressure.pdf")

    # fig3c: decoy potential Phi_decoy and effective interceptor count N_eff.
    valid_t = terminal_time_mask(t, ht)
    fig, axPhi = new_time_axis(figsize=(4.45, 2.45))
    phi = game_series(gd, "decoy_Phi")
    if phi is not None:
        n = min(len(t), len(phi))
        valid = valid_t[:n] & np.isfinite(phi[:n])
        if valid.any():
            axPhi.plot(np.asarray(t[:n])[valid], phi[:n][valid], color="#0072B2", lw=1.75,
                       label=r"$\Phi_{\mathrm{decoy}}(t)$")
    axPhi.set_ylabel(r"$\Phi_{\mathrm{decoy}}$", color="#0072B2")
    axPhi.tick_params(axis="y", labelcolor="#0072B2")
    axN = axPhi.twinx()
    neff = game_series(gd, "pen_N_eff")
    if neff is not None:
        n = min(len(t), len(neff))
        valid = valid_t[:n] & np.isfinite(neff[:n])
        if valid.any():
            axN.plot(np.asarray(t[:n])[valid], neff[:n][valid], color="#D55E00", lw=1.45, ls="--",
                     label=r"$N_{\mathrm{eff}}(t)$")
    axN.set_ylabel(r"$N_{\mathrm{eff}}$", color="#D55E00")
    axN.tick_params(axis="y", labelcolor="#D55E00")
    axN.grid(False)
    lines1, labels1 = axPhi.get_legend_handles_labels()
    lines2, labels2 = axN.get_legend_handles_labels()
    axPhi.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                 ncol=2, handlelength=1.5, borderaxespad=0.3)
    finish_time_axis(axPhi, t)
    draw_death_and_hit(axPhi, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3c_phi_neff.pdf"))
    print("  fig3c_phi_neff.pdf")

    # fig3d: penetration probability P_pen for each attacker.
    fig, axPen = new_time_axis(figsize=(4.45, 2.45))
    pen_values = plot_attacker_game_series(
        axPen, d, t, game_series(gd, "pen_P_pen"), hitter, ht,
        r"$P^{\mathrm{pen}}_{i}(t)$",
        lambda i: rf"$P^{{\mathrm{{pen}}}}_{{{i}}}$",
    )
    axPen.set_ylim(-0.03, 1.05)
    axPen.legend(loc="upper right", ncol=legend_cols(no), handlelength=1.4, borderaxespad=0.3)
    finish_time_axis(axPen, t)
    draw_death_and_hit(axPen, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3d_pen_prob.pdf"))
    print("  fig3d_pen_prob.pdf")

    # fig3e: escape geometry means Gamma and Xi.
    fig, axGX = new_time_axis(figsize=(4.45, 2.45))
    gx_values = []
    gm = game_series(gd, "esc_Gamma_mean")
    if gm is not None:
        n = min(len(t), len(gm))
        valid = valid_t[:n] & np.isfinite(gm[:n])
        if valid.any():
            gx_values.extend(gm[:n][valid])
            axGX.plot(np.asarray(t[:n])[valid], gm[:n][valid], color="#009E73", lw=1.6,
                      label=r"$\bar{\Gamma}(t)$")
    xi = game_series(gd, "esc_Xi_mean")
    if xi is not None:
        n = min(len(t), len(xi))
        valid = valid_t[:n] & np.isfinite(xi[:n])
        if valid.any():
            gx_values.extend(xi[:n][valid])
            axGX.plot(np.asarray(t[:n])[valid], xi[:n][valid], color="#CC79A7", lw=1.4, ls="--",
                      label=r"$\bar{\Xi}(t)$")
    axGX.axhline(0, color="0.4", ls=":", lw=0.7)
    axGX.set_ylabel(r"$\bar{\Gamma},\ \bar{\Xi}$")
    axGX.legend(loc="upper right", ncol=2, handlelength=1.8, borderaxespad=0.3)
    finish_time_axis(axGX, t)
    draw_death_and_hit(axGX, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3e_gamma_xi.pdf"))
    print("  fig3e_gamma_xi.pdf")

    # fig3f: hit probability and escape ability, following the original game-panel style.
    fig, axPH = new_time_axis(figsize=(4.45, 2.45))
    ph_all = game_series(gd, "hvt_P_hit")
    ee_all = game_series(gd, "esc_E_esc")
    for i in range(no):
        visible = lock_pressure_visible_mask(d, i, t, ht)
        if not visible.any():
            continue
        if ph_all is not None and i < ph_all.shape[0]:
            ph = np.asarray(ph_all[i], dtype=np.float64)[:len(t)]
            axPH.plot(t[visible], ph[visible], color=off_color(i), lw=1.6 if i == hitter else 0.8,
                      alpha=0.9 if i == hitter else 0.55,
                      label=rf"$P^{{\mathrm{{hit}}}}_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        if ee_all is not None and i < ee_all.shape[0]:
            ee = np.asarray(ee_all[i], dtype=np.float64)[:len(t)]
            axPH.plot(t[visible], ee[visible], color=off_color(i), lw=1.1 if i == hitter else 0.6, ls=":",
                      alpha=0.8 if i == hitter else 0.5)
    axPH.set_ylabel(r"$P^{\mathrm{hit}}_{i},\ E^{\mathrm{esc}}_{i}$")
    axPH.legend(loc="upper left", ncol=legend_cols(no), handlelength=1.4, borderaxespad=0.3)
    axPH.annotate(r"solid: $P^{\mathrm{hit}}$   dotted: $E^{\mathrm{esc}}$",
                  xy=(0.60, 1.02), xycoords="axes fraction", fontsize=7.5, ha="left", va="bottom")
    finish_time_axis(axPH, t)
    draw_death_and_hit(axPH, d, t, ht, death_steps)
    save_tight(fig, os.path.join(base, "fig3f_hit_escape.pdf"))
    print("  fig3f_hit_escape.pdf")


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
    hitter = get_hitter(sm); t = d["time"]
    no = n_off(d)
    nd = n_def(d)
    fig = plt.figure(figsize=(7.15, 6.15))
    ax = fig.add_subplot(111, projection="3d")
    style_3d_axes(ax)
    all_x, all_y, all_z = [], [], []
    for i in range(no):
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
        line, = ax.plot(xsm, ysm, zsm, color=off_color(i), lw=lw, alpha=alpha,
                        label=rf"$A_{{{i}}}$" + (r"$^{\star}$" if i == hitter else ""))
        line.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.75),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=off_color(i), marker="o", linewidths=1.1,
                   s=42 if i == hitter else 30, depthshade=False, zorder=7)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=off_color(i),
                   edgecolors="white", marker="D", linewidths=0.8,
                   s=50 if i == hitter else 36, depthshade=False, zorder=8)
        frac = 0.62 if i == hitter else (0.40 + 0.035 * (i % max(no, 1)))
        add_flow_arrows_3d(ax, xsm, ysm, zsm, off_color(i),
                           fractions=(frac,),
                           lw=0.90 if i == hitter else 0.58,
                           mutation_scale=9.5 if i == hitter else 7.0,
                           alpha=0.94 if i == hitter else 0.68)
    for j in range(nd):
        xs = arr([d["def_x"][j][s] for s in range(len(t))])
        ys = arr([d["def_y"][j][s] for s in range(len(t))])
        zs = arr([d["def_z"][j][s] for s in range(len(t))])
        alive = as_int_array([d["def_alive"][j][s] for s in range(len(t))]) == 1
        if not alive.any():
            continue
        xsm, ysm, zsm = xs[alive], ys[alive], zs[alive]
        all_x.extend(xsm); all_y.extend(ysm); all_z.extend(zsm)
        line, = ax.plot(xsm, ysm, zsm, color=def_color(j), lw=0.95, ls=(0, (4, 2)),
                        alpha=0.70, label=rf"$D_{{{j}}}$")
        line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="white", alpha=0.55),
                               pe.Normal()])
        ax.scatter([xsm[0]], [ysm[0]], [zsm[0]], facecolors="white",
                   edgecolors=def_color(j), marker="o", linewidths=0.9,
                   s=24, depthshade=False, zorder=6)
        ax.scatter([xsm[-1]], [ysm[-1]], [zsm[-1]], facecolors=def_color(j),
                   edgecolors="white", marker="D", linewidths=0.7,
                   s=30, depthshade=False, zorder=7)
        add_flow_arrows_3d(ax, xsm, ysm, zsm, def_color(j),
                           fractions=(0.42 + 0.03 * (j % max(nd, 1)),),
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
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_traj3d.pdf")


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
    print("\nAll done.")


if __name__ == "__main__":
    main()
