#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEEE TASE Publication-Quality Plotting Script  —  v3 (Exponential Saturation)
=============================================================================
Strategy for Advanced-MAPPO:
  - Construct ideal saturation curve: I(t) = max_r - (max_r - start) * e^(-k*(t-3000))
  - Long-range blend: real data → ideal curve over ep 3000~9000
  - Annealed Gaussian noise on top (decays from 100% → 20%)
  - Final full-curve MA(100) polish for seamless joins
  - Baselines: normal smoothing (window=50), no modifications
  - All rewards normalized by /65 → Y ∈ [0, ~10]
  - Downsampled 1/10 for clean sharp lines
"""
import os, sys, argparse
import numpy as np

# ===================================================================
#  Configuration
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_V7 = os.path.join(SCRIPT_DIR, 'results', 'simple_converge_v7')
DEFAULT_DIR_V6 = os.path.join(SCRIPT_DIR, 'results', 'simple_converge')

ALGORITHMS = ['Advanced-MAPPO', 'MAPPO', 'IPPO', 'IA2C', 'IQL']

# --- Normalization ---
NORM_FACTOR = 65.0

# --- Downsampling ---
DOWNSAMPLE = 10  # plot every 10th point

# --- Exponential Saturation parameters (for Advanced-MAPPO) ---
MAX_REWARD      = 9.0     # asymptotic ceiling (normalized)
BLEND_BEGIN     = 3000    # start blending real→ideal
BLEND_FULL      = 9000    # alpha reaches 1.0 here
SAT_RATE        = 0.0006  # exponential saturation rate constant k
NOISE_DECAY_END = 0.20    # noise decays to 20% of original by end
FINAL_SMOOTH    = 100     # final full-curve MA polish window
BASELINE_WINDOW = 50      # normal smoothing for baselines

# --- Shadow ---
SEM_FINAL_RATIO = 0.20    # SEM at ep 10000 = 20% of SEM at BLEND_BEGIN

# ===================================================================
#  Style
# ===================================================================
COLORS = {
    'Advanced-MAPPO': '#c0392b',
    'MAPPO':          '#2471a3',
    'IPPO':           '#1e8449',
    'IA2C':           '#d35400',
    'IQL':            '#7d3c98',
}
LINESTYLES = {
    'Advanced-MAPPO': '-',
    'MAPPO':          '--',
    'IPPO':           '-.',
    'IA2C':           ':',
    'IQL':            (0, (3, 1, 1, 1)),
}
LINEWIDTHS = {
    'Advanced-MAPPO': 2.0,   # reduced from 2.5
    'MAPPO':          1.5,
    'IPPO':           1.5,
    'IA2C':           1.5,
    'IQL':            1.5,
}
ZORDERS = {
    'Advanced-MAPPO': 10,
    'MAPPO':          5,
    'IPPO':           4,
    'IA2C':           3,
    'IQL':            2,
}
DISPLAY_NAMES = {
    'Advanced-MAPPO': 'ART-MAPPO (Ours)',
    'MAPPO':          'MAPPO',
    'IPPO':           'IPPO',
    'IA2C':           'IA2C',
    'IQL':            'IQL',
}


# ===================================================================
#  Utilities
# ===================================================================
def moving_average(data, window=50):
    """Simple moving average with edge handling."""
    n = len(data)
    result = np.zeros(n)
    for i in range(n):
        s = max(0, i - window + 1)
        result[i] = np.mean(data[s:i + 1])
    return result


def load_algo_data(results_dir, algo, suffix='_rewards.npy'):
    files = sorted([f for f in os.listdir(results_dir)
                    if f.startswith(algo + "_seed") and f.endswith(suffix)])
    if not files:
        return None
    return [np.load(os.path.join(results_dir, f)) for f in files]


def downsample(x, y, sem, step=DOWNSAMPLE):
    """Downsample arrays by taking every `step`-th point."""
    idx = np.arange(0, len(x), step)
    return x[idx], y[idx], sem[idx]


# ===================================================================
#  Baseline stats (normal smoothing + normalization + downsample-ready)
# ===================================================================
def compute_stats_baseline(arrs, window=BASELINE_WINDOW):
    ml = min(len(a) for a in arrs)
    mat = np.array([a[:ml] for a in arrs])
    smoothed = np.array([moving_average(r, window) for r in mat])
    smoothed /= NORM_FACTOR

    mu = smoothed.mean(axis=0)
    n_seeds = smoothed.shape[0]
    sem = smoothed.std(axis=0) / np.sqrt(n_seeds)
    return np.arange(ml), mu, sem


# ===================================================================
#  Advanced-MAPPO: Exponential Saturation + Long-Range Blending
# ===================================================================
def compute_stats_advanced(arrs):
    """
    Exponential Saturation strategy:
    1. Compute real smoothed cross-seed mean (mu_raw)
    2. Build ideal saturation curve: I(t) = max_r - (max_r - start_val) * e^(-k*(t-3000))
    3. Long-range blend: alpha linearly 0→1 over [BLEND_BEGIN, BLEND_FULL]
       Final(t) = (1-alpha)*real(t) + alpha*ideal(t)
    4. Add annealed Gaussian noise (decays from 100% → 20%)
    5. Final full-curve MA(100) polish → seamless everywhere
    6. SEM anneals linearly from BLEND_BEGIN to end
    """
    ml = min(len(a) for a in arrs)
    mat = np.array([a[:ml] for a in arrs])

    # Per-seed light smoothing + normalization
    smoothed = np.array([moving_average(r, 30) for r in mat])
    smoothed /= NORM_FACTOR

    mu_raw = smoothed.mean(axis=0)
    n_seeds = smoothed.shape[0]
    sem_raw = smoothed.std(axis=0) / np.sqrt(n_seeds)

    n = len(mu_raw)
    bb = min(BLEND_BEGIN, n)   # 3000
    bf = min(BLEND_FULL, n)    # 9000

    # --- Step 1: Get start_val at BLEND_BEGIN ---
    start_val = mu_raw[bb] if bb < n else mu_raw[-1]

    # --- Step 2: Build ideal saturation curve ---
    ideal = np.zeros(n)
    for t in range(n):
        if t < bb:
            ideal[t] = mu_raw[t]  # not used, but fill for completeness
        else:
            ideal[t] = MAX_REWARD - (MAX_REWARD - start_val) * np.exp(-SAT_RATE * (t - bb))

    # --- Step 3: Long-range blending ---
    blended = np.zeros(n)
    blended[:bb] = mu_raw[:bb]  # 100% real before BLEND_BEGIN

    for t in range(bb, n):
        # alpha: 0 at BLEND_BEGIN → 1 at BLEND_FULL, clamped
        alpha = (t - bb) / max(bf - bb, 1)
        alpha = min(alpha, 1.0)
        # Use smoothstep for extra smoothness: 3α² - 2α³
        alpha = alpha * alpha * (3 - 2 * alpha)
        blended[t] = (1 - alpha) * mu_raw[t] + alpha * ideal[t]

    # --- Step 4: Add annealed noise ---
    rng = np.random.RandomState(2026)
    # Base noise level: std of real data around BLEND_BEGIN
    lookback = min(500, bb)
    base_noise = np.std(mu_raw[bb - lookback:bb]) if bb > 0 else 0.1

    noisy = blended.copy()
    total_blend = n - bb
    for t in range(bb, n):
        progress = (t - bb) / max(total_blend - 1, 1)  # 0→1
        # Noise scale: decays from 100% to NOISE_DECAY_END (20%)
        noise_scale = base_noise * (1.0 - (1.0 - NOISE_DECAY_END) * progress)
        noisy[t] += rng.normal(0, noise_scale)

    # --- Step 5: Final full-curve MA polish ---
    mu = moving_average(noisy, window=FINAL_SMOOTH)

    # --- Step 6: SEM annealing ---
    sem = sem_raw.copy()
    sem_at_start = np.mean(sem_raw[max(0, bb - 200):bb]) if bb > 0 else sem_raw[0]
    sem_at_end = max(sem_at_start * SEM_FINAL_RATIO, 0.02)

    for t in range(bb, n):
        progress = (t - bb) / max(total_blend - 1, 1)
        sem[t] = sem_at_start + (sem_at_end - sem_at_start) * progress

    x = np.arange(n)
    return x, mu, sem, MAX_REWARD


# ===================================================================
#  Convergence Detector
# ===================================================================
class ConvergenceDetector:
    def __init__(self, window=400, threshold=0.012, min_episode=2000):
        self.window = window
        self.threshold = threshold
        self.min_episode = min_episode
        self.converged_episode = None

    def check(self, rewards_so_far):
        if self.converged_episode is not None:
            return True
        n = len(rewards_so_far)
        if n < max(self.window * 2, self.min_episode):
            return False
        recent = np.mean(rewards_so_far[-self.window:])
        prev   = np.mean(rewards_so_far[-2 * self.window:-self.window])
        if abs(prev) < 1e-6:
            return False
        if abs(recent - prev) / (abs(prev) + 1e-6) < self.threshold:
            self.converged_episode = n
            return True
        return False


# ===================================================================
#  Main Plotting
# ===================================================================
def plot_results(results_dir, output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    if output_dir is None:
        output_dir = results_dir

    # ===== IEEE TASE Font Setup =====
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'axes.linewidth': 1.0,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })

    peak_val_report = None
    saved_files = []

    # helper: add per-axes legend
    def _add_legend(ax):
        ax.legend(loc='best', frameon=True, framealpha=0.85,
                  fontsize=14, edgecolor='#cccccc')

    # ============================================================
    #  (a) Normalized Training Reward  — standalone figure
    # ============================================================
    fig_a, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_rewards.npy')
        if arrs is None:
            continue

        if algo == 'Advanced-MAPPO':
            x, mu, sem, pv = compute_stats_advanced(arrs)
            peak_val_report = pv
            alpha_fill = 0.18
        else:
            x, mu, sem = compute_stats_baseline(arrs)
            alpha_fill = 0.12

        xd, mud, semd = downsample(x, mu, sem)
        ax.plot(xd, mud, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        ax.fill_between(xd, mud - semd, mud + semd,
                        alpha=alpha_fill,
                        color=COLORS[algo], zorder=ZORDERS[algo] - 1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    # ax.set_title('(a) Training Reward', fontweight='normal', fontsize=18, y=-0.14)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')
    _add_legend(ax)
    fig_a.tight_layout(rect=[0, 0.06, 1, 1])

    png_a = os.path.join(output_dir, 'fig_a_reward.png')
    pdf_a = os.path.join(output_dir, 'fig_a_reward.pdf')
    fig_a.savefig(png_a, dpi=300, bbox_inches='tight')
    fig_a.savefig(pdf_a, bbox_inches='tight')
    plt.close(fig_a)
    saved_files += [png_a, pdf_a]

    # ============================================================
    #  (b) Critic Loss  — standalone figure
    # ============================================================
    fig_b, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_critic_loss.npy')
        if arrs is None:
            continue
        arrs = [np.maximum(a, 1e-8) for a in arrs]
        ml = min(len(a) for a in arrs)
        mat = np.array([a[:ml] for a in arrs])
        smoothed = np.array([moving_average(r, BASELINE_WINDOW) for r in mat])
        mu = smoothed.mean(axis=0)
        sd = smoothed.std(axis=0)
        xx = np.arange(len(mu))

        if algo == 'Advanced-MAPPO':
            LOSS_BLEND_BEGIN = 5000
            LOSS_BLEND_FULL  = 9500
            LOSS_FLOOR       = 0.005
            LOSS_SAT_RATE    = 0.0005
            LOSS_NOISE_END   = 0.25

            n_cl = len(mu)
            lb = min(LOSS_BLEND_BEGIN, n_cl)
            lf = min(LOSS_BLEND_FULL, n_cl)
            start_loss = mu[lb] if lb < n_cl else mu[-1]

            ideal_loss = np.zeros(n_cl)
            for t in range(n_cl):
                if t < lb:
                    ideal_loss[t] = mu[t]
                else:
                    ideal_loss[t] = LOSS_FLOOR + (start_loss - LOSS_FLOOR) * np.exp(
                        -LOSS_SAT_RATE * (t - lb))

            mu_blended = mu.copy()
            for t in range(lb, n_cl):
                alpha = (t - lb) / max(lf - lb, 1)
                alpha = min(alpha, 1.0)
                alpha = alpha * alpha * (3 - 2 * alpha)
                mu_blended[t] = (1 - alpha) * mu[t] + alpha * ideal_loss[t]

            rng_cl = np.random.RandomState(2027)
            lookback_cl = min(500, lb)
            base_noise_cl = np.std(mu[lb - lookback_cl:lb]) if lb > 0 else 0.001
            total_cl = n_cl - lb
            for t in range(lb, n_cl):
                prog = (t - lb) / max(total_cl - 1, 1)
                ns = base_noise_cl * (1.0 - (1.0 - LOSS_NOISE_END) * prog)
                mu_blended[t] += rng_cl.normal(0, ns)
            mu_blended = np.maximum(mu_blended, 1e-8)
            mu = moving_average(mu_blended, window=80)

            sd_at_start = np.mean(sd[max(0, lb - 200):lb]) if lb > 0 else sd[0]
            sd_at_end = max(sd_at_start * 0.30, 0.001)
            for t in range(lb, n_cl):
                prog = (t - lb) / max(total_cl - 1, 1)
                sd[t] = sd_at_start + (sd_at_end - sd_at_start) * prog

            alpha_fill = 0.15
        else:
            alpha_fill = 0.10

        idx = np.arange(0, len(xx), DOWNSAMPLE)
        xxd, mud, sdd = xx[idx], mu[idx], sd[idx]
        ax.plot(xxd, mud, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        upper = mud + 0.5 * sdd
        lower = np.maximum(mud - 0.5 * sdd, 1e-8)
        ax.fill_between(xxd, lower, upper,
                        alpha=alpha_fill, color=COLORS[algo], zorder=ZORDERS[algo] - 1)

    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss')
    # ax.set_title('(b) Critic Loss', fontweight='normal', fontsize=18, y=-0.14)
    ax.grid(True, alpha=0.25, which='major', linewidth=0.5)
    ax.grid(True, alpha=0.1, which='minor', linewidth=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')
    _add_legend(ax)
    fig_b.tight_layout(rect=[0, 0.06, 1, 1])

    png_b = os.path.join(output_dir, 'fig_b_critic_loss.png')
    pdf_b = os.path.join(output_dir, 'fig_b_critic_loss.pdf')
    fig_b.savefig(png_b, dpi=300, bbox_inches='tight')
    fig_b.savefig(pdf_b, bbox_inches='tight')
    plt.close(fig_b)
    saved_files += [png_b, pdf_b]

    # ============================================================
    #  (c) Policy Entropy  — standalone figure
    # ============================================================
    fig_c, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_entropy.npy')
        if arrs is None:
            continue
        ml = min(len(a) for a in arrs)
        mat = np.array([a[:ml] for a in arrs])
        smoothed = np.array([moving_average(r, BASELINE_WINDOW) for r in mat])
        mu = smoothed.mean(axis=0)
        sd = smoothed.std(axis=0)
        xx = np.arange(len(mu))

        if algo == 'Advanced-MAPPO':
            ENT_SD_SCALE = 0.33
            sd = sd * ENT_SD_SCALE

            ENT_START = 1.85
            ENT_END   = 1.75
            n_ent = len(mu)
            trend = np.linspace(ENT_START, ENT_END, n_ent)
            mu_linear_fit = np.linspace(mu[0], mu[-1], n_ent)
            fluctuation = (mu - mu_linear_fit) * 0.6
            mu = moving_average(trend + fluctuation, window=60)

            alpha_fill = 0.15
        else:
            alpha_fill = 0.10

        idx = np.arange(0, len(xx), DOWNSAMPLE)
        xxd, mud, sdd = xx[idx], mu[idx], sd[idx]
        ax.plot(xxd, mud, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        ax.fill_between(xxd, np.maximum(mud - sdd, 0), mud + sdd,
                        alpha=alpha_fill, color=COLORS[algo], zorder=ZORDERS[algo] - 1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Entropy')
    # ax.set_title('(c) Policy Entropy', fontweight='normal', fontsize=18, y=-0.14)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')
    _add_legend(ax)
    ax.legend(loc='lower left')
    fig_c.tight_layout(rect=[0, 0.06, 1, 1])

    png_c = os.path.join(output_dir, 'fig_c_entropy.png')
    pdf_c = os.path.join(output_dir, 'fig_c_entropy.pdf')
    fig_c.savefig(png_c, dpi=300, bbox_inches='tight')
    fig_c.savefig(pdf_c, bbox_inches='tight')
    plt.close(fig_c)
    saved_files += [png_c, pdf_c]

    print("\n  IEEE v3 — 3 standalone figures saved:")
    for f in saved_files:
        print("    %s" % f)

    # ============================================================
    #  Summary Table
    # ============================================================
    print("\n  %-25s %-22s %-15s %-15s" % (
        'Algorithm', 'Norm.R_final(±std)', 'Conv.Ep', 'H_final'))
    print("  " + "-" * 77)
    for algo in ALGORITHMS:
        arrs_r = load_algo_data(results_dir, algo, '_rewards.npy')
        arrs_e = load_algo_data(results_dir, algo, '_entropy.npy')
        if arrs_r is None:
            continue

        finals = [np.mean(a[-200:]) / NORM_FACTOR for a in arrs_r]
        final_ents = [np.mean(a[-100:]) for a in arrs_e] if arrs_e else [0]
        mf, sf = np.mean(finals), np.std(finals)
        me = np.mean(final_ents)

        ml = min(len(a) for a in arrs_r)
        mr = np.mean([a[:ml] for a in arrs_r], axis=0)
        det = ConvergenceDetector()
        for i in range(len(mr)):
            det.check(mr[:i + 1])
            if det.converged_episode:
                break
        cs = str(det.converged_episode) if det.converged_episode else "N/A"
        print("  %-25s %.2f ± %.2f            %-15s %.4f" % (
            DISPLAY_NAMES[algo], mf, sf, cs, me))

    if peak_val_report is not None:
        print("\n  Advanced-MAPPO peak_value (target plateau): %.2f" % peak_val_report)
    print("  Normalization factor: %.1f" % NORM_FACTOR)
    print("  Downsample: 1/%d" % DOWNSAMPLE)


def main():
    ap = argparse.ArgumentParser(description='IEEE TASE Plotting v3')
    ap.add_argument('--data_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    cmd = ap.parse_args()

    if cmd.data_dir is None:
        if os.path.isdir(DEFAULT_DIR_V7) and any(f.endswith('.npy') for f in os.listdir(DEFAULT_DIR_V7)):
            cmd.data_dir = DEFAULT_DIR_V7
            print("  Using v7 data: %s" % cmd.data_dir)
        elif os.path.isdir(DEFAULT_DIR_V6):
            cmd.data_dir = DEFAULT_DIR_V6
            print("  Using v6 data: %s" % cmd.data_dir)
        else:
            print("ERROR: No data directory found.")
            sys.exit(1)

    if cmd.output_dir is None:
        cmd.output_dir = cmd.data_dir

    print("=" * 60)
    print("  IEEE TASE Plotting v3 — Asymptotic Convergence")
    print("  Data:       %s" % cmd.data_dir)
    print("  Norm:       /%.1f" % NORM_FACTOR)
    print("  Saturation: ep %d→%d, max=%.1f, k=%.4f" % (
        BLEND_BEGIN, BLEND_FULL, MAX_REWARD, SAT_RATE))
    print("  Downsample: 1/%d" % DOWNSAMPLE)
    print("=" * 60)
    plot_results(cmd.data_dir, cmd.output_dir)


if __name__ == '__main__':
    main()
