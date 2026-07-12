#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEEE TASE Publication-Quality Plotting Script
- No MADDPG (deterministic policy, entropy not comparable)
- Moving Average smoothing (window=50)
- Times New Roman fonts, 18/16/14 sizes
- Advanced-MAPPO: red solid lw=2.5 highest zorder
- Others: dashed/dotted lw=1.5
- Shared horizontal legend on top
- (a) Reward, (b) Critic Loss, (c) Policy Entropy
"""
import os, sys, argparse
import numpy as np

# ===================================================================
#  Data directory
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_V7 = os.path.join(SCRIPT_DIR, 'results', 'simple_converge_v7')
DEFAULT_DIR_V6 = os.path.join(SCRIPT_DIR, 'results', 'simple_converge')

# ===================================================================
#  Algorithms (no MADDPG)
# ===================================================================
ALGORITHMS = ['Advanced-MAPPO', 'MAPPO', 'IPPO', 'IA2C', 'IQL']

# ===================================================================
#  Style configuration
# ===================================================================
COLORS = {
    'Advanced-MAPPO': '#c0392b',  # deep red
    'MAPPO':          '#2471a3',  # steel blue
    'IPPO':           '#1e8449',  # forest green
    'IA2C':           '#d35400',  # burnt orange
    'IQL':            '#7d3c98',  # dark purple
}
LINESTYLES = {
    'Advanced-MAPPO': '-',
    'MAPPO':          '--',
    'IPPO':           '-.',
    'IA2C':           ':',
    'IQL':            (0, (3, 1, 1, 1)),  # densely dash-dotted
}
LINEWIDTHS = {
    'Advanced-MAPPO': 2.5,
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
# Display names for legend
DISPLAY_NAMES = {
    'Advanced-MAPPO': 'Advanced-MAPPO (Ours)',
    'MAPPO':          'MAPPO',
    'IPPO':           'IPPO',
    'IA2C':           'IA2C',
    'IQL':            'IQL',
}


# ===================================================================
#  Smoothing: Moving Average
# ===================================================================
def moving_average(data, window=50):
    """Simple moving average with edge handling."""
    n = len(data)
    result = np.zeros(n)
    for i in range(n):
        s = max(0, i - window + 1)
        result[i] = np.mean(data[s:i+1])
    return result


# ===================================================================
#  Load data for one algorithm
# ===================================================================
def load_algo_data(results_dir, algo, suffix='_rewards.npy'):
    """Load all seeds for an algorithm, return list of arrays."""
    files = sorted([f for f in os.listdir(results_dir)
                    if f.startswith(algo + "_seed") and f.endswith(suffix)])
    if not files:
        return None
    arrs = [np.load(os.path.join(results_dir, f)) for f in files]
    return arrs


def compute_stats(arrs, window=50):
    """Compute smoothed mean and std across seeds."""
    ml = min(len(a) for a in arrs)
    arrs = np.array([a[:ml] for a in arrs])
    smoothed = np.array([moving_average(r, window) for r in arrs])
    mu = smoothed.mean(axis=0)
    sd = smoothed.std(axis=0)
    x = np.arange(len(mu))
    return x, mu, sd


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
        prev = np.mean(rewards_so_far[-2 * self.window:-self.window])
        if abs(prev) < 1e-6:
            return False
        if abs(recent - prev) / (abs(prev) + 1e-6) < self.threshold:
            self.converged_episode = n
            return True
        return False


# ===================================================================
#  Main Plotting Function
# ===================================================================
def plot_results(results_dir, output_dir=None, window=50):
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # ============================================================
    #  (a) Training Reward
    # ============================================================
    ax = axes[0]
    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_rewards.npy')
        if arrs is None:
            continue
        x, mu, sd = compute_stats(arrs, window)
        ax.plot(x, mu, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        ax.fill_between(x, mu - sd, mu + sd,
                        alpha=0.15 if algo == 'Advanced-MAPPO' else 0.10,
                        color=COLORS[algo], zorder=ZORDERS[algo]-1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Team Reward')
    ax.set_title('(a) Training Reward')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    # ============================================================
    #  (b) Critic Loss
    # ============================================================
    ax = axes[1]
    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_critic_loss.npy')
        if arrs is None:
            continue
        # Clip to positive for safety
        arrs = [np.maximum(a, 1e-8) for a in arrs]
        x, mu, sd = compute_stats(arrs, window)
        ax.plot(x, mu, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        upper = mu + 0.5 * sd
        lower = np.maximum(mu - 0.5 * sd, 1e-8)
        ax.fill_between(x, lower, upper,
                        alpha=0.10, color=COLORS[algo], zorder=ZORDERS[algo]-1)

    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss')
    ax.set_title('(b) Critic Loss')
    ax.grid(True, alpha=0.25, which='major', linewidth=0.5)
    ax.grid(True, alpha=0.1, which='minor', linewidth=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    # ============================================================
    #  (c) Policy Entropy
    # ============================================================
    ax = axes[2]
    for algo in ALGORITHMS:
        arrs = load_algo_data(results_dir, algo, '_entropy.npy')
        if arrs is None:
            continue
        x, mu, sd = compute_stats(arrs, window)
        ax.plot(x, mu, label=DISPLAY_NAMES[algo],
                color=COLORS[algo], linestyle=LINESTYLES[algo],
                linewidth=LINEWIDTHS[algo], zorder=ZORDERS[algo])
        ax.fill_between(x, np.maximum(mu - sd, 0), mu + sd,
                        alpha=0.10, color=COLORS[algo], zorder=ZORDERS[algo]-1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Entropy')
    ax.set_title('(c) Policy Entropy')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    # ============================================================
    #  Shared Legend (top center, horizontal)
    # ============================================================
    # Collect handles from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(ALGORITHMS), frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for top legend

    # Save
    png_path = os.path.join(output_dir, 'comparison_ieee_tase.png')
    pdf_path = os.path.join(output_dir, 'comparison_ieee_tase.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print("  IEEE plots saved:")
    print("    PNG: %s" % png_path)
    print("    PDF: %s" % pdf_path)

    # ============================================================
    #  Print summary table
    # ============================================================
    print("\n  %-25s %-18s %-15s %-15s" % ('Algorithm', 'R_final(±std)', 'Conv.Ep', 'H_final'))
    print("  " + "-"*73)
    for algo in ALGORITHMS:
        arrs_r = load_algo_data(results_dir, algo, '_rewards.npy')
        arrs_e = load_algo_data(results_dir, algo, '_entropy.npy')
        if arrs_r is None:
            continue
        finals = [np.mean(a[-100:]) for a in arrs_r]
        final_ents = [np.mean(a[-100:]) for a in arrs_e] if arrs_e else [0]
        mf, sf = np.mean(finals), np.std(finals)
        me = np.mean(final_ents)
        # Convergence
        ml = min(len(a) for a in arrs_r)
        mr = np.mean([a[:ml] for a in arrs_r], axis=0)
        det = ConvergenceDetector()
        for i in range(len(mr)):
            det.check(mr[:i+1])
            if det.converged_episode:
                break
        cs = str(det.converged_episode) if det.converged_episode else "N/A"
        print("  %-25s %.1f ± %.1f        %-15s %.4f" % (DISPLAY_NAMES[algo], mf, sf, cs, me))


def main():
    ap = argparse.ArgumentParser(description='IEEE TASE Publication Plotting')
    ap.add_argument('--data_dir', type=str, default=None,
                    help='Directory with .npy result files')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for plots (default: same as data_dir)')
    ap.add_argument('--window', type=int, default=50,
                    help='Moving average window size')
    cmd = ap.parse_args()

    # Auto-detect data directory
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

    print("="*60)
    print("  IEEE TASE Plotting")
    print("  Data: %s" % cmd.data_dir)
    print("  Window: %d" % cmd.window)
    print("="*60)
    plot_results(cmd.data_dir, cmd.output_dir, cmd.window)


if __name__ == '__main__':
    main()
