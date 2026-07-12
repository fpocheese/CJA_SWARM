#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swarm Attack result plotting.

This script plots the raw training outputs produced by
train_simple_converge_v7.py as three standalone, publication-ready figures.
All algorithms use identical statistical processing: per-seed moving average,
cross-seed mean, and standard-error shading. No algorithm-specific curve
reshaping or target-curve blending is applied.
"""
import argparse
import os
import re
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_default_data_dir():
    candidates = [
        os.path.join(SCRIPT_DIR, "raw_npy"),
        os.path.join(SCRIPT_DIR, "results", "swarm_attack"),
        SCRIPT_DIR,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


DEFAULT_DIR = resolve_default_data_dir()

ALGORITHMS = ["Advanced-MAPPO", "MAPPO", "IPPO", "IA2C", "IQL"]
DISPLAY_NAMES = {
    "Advanced-MAPPO": "RTA-MAPPO (Ours)",
    "MAPPO": "MAPPO",
    "IPPO": "IPPO",
    "IA2C": "IA2C",
    "IQL": "IQL",
}
COLORS = {
    "Advanced-MAPPO": "#b2182b",
    "MAPPO": "#2166ac",
    "IPPO": "#1b7837",
    "IA2C": "#b35806",
    "IQL": "#762a83",
}
LINESTYLES = {
    "Advanced-MAPPO": "-",
    "MAPPO": "--",
    "IPPO": "-.",
    "IA2C": ":",
    "IQL": (0, (3, 1, 1, 1)),
}
LINEWIDTHS = {
    "Advanced-MAPPO": 2.3,
    "MAPPO": 1.55,
    "IPPO": 1.55,
    "IA2C": 1.55,
    "IQL": 1.55,
}
ZORDERS = {
    "Advanced-MAPPO": 10,
    "MAPPO": 5,
    "IPPO": 4,
    "IA2C": 3,
    "IQL": 2,
}


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) == 0:
        return values.copy()
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    valid = (cumsum[window:] - cumsum[:-window]) / float(window)
    prefix = np.array([values[:idx + 1].mean() for idx in range(window - 1)])
    return np.concatenate([prefix, valid])


def load_metric(results_dir, algorithm, suffix):
    files = sorted(
        filename for filename in os.listdir(results_dir)
        if filename.startswith(algorithm + "_seed") and filename.endswith(suffix)
    )
    if not files:
        return []
    return [np.load(os.path.join(results_dir, filename)) for filename in files]


def metric_stats(arrays, window, normalize=1.0):
    min_len = min(len(array) for array in arrays)
    matrix = np.array([array[:min_len] for array in arrays], dtype=np.float64)
    smooth = np.array([moving_average(row, window) for row in matrix]) / normalize
    mean = smooth.mean(axis=0)
    sem = smooth.std(axis=0, ddof=1) / np.sqrt(smooth.shape[0]) if smooth.shape[0] > 1 else np.zeros_like(mean)
    return np.arange(min_len), mean, sem


def export_csv(output_dir, figure_name, algorithm, x_values, mean, lower, upper):
    csv_dir = os.path.join(output_dir, "plot_data", figure_name)
    os.makedirs(csv_dir, exist_ok=True)
    output_path = os.path.join(csv_dir, algorithm + ".csv")
    data = np.column_stack([x_values, mean, lower, upper])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="episode,mean,shadow_lower,shadow_upper",
        comments="",
        fmt="%.8f",
    )


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.9,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def plot_metric(results_dir, output_dir, figure_name, suffix, ylabel, *, window, yscale="linear", normalize=1.0):
    plt = setup_style()
    from matplotlib.ticker import AutoMinorLocator

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.1))
    summary_rows = []

    for algorithm in ALGORITHMS:
        arrays = load_metric(results_dir, algorithm, suffix)
        if not arrays:
            continue
        clipped_arrays = [np.maximum(array, 1e-8) for array in arrays] if yscale == "log" else arrays
        x_values, mean, sem = metric_stats(clipped_arrays, window, normalize=normalize)
        lower = np.maximum(mean - sem, 1e-8 if yscale == "log" else 0.0)
        upper = mean + sem

        ax.plot(
            x_values,
            mean,
            label=DISPLAY_NAMES[algorithm],
            color=COLORS[algorithm],
            linestyle=LINESTYLES[algorithm],
            linewidth=LINEWIDTHS[algorithm],
            zorder=ZORDERS[algorithm],
        )
        ax.fill_between(
            x_values,
            lower,
            upper,
            color=COLORS[algorithm],
            alpha=0.16 if algorithm == "Advanced-MAPPO" else 0.10,
            zorder=ZORDERS[algorithm] - 1,
        )
        export_csv(output_dir, figure_name, algorithm, x_values, mean, lower, upper)
        summary_rows.append((algorithm, float(np.mean(mean[-max(1, min(200, len(mean))):])), len(arrays)))

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.20, linewidth=0.45)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yscale != "log":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction="in", which="both")
    ax.legend(
        loc="best",
        ncol=2,
        frameon=True,
        framealpha=0.82,
        edgecolor="#cccccc",
        borderpad=0.35,
        labelspacing=0.22,
        handlelength=2.2,
        columnspacing=0.8,
    )
    fig.tight_layout()

    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, figure_name + "." + extension))
    plt.close(fig)
    return summary_rows


def infer_training_metadata(results_dir):
    lengths = []
    seeds = set()
    pattern = re.compile(r"_seed(\d+)_rewards\.npy$")
    for algorithm in ALGORITHMS:
        for filename in os.listdir(results_dir):
            if filename.startswith(algorithm + "_seed") and filename.endswith("_rewards.npy"):
                values = np.load(os.path.join(results_dir, filename), mmap_mode="r")
                lengths.append(len(values))
                match = pattern.search(filename)
                if match:
                    seeds.add(int(match.group(1)))
    episodes = min(lengths) if lengths else 0
    return episodes, sorted(seeds)


def write_convergence_summary(output_dir, results_dir):
    rows = []
    for algorithm in ALGORITHMS:
        arrays = load_metric(results_dir, algorithm, "_rewards.npy")
        if not arrays:
            continue
        min_len = min(len(array) for array in arrays)
        matrix = np.array([array[:min_len] for array in arrays], dtype=np.float64)
        mean_curve = matrix.mean(axis=0)
        seed_last200 = np.array([array[-min(200, len(array)):].mean() for array in matrix])
        start_900 = min(900, min_len)
        end_1000 = min(1000, min_len)
        start_1000 = min(1000, min_len)
        end_1200 = min(1200, min_len)
        start_1500 = min(1500, min_len)
        end_1700 = min(1700, min_len)
        start_1800 = min(1800, min_len)
        end_2000 = min(2000, min_len)
        last500_start = max(0, min_len - 500)

        mean_900_1000 = mean_curve[start_900:end_1000].mean() if end_1000 > start_900 else np.nan
        mean_1000_1200 = mean_curve[start_1000:end_1200].mean() if end_1200 > start_1000 else np.nan
        mean_1500_1700 = mean_curve[start_1500:end_1700].mean() if end_1700 > start_1500 else np.nan
        mean_1800_2000 = mean_curve[start_1800:end_2000].mean() if end_2000 > start_1800 else np.nan
        delta = (mean_1800_2000 - mean_1000_1200) / (abs(mean_1000_1200) + 1e-8)
        cv_last500 = mean_curve[last500_start:].std() / (abs(mean_curve[last500_start:].mean()) + 1e-8)

        rows.append({
            "algorithm": algorithm,
            "display_name": DISPLAY_NAMES[algorithm],
            "seeds": matrix.shape[0],
            "last200_mean": seed_last200.mean(),
            "last200_seed_std": seed_last200.std(),
            "mean_900_1000": mean_900_1000,
            "mean_1000_1200": mean_1000_1200,
            "mean_1500_1700": mean_1500_1700,
            "mean_1800_2000": mean_1800_2000,
            "delta_1000_to_final": delta,
            "cv_mean_last500": cv_last500,
        })

    csv_path = os.path.join(output_dir, "convergence_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "algorithm,display_name,seeds,last200_mean,last200_seed_std,"
            "mean_900_1000,mean_1000_1200,mean_1500_1700,mean_1800_2000,"
            "delta_1000_to_final,cv_mean_last500\n"
        )
        for row in rows:
            file.write(
                f"{row['algorithm']},{row['display_name']},{row['seeds']},"
                f"{row['last200_mean']:.8f},{row['last200_seed_std']:.8f},"
                f"{row['mean_900_1000']:.8f},{row['mean_1000_1200']:.8f},"
                f"{row['mean_1500_1700']:.8f},{row['mean_1800_2000']:.8f},"
                f"{row['delta_1000_to_final']:.8f},{row['cv_mean_last500']:.8f}\n"
            )
    return rows


def write_report(output_dir, results_dir, reward_rows, loss_rows, entropy_rows, window, reward_norm,
                 train_episodes=None, train_seeds=None):
    inferred_episodes, inferred_seeds = infer_training_metadata(results_dir)
    train_episodes = train_episodes or inferred_episodes
    train_seeds = train_seeds or inferred_seeds
    seeds_text = " ".join(str(seed) for seed in train_seeds)
    command_data_dir = os.path.relpath(results_dir, os.getcwd()).replace(os.sep, "/")
    command_output_dir = os.path.relpath(output_dir, os.getcwd()).replace(os.sep, "/")
    convergence_rows = write_convergence_summary(output_dir, results_dir)
    report_path = os.path.join(output_dir, "TRAINING_AND_PLOTTING_METHOD.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Swarm Attack 模型训练结果图绘制方法\n\n")
        file.write("## 数据来源\n\n")
        file.write(f"- 原始结果目录: `{results_dir}`\n")
        file.write("- 训练脚本: `onpolicy/scripts/train_simple_converge_v7.py`\n")
        file.write("- 绘图脚本: `onpolicy/scripts/plot_results_swarm_attack.py`\n")
        file.write("- 对比算法: Advanced-MAPPO, MAPPO, IPPO, IA2C, IQL\n\n")
        file.write("## 本轮训练设置\n\n")
        file.write(f"- 总训练回合数: `{train_episodes}` episode。\n")
        file.write(f"- 随机种子: `{seeds_text}`。\n")
        file.write("- MAPPO 使用集中式 critic 并降低后期学习率，使曲线在 2000 episode 内收敛。\n")
        file.write("- Advanced-MAPPO/Ours 使用 hold-then-decay 学习率和熵系数调度，前期保持学习强度，约 1000 episode 后进入低学习率平台期。\n")
        file.write("- IPPO、IA2C、IQL 沿用原配置，仅跑满 2000 episode。\n\n")
        file.write("## 原图代码定位\n\n")
        file.write("- 原三联图绘图脚本: `onpolicy/scripts/plot_results_ieee.py` 和 `onpolicy/scripts/plot_results_ieee_v2.py`。\n")
        file.write("- 已有三图分离脚本: `onpolicy/scripts/plot_results_ieee_v3.py`，会输出 `fig_a_reward`, `fig_b_critic_loss`, `fig_c_entropy`。\n")
        file.write("- 本次实验使用 `onpolicy/scripts/plot_results_swarm_attack.py`，直接输出三张独立论文图，并对所有算法采用同一套统计处理。\n\n")
        file.write("## 统计与绘图方法\n\n")
        file.write(f"- 每条 seed 曲线先使用 `{window}` episode 移动平均进行平滑。\n")
        file.write("- 实线表示跨 seed 均值，阴影表示均值标准误差。\n")
        file.write(f"- Reward 为了紧凑展示除以 `{reward_norm}`；critic loss 使用 log 坐标；entropy 不做归一化。\n")
        file.write("- 所有算法使用完全一致的平滑、均值和阴影计算流程，不对单个算法做额外曲线重塑或目标曲线拼接。\n\n")
        file.write("## 输出文件\n\n")
        file.write("- `swarm_attack_reward.png` / `swarm_attack_reward.pdf`: 归一化团队奖励。\n")
        file.write("- `swarm_attack_critic_loss.png` / `swarm_attack_critic_loss.pdf`: critic loss, log 坐标。\n")
        file.write("- `swarm_attack_entropy.png` / `swarm_attack_entropy.pdf`: policy entropy。\n")
        file.write("- `plot_data/`: 每条曲线的均值和阴影上下界 CSV。\n\n")
        file.write("## 收敛性复核\n\n")
        file.write("| Algorithm | Last 200 reward | 1000-1200 | 1800-2000 | Delta | CV last500 |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in convergence_rows:
            file.write(
                f"| {row['display_name']} | {row['last200_mean']:.2f} | "
                f"{row['mean_1000_1200']:.2f} | {row['mean_1800_2000']:.2f} | "
                f"{row['delta_1000_to_final']:.4f} | {row['cv_mean_last500']:.4f} |\n"
            )
        file.write("\n")
        file.write("## 复现实验命令\n\n")
        file.write("```bash\n")
        file.write("conda run -n rlgpu python onpolicy/scripts/train_simple_converge_v7.py \\\n")
        file.write(f"  --save_dir {command_data_dir} \\\n")
        file.write(f"  --num_episodes {train_episodes} \\\n")
        file.write(f"  --seeds {seeds_text} \\\n")
        file.write("  --algorithms Advanced-MAPPO MAPPO IPPO IA2C IQL\n\n")
        file.write("conda run -n rlgpu python onpolicy/scripts/plot_results_swarm_attack.py \\\n")
        file.write(f"  --data_dir {command_data_dir} \\\n")
        file.write(f"  --output_dir {command_output_dir} \\\n")
        file.write(f"  --train_episodes {train_episodes} \\\n")
        file.write(f"  --train_seeds {seeds_text}\n")
        file.write("```\n\n")
        file.write("## 当前结果摘要\n\n")
        file.write("| Metric | Algorithm | Final mean | Seeds |\n")
        file.write("| --- | --- | ---: | ---: |\n")
        for metric_name, rows in (("Reward", reward_rows), ("Critic loss", loss_rows), ("Entropy", entropy_rows)):
            for algorithm, final_mean, seed_count in rows:
                file.write(f"| {metric_name} | {DISPLAY_NAMES[algorithm]} | {final_mean:.6f} | {seed_count} |\n")
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("metric,algorithm,display_name,final_mean,seeds\n")
        for metric_name, rows in (("Reward", reward_rows), ("Critic loss", loss_rows), ("Entropy", entropy_rows)):
            for algorithm, final_mean, seed_count in rows:
                file.write(f"{metric_name},{algorithm},{DISPLAY_NAMES[algorithm]},{final_mean:.8f},{seed_count}\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Plot swarm_attack training results as three standalone figures.")
    parser.add_argument("--data_dir", default=DEFAULT_DIR)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--reward_norm", type=float, default=65.0)
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--train_seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir or (SCRIPT_DIR if os.path.basename(data_dir) == "raw_npy" else data_dir))
    if not os.path.isdir(data_dir):
        print("ERROR: data directory not found: %s" % data_dir)
        sys.exit(1)

    reward_rows = plot_metric(
        data_dir, output_dir, "swarm_attack_reward", "_rewards.npy", "Reward",
        window=args.window, normalize=args.reward_norm,
    )
    loss_rows = plot_metric(
        data_dir, output_dir, "swarm_attack_critic_loss", "_critic_loss.npy", "Critic Loss",
        window=args.window, yscale="log",
    )
    entropy_rows = plot_metric(
        data_dir, output_dir, "swarm_attack_entropy", "_entropy.npy", "Policy Entropy",
        window=args.window,
    )
    report_path = write_report(
        output_dir, data_dir, reward_rows, loss_rows, entropy_rows,
        args.window, args.reward_norm, args.train_episodes, args.train_seeds,
    )

    print("Swarm attack figures saved to: %s" % output_dir)
    print("Method report saved to: %s" % report_path)


if __name__ == "__main__":
    main()