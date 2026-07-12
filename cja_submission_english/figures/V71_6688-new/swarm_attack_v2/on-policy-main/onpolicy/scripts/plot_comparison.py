"""
Publication-Quality Comparison Plot Generator
生成顶刊级别的多算法训练对比图

Features:
- Mean ± Std shading (多种子统计)
- 学术风格排版 (serif字体, LaTeX风格标签)
- 支持多指标 (reward, win_rate等)
- 平滑曲线 (Savitzky-Golay / 移动平均)
- 高DPI输出 (300+), PDF/PNG格式
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========== 全局学术风格配置 ==========
def set_journal_style():
    """设置顶刊论文级别的matplotlib样式"""
    rcParams.update({
        # 字体
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        # 线条
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        # 坐标轴
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        # 刻度
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        # 图例
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        # 图形
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# ========== 算法颜色和样式 ==========
ALGO_STYLES = {
    'Advanced-MAPPO': {
        'color': '#E74C3C',     # 红色 - 主角
        'linestyle': '-',
        'marker': 'o',
        'label': 'Advanced-MAPPO (Ours)',
        'zorder': 10,
    },
    'MAPPO': {
        'color': '#3498DB',     # 蓝色
        'linestyle': '-',
        'marker': 's',
        'label': 'MAPPO',
        'zorder': 8,
    },
    'IPPO': {
        'color': '#2ECC71',     # 绿色
        'linestyle': '--',
        'marker': '^',
        'label': 'IPPO',
        'zorder': 6,
    },
    'IA2C': {
        'color': '#9B59B6',     # 紫色
        'linestyle': '-.',
        'marker': 'D',
        'label': 'IA2C',
        'zorder': 4,
    },
    'IQL': {
        'color': '#F39C12',     # 橙色
        'linestyle': ':',
        'marker': 'v',
        'label': 'IQL',
        'zorder': 2,
    },
}

# 备用颜色 (如果有额外算法)
EXTRA_COLORS = ['#1ABC9C', '#E67E22', '#34495E', '#95A5A6']


def smooth_curve(y, window_size=7, poly_order=3):
    """
    Savitzky-Golay平滑, 若数据太短则用移动平均
    """
    if len(y) < window_size:
        return y
    try:
        from scipy.signal import savgol_filter
        if window_size % 2 == 0:
            window_size += 1
        return savgol_filter(y, window_size, poly_order)
    except ImportError:
        # 退化到简单移动平均
        kernel = np.ones(window_size) / window_size
        return np.convolve(y, kernel, mode='same')


def moving_average(y, window=5):
    """简单移动平均"""
    if len(y) <= window:
        return y
    ret = np.cumsum(y, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    result = ret[window - 1:] / window
    # 前面填充
    pad = y[:window - 1]
    return np.concatenate([pad, result])


def load_reward_curves(save_dir, algo_list):
    """
    加载所有算法的reward曲线数据
    
    数据文件格式: {save_dir}/{algo_name}_seed{i}_rewards.npy
    每个文件是一个1D numpy数组, 代表每个episode/eval间隔的奖励
    
    Returns:
        data: dict {algo_name: np.array of shape (num_seeds, num_steps)}
    """
    data = {}
    for algo in algo_list:
        pattern = os.path.join(save_dir, f"{algo}_seed*_rewards.npy")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            print(f"[WARNING] No reward files found for {algo}: {pattern}")
            continue
        
        curves = []
        for f in files:
            curve = np.load(f)
            curves.append(curve)
        
        # 对齐长度 (取最短)
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        data[algo] = np.array(curves)  # (num_seeds, num_steps)
        print(f"[INFO] Loaded {algo}: {len(files)} seeds, {min_len} steps each")
    
    return data


def plot_journal_comparison(save_dir, algo_list, smooth_window=7, 
                            x_label='Training Episodes (×10³)',
                            y_label='Average Episode Reward',
                            title='Multi-Algorithm Training Comparison',
                            filename='comparison_reward',
                            x_scale=1000, show_markers=True,
                            marker_interval=None):
    """
    生成顶刊级别的训练对比图
    
    Args:
        save_dir: 数据和图片保存目录
        algo_list: 算法名称列表
        smooth_window: 平滑窗口大小
        x_label: X轴标签
        y_label: Y轴标签
        title: 图标题
        filename: 输出文件名 (不含扩展名)
        x_scale: X轴缩放因子
        show_markers: 是否显示数据点标记
        marker_interval: 标记间隔 (None=自动)
    """
    set_journal_style()
    
    data = load_reward_curves(save_dir, algo_list)
    if len(data) == 0:
        print("[ERROR] No data loaded, cannot generate plot.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    
    for algo in algo_list:
        if algo not in data:
            continue
        
        curves = data[algo]  # (num_seeds, num_steps)
        num_steps = curves.shape[1]
        
        # 计算统计量
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        
        # 平滑
        mean_smooth = smooth_curve(mean, window_size=smooth_window)
        std_smooth = smooth_curve(std, window_size=max(smooth_window, 5))
        
        # X轴
        x = np.arange(num_steps) / max(x_scale, 1)
        
        # 获取样式
        style = ALGO_STYLES.get(algo, {
            'color': EXTRA_COLORS[hash(algo) % len(EXTRA_COLORS)],
            'linestyle': '-',
            'marker': 'o',
            'label': algo,
            'zorder': 1,
        })
        
        # 绘制mean曲线
        if show_markers:
            if marker_interval is None:
                mi = max(1, num_steps // 8)
            else:
                mi = marker_interval
            ax.plot(x, mean_smooth, 
                    color=style['color'], 
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markevery=mi,
                    markersize=5,
                    label=style['label'],
                    zorder=style['zorder'])
        else:
            ax.plot(x, mean_smooth,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    label=style['label'],
                    zorder=style['zorder'])
        
        # 绘制std shading
        ax.fill_between(x, 
                        mean_smooth - std_smooth, 
                        mean_smooth + std_smooth,
                        color=style['color'], alpha=0.15, zorder=style['zorder']-1)
    
    # 标签和标题
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 图例
    ax.legend(loc='best', ncol=1, handlelength=2.5)
    
    # 调整边距
    ax.tick_params(axis='both', which='both', top=True, right=True)
    
    # 保存
    for ext in ['pdf', 'png']:
        out_path = os.path.join(save_dir, f"{filename}.{ext}")
        fig.savefig(out_path, format=ext, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {out_path}")
    
    plt.close(fig)
    print(f"[DONE] Comparison plot saved to {save_dir}/{filename}.[pdf|png]")


def plot_multi_metric(save_dir, algo_list, metrics=None, smooth_window=7):
    """
    绘制多指标子图 (reward + win_rate + value_loss 等)
    
    每个指标一个子图, 组合成一个大figure
    """
    if metrics is None:
        metrics = [
            {'name': 'rewards', 'ylabel': 'Average Reward', 'title': 'Episode Reward'},
        ]
    
    set_journal_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        metric_name = metric['name']
        
        for algo in algo_list:
            pattern = os.path.join(save_dir, f"{algo}_seed*_{metric_name}.npy")
            files = sorted(glob.glob(pattern))
            if len(files) == 0:
                continue
            
            curves = []
            for f in files:
                curves.append(np.load(f))
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            arr = np.array(curves)
            
            mean = smooth_curve(np.mean(arr, axis=0), window_size=smooth_window)
            std = smooth_curve(np.std(arr, axis=0), window_size=max(smooth_window, 5))
            x = np.arange(len(mean))
            
            style = ALGO_STYLES.get(algo, {
                'color': EXTRA_COLORS[hash(algo) % len(EXTRA_COLORS)],
                'linestyle': '-', 'label': algo, 'zorder': 1,
            })
            
            ax.plot(x, mean, color=style['color'], linestyle=style['linestyle'],
                    label=style['label'], zorder=style.get('zorder', 1))
            ax.fill_between(x, mean - std, mean + std,
                           color=style['color'], alpha=0.15)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.get('ylabel', metric_name))
        ax.set_title(metric.get('title', metric_name))
        ax.legend(loc='best')
        ax.tick_params(axis='both', which='both', top=True, right=True)
    
    fig.tight_layout()
    out_path = os.path.join(save_dir, 'multi_metric_comparison')
    fig.savefig(f"{out_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_path}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] Multi-metric comparison plot")


def plot_convergence_table(save_dir, algo_list):
    """
    生成收敛性能汇总表格图 (用于论文table替代)
    包含: 最终均值, 最终std, 收敛episode, 最大reward
    """
    set_journal_style()
    
    data = load_reward_curves(save_dir, algo_list)
    if len(data) == 0:
        return
    
    # 收集统计信息
    rows = []
    for algo in algo_list:
        if algo not in data:
            continue
        curves = data[algo]
        final_mean = np.mean(curves[:, -10:])  # 最后10步均值
        final_std = np.std(np.mean(curves[:, -10:], axis=1))
        max_reward = np.max(np.mean(curves, axis=0))
        
        # 收敛episode (超过最终均值90%的第一个点)
        threshold = final_mean * 0.9
        mean_curve = np.mean(curves, axis=0)
        converge_idx = np.where(mean_curve >= threshold)[0]
        converge_ep = converge_idx[0] if len(converge_idx) > 0 else len(mean_curve)
        
        rows.append([algo, f"{final_mean:.1f}", f"{final_std:.1f}", 
                     f"{converge_ep}", f"{max_reward:.1f}"])
    
    # 绘制表格
    fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(rows)))
    ax.axis('off')
    
    columns = ['Algorithm', 'Final Reward', 'Std', 'Conv. Episode', 'Max Reward']
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(columns))))
    
    # 美化表格
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')
    
    out_path = os.path.join(save_dir, 'convergence_table')
    fig.savefig(f"{out_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_path}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] Convergence table: {out_path}.[pdf|png]")


def plot_final_bar_chart(save_dir, algo_list):
    """
    绘制最终性能柱状图 (带误差棒)
    """
    set_journal_style()
    
    data = load_reward_curves(save_dir, algo_list)
    if len(data) == 0:
        return
    
    algos = []
    means = []
    stds = []
    colors = []
    
    for algo in algo_list:
        if algo not in data:
            continue
        curves = data[algo]
        final_rewards = np.mean(curves[:, -10:], axis=1)
        algos.append(ALGO_STYLES.get(algo, {}).get('label', algo))
        means.append(np.mean(final_rewards))
        stds.append(np.std(final_rewards))
        colors.append(ALGO_STYLES.get(algo, {}).get('color', '#333333'))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=0.8, alpha=0.85, width=0.6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha='right')
    ax.set_ylabel('Final Average Reward')
    ax.set_title('Final Performance Comparison')
    
    # 数值标注
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + s + 0.5,
                f'{m:.1f}', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    out_path = os.path.join(save_dir, 'final_performance_bar')
    fig.savefig(f"{out_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_path}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] Bar chart: {out_path}.[pdf|png]")


# ========== 主程序 ==========
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Publication-Quality Comparison Plotter")
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory with reward .npy files')
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL'])
    parser.add_argument('--smooth', type=int, default=7)
    parser.add_argument('--x_scale', type=int, default=1000)
    parser.add_argument('--no_markers', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Publication-Quality Comparison Plot Generator")
    print("=" * 60)
    
    # 1. 主对比图 (reward曲线)
    plot_journal_comparison(
        save_dir=args.save_dir,
        algo_list=args.algos,
        smooth_window=args.smooth,
        x_scale=args.x_scale,
        show_markers=not args.no_markers,
    )
    
    # 2. 最终性能柱状图
    plot_final_bar_chart(args.save_dir, args.algos)
    
    # 3. 收敛表格
    plot_convergence_table(args.save_dir, args.algos)
    
    print("\n[ALL DONE] All plots generated successfully!")
