#!/usr/bin/env python
"""
性能对比脚本：原版MAPPO vs 改进版MAPPO
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_logs(log_dir):
    """加载训练日志"""
    log_file = Path(log_dir) / "summary.json"
    if log_file.exists():
        with open(log_file, 'r') as f:
            data = json.load(f)
        return data
    return None


def compare_performance(original_dir, advanced_dir, save_dir="./comparison_results"):
    """
    对比原版和改进版的性能
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MAPPO算法性能对比分析")
    print("="*80)
    
    # 加载数据
    print("\n加载训练日志...")
    original_data = load_training_logs(original_dir)
    advanced_data = load_training_logs(advanced_dir)
    
    if original_data is None or advanced_data is None:
        print("错误: 无法加载训练日志")
        return
    
    # 提取关键指标
    metrics = {
        'average_episode_rewards': '平均回报',
        'policy_loss': '策略损失',
        'value_loss': '价值损失',
        'dist_entropy': '熵',
    }
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # 绘制原版数据
        if metric_key in original_data:
            original_values = np.array(original_data[metric_key])
            ax.plot(original_values, label='原版MAPPO', alpha=0.7, linewidth=2)
        
        # 绘制改进版数据
        if metric_key in advanced_data:
            advanced_values = np.array(advanced_data[metric_key])
            ax.plot(advanced_values, label='改进版MAPPO', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('训练步数', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name}对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n对比图表已保存到: {save_dir}/performance_comparison.png")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("性能统计对比")
    print("="*80)
    
    comparison_results = {}
    
    for metric_key, metric_name in metrics.items():
        if metric_key in original_data and metric_key in advanced_data:
            original_values = np.array(original_data[metric_key])
            advanced_values = np.array(advanced_data[metric_key])
            
            # 计算最后10%的平均值（稳定性能）
            window = max(1, len(original_values) // 10)
            original_final = np.mean(original_values[-window:])
            advanced_final = np.mean(advanced_values[-window:])
            
            improvement = ((advanced_final - original_final) / abs(original_final)) * 100
            
            comparison_results[metric_key] = {
                'original': original_final,
                'advanced': advanced_final,
                'improvement': improvement
            }
            
            print(f"\n{metric_name}:")
            print(f"  原版MAPPO:    {original_final:.4f}")
            print(f"  改进版MAPPO:  {advanced_final:.4f}")
            print(f"  提升幅度:     {improvement:+.2f}%")
    
    # 收敛速度分析
    print("\n" + "="*80)
    print("收敛速度分析")
    print("="*80)
    
    if 'average_episode_rewards' in original_data and 'average_episode_rewards' in advanced_data:
        original_rewards = np.array(original_data['average_episode_rewards'])
        advanced_rewards = np.array(advanced_data['average_episode_rewards'])
        
        # 计算达到某个阈值所需的步数
        threshold = np.mean(original_rewards[-len(original_rewards)//10:]) * 0.8
        
        original_convergence = np.argmax(original_rewards > threshold) if np.any(original_rewards > threshold) else len(original_rewards)
        advanced_convergence = np.argmax(advanced_rewards > threshold) if np.any(advanced_rewards > threshold) else len(advanced_rewards)
        
        speedup = ((original_convergence - advanced_convergence) / original_convergence) * 100
        
        print(f"\n达到80%最终性能所需步数:")
        print(f"  原版MAPPO:    {original_convergence}")
        print(f"  改进版MAPPO:  {advanced_convergence}")
        print(f"  加速比例:     {speedup:+.2f}%")
        
        comparison_results['convergence_speed'] = {
            'original': original_convergence,
            'advanced': advanced_convergence,
            'speedup': speedup
        }
    
    # 保存对比结果
    results_file = f"{save_dir}/comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\n详细对比结果已保存到: {results_file}")
    
    # 生成总结报告
    generate_summary_report(comparison_results, save_dir)
    
    print("\n" + "="*80)
    print("对比分析完成!")
    print("="*80 + "\n")


def generate_summary_report(results, save_dir):
    """生成总结报告"""
    report_file = f"{save_dir}/summary_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MAPPO算法改进效果总结报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("【主要改进技术】\n")
        f.write("1. 多头注意力机制 (Multi-Head Attention)\n")
        f.write("2. 残差连接 (Residual Connections)\n")
        f.write("3. Dual-clip PPO\n")
        f.write("4. 自适应KL惩罚\n")
        f.write("5. 价值函数热身\n\n")
        
        f.write("【性能提升】\n")
        for key, value in results.items():
            if key == 'convergence_speed':
                f.write(f"收敛速度提升: {value['speedup']:+.2f}%\n")
            elif key == 'average_episode_rewards':
                f.write(f"最终回报提升: {value['improvement']:+.2f}%\n")
        
        f.write("\n【建议】\n")
        if results.get('average_episode_rewards', {}).get('improvement', 0) > 10:
            f.write("✓ 改进版算法显著优于原版，建议使用改进版\n")
        elif results.get('average_episode_rewards', {}).get('improvement', 0) > 0:
            f.write("✓ 改进版算法有一定提升，可以考虑使用\n")
        else:
            f.write("× 改进版在当前任务上效果不明显，可能需要调整超参数\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"总结报告已保存到: {report_file}")


def plot_training_curves(log_dir, save_dir, title="训练曲线"):
    """
    绘制单个实验的训练曲线
    """
    data = load_training_logs(log_dir)
    if data is None:
        print(f"无法加载日志: {log_dir}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['average_episode_rewards', 'policy_loss', 'value_loss', 'dist_entropy']
    labels = ['平均回报', '策略损失', '价值损失', '熵']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        if metric in data:
            values = np.array(data[metric])
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_xlabel('训练步数', fontsize=12)
            axes[idx].set_ylabel(label, fontsize=12)
            axes[idx].set_title(f'{label}', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_dir}/training_curves.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="对比原版和改进版MAPPO的性能")
    parser.add_argument("--original_dir", type=str, required=True,
                        help="原版MAPPO的日志目录")
    parser.add_argument("--advanced_dir", type=str, required=True,
                        help="改进版MAPPO的日志目录")
    parser.add_argument("--save_dir", type=str, default="./comparison_results",
                        help="保存对比结果的目录")
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.original_dir):
        print(f"错误: 原版日志目录不存在: {args.original_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.advanced_dir):
        print(f"错误: 改进版日志目录不存在: {args.advanced_dir}")
        sys.exit(1)
    
    # 执行对比
    compare_performance(args.original_dir, args.advanced_dir, args.save_dir)
    
    print("\n使用示例:")
    print("python compare_algorithms.py \\")
    print("  --original_dir ./results/MPE/simple_world_comm/rmappo/exp1 \\")
    print("  --advanced_dir ./results/MPE/simple_world_comm/advanced_mappo/exp1 \\")
    print("  --save_dir ./comparison_results")
