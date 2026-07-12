#!/usr/bin/env python
"""
多算法对比训练脚本
支持: MAPPO, Advanced-MAPPO, IPPO, IA2C, IQL(Independent-Q) 等5种算法对比
统一使用现有的on-policy框架，通过修改base_runner中的算法选择来实现
所有算法共用同一个环境，保存各自的训练曲线用于最终对比
"""
import sys
import os
import numpy as np
from pathlib import Path
import torch
import time
import json
import argparse

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def train_single_algorithm(algo_name, all_args, seed=1):
    """训练单个算法并返回训练奖励曲线"""
    print(f"\n{'='*70}")
    print(f"  开始训练算法: {algo_name}  |  seed={seed}")
    print(f"{'='*70}\n")

    # 设置随机种子
    all_args.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 训练时不加载预训练模型
    all_args.model_dir = None
    all_args.use_render = False
    all_args.use_wandb = False

    # 设备
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 运行目录
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "comparison" / algo_name / f"seed{seed}"
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 创建环境
    envs = make_train_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # 根据算法名选择不同的Runner
    from onpolicy.runner.shared.base_runner_multi import MultiAlgoRunner
    runner = MultiAlgoRunner(config, algo_name)

    # 开始训练
    reward_curve = runner.run()

    envs.close()

    # 保存奖励曲线
    curve_file = str(run_dir / "reward_curve.npy")
    np.save(curve_file, np.array(reward_curve))
    print(f"\n[{algo_name}] 训练完成, 奖励曲线已保存到 {curve_file}")

    return reward_curve


def get_train_args():
    """获取训练参数"""
    parser = get_config()
    parser.add_argument('--scenario_name', type=str, default='simple_world_comm')
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=20)
    # 对比训练专用参数
    parser.add_argument('--compare_steps', type=int, default=3000000,
                        help="对比训练的总步数")
    parser.add_argument('--num_seeds', type=int, default=3,
                        help="每个算法跑几个seed")

    all_args = parser.parse_known_args(sys.argv[1:])[0]
    return all_args


def main():
    all_args = get_train_args()

    # 覆盖一些关键参数用于对比训练
    all_args.num_env_steps = all_args.compare_steps
    all_args.n_rollout_threads = 1
    all_args.episode_length = 1500
    all_args.hidden_size = 1024
    all_args.layer_N = 2
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.algorithm_name = "rmappo"
    all_args.use_valuenorm = True
    all_args.use_popart = False
    all_args.ppo_epoch = 10
    all_args.clip_param = 0.1
    all_args.lr = 5e-4
    all_args.critic_lr = 5e-4
    all_args.entropy_coef = 0.01
    all_args.max_grad_norm = 0.3
    all_args.value_loss_coef = 0.3
    all_args.huber_delta = 15.0
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95
    all_args.gain = 0.001
    all_args.log_interval = 1
    all_args.save_interval = 50
    all_args.model_dir = None
    all_args.use_render = False
    all_args.use_wandb = False
    all_args.use_linear_lr_decay = True
    all_args.use_centralized_V = True

    # 需要对比的算法列表
    algo_list = [
        "MAPPO",
        "Advanced-MAPPO",
        "IPPO",
        "IA2C",
        "IQL"
    ]

    num_seeds = all_args.num_seeds
    all_curves = {}  # {algo_name: [curve_seed1, curve_seed2, ...]}

    for algo_name in algo_list:
        all_curves[algo_name] = []
        for seed in range(1, num_seeds + 1):
            curve = train_single_algorithm(algo_name, all_args, seed=seed)
            all_curves[algo_name].append(curve)

    # 保存所有曲线 (格式与plot_comparison.py一致: {algo}_seed{i}_rewards.npy)
    save_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "comparison"
    os.makedirs(str(save_dir), exist_ok=True)

    for algo_name, curves in all_curves.items():
        for seed_idx, curve in enumerate(curves):
            np.save(str(save_dir / f"{algo_name}_seed{seed_idx+1}_rewards.npy"), np.array(curve))
        # 也保存合并版本
        min_len = min(len(c) for c in curves)
        curves_arr = np.array([c[:min_len] for c in curves])
        np.save(str(save_dir / f"{algo_name}_curves.npy"), curves_arr)

    # 生成对比图
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_comparison import plot_journal_comparison, plot_final_bar_chart, plot_convergence_table
    plot_journal_comparison(str(save_dir), algo_list)
    plot_final_bar_chart(str(save_dir), algo_list)
    plot_convergence_table(str(save_dir), algo_list)

    print(f"\n{'='*70}")
    print("  所有算法训练完成！对比图已生成。")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
