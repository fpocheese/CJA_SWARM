#!/usr/bin/env python
"""
单算法训练脚本 - 用于并行训练
用法: python train_single_algo.py --algo MAPPO --seed 1 --compare_steps 150000
"""
import sys
import os
import numpy as np
from pathlib import Path
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        from onpolicy.envs.env_wrappers import SubprocVecEnv
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def main():
    parser = get_config()
    parser.add_argument('--scenario_name', type=str, default='simple_world_comm')
    parser.add_argument('--num_landmarks', type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=20)
    parser.add_argument('--algo', type=str, required=True,
                        choices=['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL'])
    parser.add_argument('--compare_steps', type=int, default=150000)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for sensitivity/result curves.')
    parser.add_argument('--sensitivity_tag', type=str, default='baseline',
                        help='Label printed for reward or hyperparameter sensitivity runs.')
    parser.add_argument('--case_3d', type=str, default='case1', choices=['case1', 'case2'],
                        help='3D engagement case used by simple_world_comm_3d.')
    parser.add_argument('--hit_radius_3d', type=float, default=20.0,
                        help='Hit radius for the 3D cooperative interception scenario.')

    parser.set_defaults(
        model_dir=None,
        use_render=False,
        use_wandb=False,
        n_rollout_threads=1,
        episode_length=1500,
        hidden_size=256,
        layer_N=1,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        algorithm_name="rmappo",
        use_valuenorm=True,
        use_popart=False,
        ppo_epoch=10,
        clip_param=0.1,
        lr=5e-4,
        critic_lr=5e-4,
        entropy_coef=0.01,
        max_grad_norm=0.3,
        value_loss_coef=0.3,
        huber_delta=15.0,
        gamma=0.99,
        gae_lambda=0.95,
        gain=0.001,
        log_interval=5,
        save_interval=100,
        use_linear_lr_decay=True,
        use_centralized_V=True,
    )

    all_args = parser.parse_known_args(sys.argv[1:])[0]

    algo_name = all_args.algo
    seed = all_args.seed if all_args.seed != 0 else 1

    # 覆盖运行态参数；训练超参数默认值已通过 parser.set_defaults 设置，命令行可覆盖。
    all_args.seed = seed
    all_args.num_env_steps = all_args.compare_steps

    print(f"\n{'='*60}")
    print(f"  算法: {algo_name} | 种子: {seed} | 步数: {all_args.compare_steps} | 标签: {all_args.sensitivity_tag}")
    print(f"  Scenario: {all_args.scenario_name} | case_3d={getattr(all_args, 'case_3d', 'n/a')}")
    print(f"  Reward weights: dist={all_args.reward_w_dist}, angle={all_args.reward_w_angle}, "
          f"hit={all_args.reward_w_hit}, coord={all_args.reward_w_coord}, energy={all_args.reward_w_energy}")
    print(f"  Hyperparams: clip={all_args.clip_param}, entropy={all_args.entropy_coef}, "
          f"gae={all_args.gae_lambda}, target_kl={getattr(all_args, 'target_kl', None)}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 统一保存目录
    if all_args.save_dir:
        save_base = Path(all_args.save_dir)
    else:
        save_base = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "comparison"
    run_dir = save_base / algo_name / f"seed{seed}"
    os.makedirs(str(run_dir), exist_ok=True)

    envs = make_train_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from onpolicy.runner.shared.base_runner_multi import MultiAlgoRunner
    runner = MultiAlgoRunner(config, algo_name)

    t_start = time.time()
    reward_curve = runner.run()
    t_end = time.time()

    envs.close()

    # 保存奖励曲线 (统一格式)
    os.makedirs(str(save_base), exist_ok=True)
    curve_file = str(save_base / f"{algo_name}_seed{seed}_rewards.npy")
    np.save(curve_file, np.array(reward_curve))

    print(f"\n{'='*60}")
    print(f"  {algo_name} seed{seed} 完成!")
    print(f"  耗时: {(t_end-t_start)/60:.1f} 分钟")
    print(f"  Episodes: {len(reward_curve)}")
    print(f"  最终奖励: {reward_curve[-1]:.2f}")
    print(f"  曲线保存: {curve_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
