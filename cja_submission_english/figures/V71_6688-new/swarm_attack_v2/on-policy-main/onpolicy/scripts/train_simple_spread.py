#!/usr/bin/env python
"""
Simple Spread 多算法对比训练脚本
支持: MAPPO, Advanced-MAPPO, IPPO, IA2C, IQL, MADDPG
环境: simple_spread (3 agents, 3 landmarks, 连续动作空间)

用法:
  python train_simple_spread.py                      # 训练所有算法
  python train_simple_spread.py --algo MADDPG         # 只训练MADDPG
  python train_simple_spread.py --algo MAPPO --seed 1 # 指定算法和种子
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy

# 确保可以导入onpolicy包
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

from onpolicy.config import get_config
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _t2n(x):
    return x.detach().cpu().numpy()


# ====================== 环境创建 ======================
def make_simple_spread_env(num_agents=3, num_landmarks=3, episode_length=25, seed=0):
    """创建SimpleSpreadEnv并包装成DummyVecEnv兼容格式"""
    from onpolicy.envs.mpe.simple_spread_env import SimpleSpreadEnv

    class DummyVecEnvSimple:
        """极简的VecEnv包装器（单环境）"""
        def __init__(self, env):
            self.env = env
            self.num_envs = 1
            self.observation_space = env.observation_space
            self.share_observation_space = env.share_observation_space
            self.action_space = env.action_space
            self.n = env.n

        def reset(self):
            obs_n = self.env.reset()
            return np.array([obs_n])  # (1, n_agents, obs_dim)

        def step(self, actions):
            """actions: (1, n_agents, act_dim)"""
            action_n = actions[0]  # 取出第一个env的动作
            obs_n, rew_n, done_n, info_n = self.env.step(action_n)
            obs = np.array([obs_n])
            rews = np.array([rew_n], dtype=np.float32)
            dones = np.array([done_n])
            infos = np.array([info_n])
            # 自动reset
            if np.all(dones):
                obs = np.array([self.env.reset()])
            return obs, rews, dones, infos

        def close(self):
            self.env.close()

    env = SimpleSpreadEnv(num_agents=num_agents, num_landmarks=num_landmarks,
                          episode_length=episode_length)
    env.seed(seed)
    return DummyVecEnvSimple(env)


# ====================== On-Policy Runner ======================
class OnPolicyRunner:
    """通用on-policy训练Runner (MAPPO/Advanced-MAPPO/IPPO/IA2C/IQL)"""
    def __init__(self, all_args, envs, algo_name, device, num_agents):
        self.all_args = all_args
        self.envs = envs
        self.algo_name = algo_name
        self.device = device
        self.num_agents = num_agents
        self.episode_length = all_args.episode_length
        self.n_rollout_threads = 1
        self.hidden_size = all_args.hidden_size
        self.recurrent_N = all_args.recurrent_N
        self.use_centralized_V = all_args.use_centralized_V

        # 根据算法决定是否使用中心化V
        if algo_name in ["IPPO", "IA2C", "IQL"]:
            self.use_centralized_V = False

        # 创建policy和trainer
        self._setup_algorithm(algo_name)

        # 创建buffer
        share_obs_space = envs.share_observation_space[0] if self.use_centralized_V else envs.observation_space[0]
        self.buffer = SharedReplayBuffer(
            all_args, num_agents,
            envs.observation_space[0], share_obs_space, envs.action_space[0]
        )

    def _setup_algorithm(self, algo_name):
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]

        if self.use_centralized_V:
            share_obs_space = self.envs.share_observation_space[0]
        else:
            share_obs_space = obs_space

        if algo_name == "MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, share_obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        elif algo_name == "Advanced-MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_advanced import R_MAPPOPolicy_Advanced as Policy
            from onpolicy.algorithms.r_mappo.r_mappo_advanced import R_MAPPO_Advanced as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, share_obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        elif algo_name == "IPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        elif algo_name == "IA2C":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.ia2c.ia2c import IA2C as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        elif algo_name == "IQL":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.iql.iql import IQL as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        else:
            raise ValueError(f"Unknown algo: {algo_name}")

    def run(self, num_episodes):
        """训练指定episode数量，返回奖励曲线"""
        reward_curve = []
        for episode in range(num_episodes):
            episode_reward = self._run_one_episode()
            reward_curve.append(episode_reward)
            if (episode + 1) % 50 == 0 or episode == 0:
                avg_last50 = np.mean(reward_curve[-50:])
                print(f"  [{self.algo_name}] Episode {episode+1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | Avg50: {avg_last50:.2f}")
        return reward_curve

    def _run_one_episode(self):
        """运行一个episode并更新"""
        # warmup
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(1, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        episode_reward = 0.0

        for step in range(self.episode_length):
            # collect
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self._collect(step)
            obs, rewards, dones, infos = self.envs.step(actions)

            # process rewards
            rewards = np.array(rewards, dtype=np.float32)
            if rewards.ndim == 2:
                rewards = rewards[:, :, np.newaxis]
            dones = np.array(dones, dtype=bool)

            episode_reward += rewards.mean()

            # insert
            rnn_states[dones == True] = np.zeros(
                ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic[dones == True] = np.zeros(
                ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            if self.use_centralized_V:
                share_obs = obs.reshape(1, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                share_obs = obs

            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                              actions, action_log_probs, values, rewards, masks)

        # compute returns and train
        self._compute()
        self.trainer.prep_training()
        self.trainer.train(self.buffer)
        self.buffer.after_update()

        return episode_reward

    @torch.no_grad()
    def _collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]))
        values = np.array(np.split(_t2n(value), 1))
        actions = np.array(np.split(_t2n(action), 1))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), 1))
        rnn_states = np.array(np.split(_t2n(rnn_states), 1))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 1))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def _compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), 1))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)


# ====================== MADDPG Runner ======================
class MADDPGRunner:
    """MADDPG off-policy训练Runner"""
    def __init__(self, all_args, envs, device, num_agents):
        self.envs = envs
        self.device = device
        self.num_agents = num_agents
        self.episode_length = all_args.episode_length
        self.algo_name = "MADDPG"

        obs_dim = envs.observation_space[0].shape[0]
        share_obs_dim = envs.share_observation_space[0].shape[0]
        act_dim = envs.action_space[0].shape[0]

        from onpolicy.algorithms.maddpg.maddpg import MADDPG
        self.maddpg = MADDPG(all_args, obs_dim, share_obs_dim, act_dim,
                             num_agents, device=device)

    def run(self, num_episodes):
        """训练指定episode数量"""
        reward_curve = []
        # 先收集一些经验
        for ep in range(num_episodes):
            ep_reward = self._run_one_episode()
            reward_curve.append(ep_reward)
            if (ep + 1) % 50 == 0 or ep == 0:
                avg_last50 = np.mean(reward_curve[-50:])
                print(f"  [MADDPG] Episode {ep+1}/{num_episodes} | "
                      f"Reward: {ep_reward:.2f} | Avg50: {avg_last50:.2f}")
        return reward_curve

    def _run_one_episode(self):
        obs_list = self.envs.reset()  # (1, n_agents, obs_dim)
        obs = obs_list[0]  # (n_agents, obs_dim)

        # share_obs = concat all obs
        share_obs = obs.flatten()  # (n_agents * obs_dim)
        share_obs_all = np.tile(share_obs, (self.num_agents, 1))  # (n_agents, share_obs_dim)

        episode_reward = 0.0
        self.maddpg.prep_rollout()

        for step in range(self.episode_length):
            # 选择动作
            actions = self.maddpg.policy.select_action(obs, self.maddpg.noise_scale)
            # (n_agents, act_dim)

            # 环境step
            actions_env = actions[np.newaxis, :]  # (1, n_agents, act_dim)
            next_obs_list, rew_list, done_list, info_list = self.envs.step(actions_env)

            next_obs = next_obs_list[0]  # (n_agents, obs_dim)
            rews = np.array(rew_list[0], dtype=np.float32).reshape(self.num_agents, 1)
            dones = np.array(done_list[0])

            next_share_obs = next_obs.flatten()
            next_share_obs_all = np.tile(next_share_obs, (self.num_agents, 1))

            masks = (1.0 - dones.astype(np.float32)).reshape(self.num_agents, 1)

            episode_reward += rews.mean()

            # 存入replay buffer
            self.maddpg.store_transition(obs, share_obs_all, actions, rews,
                                          next_obs, next_share_obs_all, masks)

            obs = next_obs
            share_obs_all = next_share_obs_all

            # 更新 (每步更新一次)
            self.maddpg.prep_training()
            self.maddpg.update()

        return episode_reward


# ====================== 主训练函数 ======================
def train_one(algo_name, seed, num_episodes=1000, num_agents=3, num_landmarks=3,
              episode_length=25, save_dir=None):
    """训练单个算法"""
    print(f"\n{'='*60}")
    print(f"  算法: {algo_name} | 种子: {seed} | Episodes: {num_episodes}")
    print(f"  环境: simple_spread | Agents: {num_agents} | Landmarks: {num_landmarks}")
    print(f"{'='*60}")

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建环境
    envs = make_simple_spread_env(num_agents=num_agents, num_landmarks=num_landmarks,
                                   episode_length=episode_length, seed=seed)

    # 构造args (复用config.py的默认值)
    parser = get_config()
    all_args = parser.parse_known_args([])[0]

    # 覆盖关键超参数（针对simple_spread短episode环境调优）
    all_args.episode_length = episode_length
    all_args.hidden_size = 64     # simple_spread用小网络就够
    all_args.layer_N = 1
    all_args.lr = 3e-4            # 适中的学习率，防止PPO过度更新
    all_args.critic_lr = 3e-4
    all_args.ppo_epoch = 5        # 减少PPO epoch次数，防止过拟合
    all_args.clip_param = 0.2
    all_args.entropy_coef = 0.01
    all_args.max_grad_norm = 10.0  # 放宽梯度裁剪
    all_args.value_loss_coef = 0.5
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95
    all_args.gain = 0.01
    all_args.use_centralized_V = True
    all_args.use_recurrent_policy = True    # 保持RNN以兼容框架
    all_args.use_naive_recurrent_policy = False
    all_args.recurrent_N = 1
    all_args.use_valuenorm = True
    all_args.use_popart = False
    all_args.use_linear_lr_decay = False    # 不衰减学习率
    all_args.n_rollout_threads = 1
    all_args.num_mini_batch = 1
    all_args.huber_delta = 10.0
    all_args.use_feature_normalization = True
    all_args.use_orthogonal = True
    all_args.data_chunk_length = 10
    all_args.use_huber_loss = True
    all_args.use_clipped_value_loss = True
    all_args.use_max_grad_norm = True
    all_args.use_gae = True
    all_args.use_proper_time_limits = False
    all_args.use_value_active_masks = True
    all_args.use_policy_active_masks = True

    # MADDPG专用参数
    all_args.maddpg_batch_size = 256
    all_args.tau = 0.01

    t_start = time.time()

    if algo_name == "MADDPG":
        runner = MADDPGRunner(all_args, envs, device, num_agents)
    else:
        runner = OnPolicyRunner(all_args, envs, algo_name, device, num_agents)

    reward_curve = runner.run(num_episodes)

    t_end = time.time()
    envs.close()

    # 保存奖励曲线
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simple_spread')
    os.makedirs(save_dir, exist_ok=True)
    curve_file = os.path.join(save_dir, f"{algo_name}_seed{seed}_rewards.npy")
    np.save(curve_file, np.array(reward_curve))

    print(f"\n  {algo_name} seed{seed} 完成! 耗时: {(t_end-t_start)/60:.1f}分钟")
    print(f"  最终50ep平均奖励: {np.mean(reward_curve[-50:]):.2f}")
    print(f"  保存: {curve_file}\n")

    return reward_curve


# ====================== 绘图函数 ======================
def plot_comparison(save_dir):
    """生成对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    algos = ['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL', 'MADDPG']
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    seeds = [1, 2, 3]

    # ===== 图1: 奖励曲线对比 =====
    fig, ax = plt.subplots(figsize=(12, 7))
    algo_final = {}

    for algo, color in zip(algos, colors):
        all_curves = []
        for seed in seeds:
            fpath = os.path.join(save_dir, f"{algo}_seed{seed}_rewards.npy")
            if os.path.exists(fpath):
                curve = np.load(fpath)
                all_curves.append(curve)

        if not all_curves:
            print(f"  [警告] 未找到 {algo} 的数据，跳过")
            continue

        # 对齐长度
        min_len = min(len(c) for c in all_curves)
        all_curves = [c[:min_len] for c in all_curves]
        data = np.array(all_curves)

        # 平滑
        window = min(20, min_len // 5) if min_len > 20 else 1
        smoothed = np.array([np.convolve(d, np.ones(window)/window, mode='valid') for d in data])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean))

        ax.plot(x, mean, label=algo, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        algo_final[algo] = (data[:, -50:].mean(), data[:, -50:].std())

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Simple Spread: Multi-Algorithm Comparison')
    ax.legend(loc='lower right', framealpha=0.9)

    fig.savefig(os.path.join(save_dir, 'comparison_reward.pdf'))
    fig.savefig(os.path.join(save_dir, 'comparison_reward.png'))
    plt.close()
    print(f"  ✓ 奖励曲线图保存完成")

    # ===== 图2: 最终性能柱状图 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_algos = [a for a in algos if a in algo_final]
    means = [algo_final[a][0] for a in valid_algos]
    stds = [algo_final[a][1] for a in valid_algos]
    valid_colors = [colors[algos.index(a)] for a in valid_algos]

    bars = ax.bar(valid_algos, means, yerr=stds, capsize=6, color=valid_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.2,
                f'{m:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Average Reward (Last 50 Episodes)')
    ax.set_title('Final Performance Comparison')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.savefig(os.path.join(save_dir, 'final_performance_bar.pdf'))
    fig.savefig(os.path.join(save_dir, 'final_performance_bar.png'))
    plt.close()
    print(f"  ✓ 最终性能柱状图保存完成")

    # ===== 图3: 收敛性能表 =====
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [['Algorithm', 'Final Reward', 'Best Reward', 'Convergence Episode']]

    for algo in valid_algos:
        all_curves = []
        for seed in seeds:
            fpath = os.path.join(save_dir, f"{algo}_seed{seed}_rewards.npy")
            if os.path.exists(fpath):
                all_curves.append(np.load(fpath))

        if all_curves:
            min_len = min(len(c) for c in all_curves)
            all_curves = [c[:min_len] for c in all_curves]
            data = np.array(all_curves)
            mean_curve = data.mean(axis=0)

            final_rew = f"{mean_curve[-50:].mean():.2f} ± {data[:,-50:].std():.2f}"
            best_rew = f"{mean_curve.max():.2f}"

            # 收敛episode: 首次达到最终奖励90%的位置
            target = mean_curve[-50:].mean() * 0.9
            conv_ep = "N/A"
            for idx, val in enumerate(mean_curve):
                if val >= target and target < 0:  # reward是负的，val >= target 意味着已经好于target
                    # 对负值reward,  "好" 意味着 less negative
                    pass
                if target < 0:
                    if val >= target:
                        conv_ep = str(idx)
                        break
                else:
                    if val >= target:
                        conv_ep = str(idx)
                        break

            table_data.append([algo, final_rew, best_rew, conv_ep])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 表头颜色
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Convergence Performance Summary', fontsize=14, pad=20)

    fig.savefig(os.path.join(save_dir, 'convergence_table.pdf'))
    fig.savefig(os.path.join(save_dir, 'convergence_table.png'))
    plt.close()
    print(f"  ✓ 收敛表保存完成")

    print(f"\n  所有图表保存在: {save_dir}/")


# ====================== 主入口 ======================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default=None,
                        help='只训练指定算法 (MAPPO/Advanced-MAPPO/IPPO/IA2C/IQL/MADDPG)')
    parser.add_argument('--seed', type=int, default=None, help='指定随机种子')
    parser.add_argument('--num_episodes', type=int, default=1000, help='训练episode数')
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--num_landmarks', type=int, default=3)
    parser.add_argument('--episode_length', type=int, default=25)
    parser.add_argument('--plot_only', action='store_true', help='只生成图表')
    cmd_args = parser.parse_args()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simple_spread')
    os.makedirs(save_dir, exist_ok=True)

    if cmd_args.plot_only:
        plot_comparison(save_dir)
        return

    all_algos = ['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL', 'MADDPG']
    all_seeds = [1, 2, 3]

    if cmd_args.algo:
        algos_to_run = [cmd_args.algo]
    else:
        algos_to_run = all_algos

    if cmd_args.seed is not None:
        seeds_to_run = [cmd_args.seed]
    else:
        seeds_to_run = all_seeds

    total_runs = len(algos_to_run) * len(seeds_to_run)
    run_idx = 0
    t_total_start = time.time()

    for algo in algos_to_run:
        for seed in seeds_to_run:
            run_idx += 1
            print(f"\n{'#'*60}")
            print(f"  [{run_idx}/{total_runs}] {algo} / seed {seed}")
            print(f"{'#'*60}")
            train_one(algo, seed, num_episodes=cmd_args.num_episodes,
                     num_agents=cmd_args.num_agents, num_landmarks=cmd_args.num_landmarks,
                     episode_length=cmd_args.episode_length, save_dir=save_dir)

    t_total_end = time.time()
    print(f"\n{'='*60}")
    print(f"  全部训练完成! 总耗时: {(t_total_end-t_total_start)/60:.1f}分钟")
    print(f"{'='*60}")

    # 生成对比图
    print("\n正在生成对比图表...")
    plot_comparison(save_dir)


if __name__ == "__main__":
    main()
