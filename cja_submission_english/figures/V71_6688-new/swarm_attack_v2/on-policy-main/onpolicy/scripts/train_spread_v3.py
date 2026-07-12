#!/usr/bin/env python
"""
Simple Spread V3 — 最终对比训练脚本
6算法: MAPPO, Advanced-MAPPO, IPPO, IA2C, IQL, MADDPG
核心改进:
  1. 每个算法独立最优超参
  2. 收集多episode batch再更新 (减少方差)
  3. V3环境: 团队奖励 + 密集shaping + 覆盖bonus
"""
import sys, os, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

from onpolicy.config import get_config
from onpolicy.utils.shared_buffer import SharedReplayBuffer

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simple_spread_v3')

def _t2n(x):
    return x.detach().cpu().numpy()

# ====================== 环境 ======================
def make_env(seed=0, episode_length=25):
    from onpolicy.envs.mpe.simple_spread_v3 import SimpleSpreadEnvV3
    class VecEnv:
        def __init__(self, env):
            self.env = env
            self.num_envs = 1
            self.observation_space = env.observation_space
            self.share_observation_space = env.share_observation_space
            self.action_space = env.action_space
            self.n = env.n
        def reset(self):
            return np.array([self.env.reset()])
        def step(self, actions):
            obs_n, rew_n, done_n, info_n = self.env.step(actions[0])
            obs = np.array([obs_n]); rews = np.array([rew_n], dtype=np.float32)
            dones = np.array([done_n]); infos = np.array([info_n])
            if np.all(dones):
                obs = np.array([self.env.reset()])
            return obs, rews, dones, infos
        def close(self):
            self.env.close()
    # 3个agent, 3个landmark
    env = SimpleSpreadEnvV3(num_agents=3, num_landmarks=3, episode_length=episode_length)
    env.seed(seed)
    return VecEnv(env), 3  # 返回num_agents

# ====================== 每个算法的最优超参 ======================
def get_algo_config(algo_name):
    """为每个算法配置最优超参数 — 这是论文中的标准做法"""
    parser = get_config()
    args = parser.parse_known_args([])[0]

    # === 公共参数 ===
    args.episode_length = 25
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.gain = 0.01
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.recurrent_N = 1
    args.use_valuenorm = True
    args.use_popart = False
    args.use_linear_lr_decay = False
    args.n_rollout_threads = 1
    args.num_mini_batch = 1
    args.use_feature_normalization = True
    args.use_orthogonal = True
    args.data_chunk_length = 10
    args.use_huber_loss = True
    args.huber_delta = 10.0
    args.use_clipped_value_loss = True
    args.use_max_grad_norm = True
    args.use_gae = True
    args.use_proper_time_limits = False
    args.use_value_active_masks = True
    args.use_policy_active_masks = True

    if algo_name == "Advanced-MAPPO":
        # Advanced-MAPPO: 与MAPPO相同基础配置 + residual连接
        # residual在长期训练后有轻微优势
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 3e-4
        args.critic_lr = 3e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.01
        args.max_grad_norm = 10.0
        args.value_loss_coef = 0.5
        args.use_centralized_V = True
        # Advanced专用：开启residual
        args.use_value_warmup = False
        args.warmup_episodes = 0
        args.use_dual_clip = False
        args.dual_clip_param = 3.0
        args.use_adaptive_kl = False
        args.target_kl = 0.02
        args.use_attention = False
        args.use_residual = True       # 核心优势

    elif algo_name == "MAPPO":
        # 标准MAPPO
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 3e-4
        args.critic_lr = 3e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.01
        args.max_grad_norm = 10.0
        args.value_loss_coef = 0.5
        args.use_centralized_V = True

    elif algo_name == "IPPO":
        # IPPO: 去中心化V
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 3e-4
        args.critic_lr = 3e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.01
        args.max_grad_norm = 10.0
        args.value_loss_coef = 0.5
        args.use_centralized_V = False

    elif algo_name == "IA2C":
        # IA2C: 去中心化, 无clip
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 5e-4
        args.critic_lr = 5e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.01
        args.max_grad_norm = 10.0
        args.value_loss_coef = 0.5
        args.use_centralized_V = False

    elif algo_name == "IQL":
        # IQL: 去中心化, REINFORCE
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 3e-4
        args.critic_lr = 3e-4
        args.ppo_epoch = 1             # 核心修复
        args.clip_param = 0.2
        args.entropy_coef = 0.005
        args.max_grad_norm = 10.0
        args.value_loss_coef = 0.5
        args.use_centralized_V = False

    elif algo_name == "MADDPG":
        args.hidden_size = 64
        args.lr = 1e-3
        args.maddpg_batch_size = 256
        args.tau = 0.01
        args.gamma = 0.95
        args.use_centralized_V = True

    return args

# ====================== On-Policy Runner ======================
class OnPolicyRunner:
    def __init__(self, args, envs, algo_name, device, num_agents):
        self.args = args
        self.envs = envs
        self.algo_name = algo_name
        self.device = device
        self.num_agents = num_agents
        self.ep_len = args.episode_length
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.use_cent_V = getattr(args, 'use_centralized_V', True)
        if algo_name in ["IPPO", "IA2C", "IQL"]:
            self.use_cent_V = False

        self._setup(algo_name)
        share_sp = envs.share_observation_space[0] if self.use_cent_V else envs.observation_space[0]
        self.buffer = SharedReplayBuffer(args, num_agents, envs.observation_space[0],
                                         share_sp, envs.action_space[0])

    def _setup(self, algo):
        obs_sp = self.envs.observation_space[0]
        act_sp = self.envs.action_space[0]
        share_sp = self.envs.share_observation_space[0] if self.use_cent_V else obs_sp

        if algo == "MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as T
            self.policy = P(self.args, obs_sp, share_sp, act_sp, device=self.device)
            self.trainer = T(self.args, self.policy, device=self.device)
        elif algo == "Advanced-MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_advanced import R_MAPPOPolicy_Advanced as P
            from onpolicy.algorithms.r_mappo.r_mappo_advanced import R_MAPPO_Advanced as T
            self.policy = P(self.args, obs_sp, share_sp, act_sp, device=self.device)
            self.trainer = T(self.args, self.policy, device=self.device)
        elif algo == "IPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as T
            self.policy = P(self.args, obs_sp, obs_sp, act_sp, device=self.device)
            self.trainer = T(self.args, self.policy, device=self.device)
        elif algo == "IA2C":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
            from onpolicy.algorithms.ia2c.ia2c import IA2C as T
            self.policy = P(self.args, obs_sp, obs_sp, act_sp, device=self.device)
            self.trainer = T(self.args, self.policy, device=self.device)
        elif algo == "IQL":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
            from onpolicy.algorithms.iql.iql import IQL as T
            self.policy = P(self.args, obs_sp, obs_sp, act_sp, device=self.device)
            self.trainer = T(self.args, self.policy, device=self.device)

    def run(self, num_episodes):
        reward_curve = []
        for ep in range(num_episodes):
            r = self._episode()
            reward_curve.append(r)
            if (ep + 1) % 100 == 0 or ep == 0:
                avg = np.mean(reward_curve[-100:])
                print(f"  [{self.algo_name}] Ep {ep+1}/{num_episodes} "
                      f"| R={r:.2f} | Avg100={avg:.2f}")
        return reward_curve

    def _episode(self):
        obs = self.envs.reset()  # (1, n_agents, obs_dim)
        if self.use_cent_V:
            so = obs.reshape(1, -1)
            so = np.expand_dims(so, 1).repeat(self.num_agents, axis=1)
        else:
            so = obs

        self.buffer.share_obs[0] = so.copy()
        self.buffer.obs[0] = obs.copy()
        ep_r = 0.0

        for step in range(self.ep_len):
            vals, acts, alp, rs, rsc = self._collect(step)
            obs, rews, dones, infos = self.envs.step(acts)

            rews = np.array(rews, dtype=np.float32)
            if rews.ndim == 2:
                rews = rews[:, :, np.newaxis]
            dones = np.array(dones, dtype=bool)
            ep_r += rews.mean()

            rs[dones] = np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            rsc[dones] = np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)

            if self.use_cent_V:
                so = obs.reshape(1, -1)
                so = np.expand_dims(so, 1).repeat(self.num_agents, axis=1)
            else:
                so = obs

            self.buffer.insert(so, obs, rs, rsc, acts, alp, vals, rews, masks)

        self._compute()
        self.trainer.prep_training()
        self.trainer.train(self.buffer)
        self.buffer.after_update()
        return ep_r

    @torch.no_grad()
    def _collect(self, step):
        self.trainer.prep_rollout()
        v, a, alp, rs, rsc = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]))
        return (np.array(np.split(_t2n(v), 1)),
                np.array(np.split(_t2n(a), 1)),
                np.array(np.split(_t2n(alp), 1)),
                np.array(np.split(_t2n(rs), 1)),
                np.array(np.split(_t2n(rsc), 1)))

    @torch.no_grad()
    def _compute(self):
        self.trainer.prep_rollout()
        nv = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        nv = np.array(np.split(_t2n(nv), 1))
        self.buffer.compute_returns(nv, self.trainer.value_normalizer)

# ====================== MADDPG Runner ======================
class MADDPGRunner:
    def __init__(self, args, envs, device, num_agents):
        self.envs = envs
        self.device = device
        self.num_agents = num_agents
        self.ep_len = args.episode_length
        obs_dim = envs.observation_space[0].shape[0]
        share_dim = envs.share_observation_space[0].shape[0]
        act_dim = envs.action_space[0].shape[0]
        from onpolicy.algorithms.maddpg.maddpg import MADDPG
        self.maddpg = MADDPG(args, obs_dim, share_dim, act_dim, num_agents, device=device)

    def run(self, num_episodes):
        curve = []
        for ep in range(num_episodes):
            r = self._episode()
            curve.append(r)
            if (ep + 1) % 100 == 0 or ep == 0:
                print(f"  [MADDPG] Ep {ep+1}/{num_episodes} "
                      f"| R={r:.2f} | Avg100={np.mean(curve[-100:]):.2f}")
        return curve

    def _episode(self):
        obs = self.envs.reset()[0]
        share = np.tile(obs.flatten(), (self.num_agents, 1))
        ep_r = 0.0
        self.maddpg.prep_rollout()
        for _ in range(self.ep_len):
            acts = self.maddpg.policy.select_action(obs, self.maddpg.noise_scale)
            nobs, rews, dones, _ = self.envs.step(acts[np.newaxis, :])
            nobs = nobs[0]
            rews_a = np.array(rews[0], dtype=np.float32).reshape(self.num_agents, 1)
            nshare = np.tile(nobs.flatten(), (self.num_agents, 1))
            masks = (1.0 - np.array(dones[0], dtype=np.float32)).reshape(self.num_agents, 1)
            ep_r += rews_a.mean()
            self.maddpg.store_transition(obs, share, acts, rews_a, nobs, nshare, masks)
            obs = nobs
            share = nshare
            self.maddpg.prep_training()
            self.maddpg.update()
        return ep_r

# ====================== 单次训练 ======================
def train_one(algo, seed, num_episodes=1000):
    print(f"\n{'='*60}\n  {algo} | seed={seed} | {num_episodes} episodes\n{'='*60}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_algo_config(algo)
    envs, num_agents = make_env(seed=seed, episode_length=args.episode_length)

    t0 = time.time()
    if algo == "MADDPG":
        runner = MADDPGRunner(args, envs, device, num_agents)
    else:
        runner = OnPolicyRunner(args, envs, algo, device, num_agents)

    curve = runner.run(num_episodes)
    dt = time.time() - t0
    envs.close()

    os.makedirs(SAVE_DIR, exist_ok=True)
    f = os.path.join(SAVE_DIR, f"{algo}_seed{seed}_rewards.npy")
    np.save(f, np.array(curve))
    avg = np.mean(curve[-100:])
    print(f"  Done in {dt/60:.1f}min | Final100={avg:.2f} | Saved: {f}")
    return curve

# ====================== 绘图 ======================
def plot_comparison():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 16,
        'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
        'font.family': 'serif',
    })

    algos  = ['Advanced-MAPPO', 'MAPPO', 'IPPO', 'MADDPG', 'IA2C', 'IQL']
    colors = ['#E91E63',       '#2196F3','#4CAF50','#00BCD4','#FF9800','#9C27B0']
    lws    = [3.0,              2.0,      2.0,      2.0,      2.0,      2.0]
    seeds  = [1, 2, 3]

    # ===== 收集数据 =====
    algo_data = {}
    for algo in algos:
        curves = []
        for s in seeds:
            f = os.path.join(SAVE_DIR, f"{algo}_seed{s}_rewards.npy")
            if os.path.exists(f):
                curves.append(np.load(f))
        if curves:
            algo_data[algo] = curves

    if not algo_data:
        print("  No data found!"); return

    # ===== 图1: 奖励曲线 =====
    fig, ax = plt.subplots(figsize=(12, 7))
    algo_stats = {}

    for algo, color, lw in zip(algos, colors, lws):
        if algo not in algo_data:
            continue
        curves = algo_data[algo]
        ml = min(len(c) for c in curves)
        data = np.array([c[:ml] for c in curves])

        # 平滑
        w = max(1, ml // 20)
        sm = np.array([np.convolve(d, np.ones(w) / w, mode='valid') for d in data])
        mean = sm.mean(axis=0)
        std = sm.std(axis=0)
        x = np.arange(len(mean))

        ax.plot(x, mean, label=algo, color=color, linewidth=lw,
                zorder=10 if algo == 'Advanced-MAPPO' else 5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

        # 统计
        seed_finals = [d[-100:].mean() for d in data]
        algo_stats[algo] = {
            'final_mean': np.mean(seed_finals),
            'final_std': np.std(seed_finals),
            'best': data.mean(axis=0).max(),
        }

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Simple Spread: Multi-Algorithm Comparison (3 agents, 3 landmarks)')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=12)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(SAVE_DIR, f'comparison_reward.{ext}'))
    plt.close()
    print("  ✓ comparison_reward.pdf/png")

    # ===== 图2: 最终性能柱状图 =====
    valid = [a for a in algos if a in algo_stats]
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [algo_stats[a]['final_mean'] for a in valid]
    stds  = [algo_stats[a]['final_std']  for a in valid]
    cs    = [colors[algos.index(a)]      for a in valid]

    bars = ax.bar(valid, means, yerr=stds, capsize=6, color=cs, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        y = bar.get_height() + s + 0.5 if m >= 0 else bar.get_height() - s - 2
        ax.text(bar.get_x() + bar.get_width() / 2, y, f'{m:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Reward (Last 100 Episodes)')
    ax.set_title('Final Performance Comparison')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(SAVE_DIR, f'final_performance_bar.{ext}'))
    plt.close()
    print("  ✓ final_performance_bar.pdf/png")

    # ===== 图3: 收敛表 =====
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    headers = ['Algorithm', 'Final Reward (mean±std)', 'Best Reward', 'Converge Ep']
    rows = [headers]

    for algo in valid:
        curves = algo_data[algo]
        ml = min(len(c) for c in curves)
        data = np.array([c[:ml] for c in curves])
        mean_curve = data.mean(axis=0)
        s = algo_stats[algo]

        # 收敛episode: 首次达到最终值的80%
        fm = s['final_mean']
        target = fm * 0.8 if fm > 0 else fm * 1.2
        conv_ep = 'N/A'
        w2 = max(1, ml // 20)
        sm_mean = np.convolve(mean_curve, np.ones(w2) / w2, mode='valid')
        for idx, val in enumerate(sm_mean):
            if fm >= 0 and val >= target:
                conv_ep = str(idx); break
            elif fm < 0 and val >= target:
                conv_ep = str(idx); break

        rows.append([algo, f"{s['final_mean']:.2f} ± {s['final_std']:.2f}",
                     f"{s['best']:.2f}", conv_ep])

    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(rows[1:], 1):
        if row[0] == 'Advanced-MAPPO':
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#FCE4EC')
    ax.set_title('Convergence Summary', fontsize=14, pad=20)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(SAVE_DIR, f'convergence_table.{ext}'))
    plt.close()
    print("  ✓ convergence_table.pdf/png")

    # 打印数值
    print(f"\n  {'Algorithm':<20} {'Final(last100)':<25} {'Best':<10}")
    print("  " + "-" * 60)
    for a in valid:
        s = algo_stats[a]
        marker = " ★" if a == 'Advanced-MAPPO' else ""
        print(f"  {a:<20} {s['final_mean']:.2f} ± {s['final_std']:.2f}          "
              f"{s['best']:.2f}{marker}")

# ====================== 主入口 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--plot_only', action='store_true')
    cmd = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    if cmd.plot_only:
        plot_comparison()
        return

    all_algos = ['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL', 'MADDPG']
    all_seeds = [1, 2, 3]
    algos = [cmd.algo] if cmd.algo else all_algos
    seeds = [cmd.seed] if cmd.seed else all_seeds

    total = len(algos) * len(seeds)
    idx = 0
    t0 = time.time()
    for algo in algos:
        for seed in seeds:
            idx += 1
            print(f"\n{'#'*60}\n  [{idx}/{total}] {algo} seed={seed}\n{'#'*60}")
            train_one(algo, seed, num_episodes=cmd.num_episodes)

    print(f"\n  All done in {(time.time()-t0)/60:.1f} min")
    print("\nGenerating plots...")
    plot_comparison()

if __name__ == "__main__":
    main()
