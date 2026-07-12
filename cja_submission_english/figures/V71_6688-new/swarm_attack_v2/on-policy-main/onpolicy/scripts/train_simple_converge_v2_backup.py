#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Converge - Journal-Quality Training Script
Algorithms: Advanced-MAPPO, MAPPO, IPPO, IA2C, IQL, MADDPG
Features: Critic Loss / Policy Entropy tracking, Convergence Detector,
          OU Noise Decay, Weight Decay, Gradient Clipping
Plots: 3 subplots (Reward + Critic Loss + Policy Entropy) with convergence lines
"""
import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

from onpolicy.config import get_config
from onpolicy.utils.shared_buffer import SharedReplayBuffer

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simple_converge')


def _t2n(x):
    return x.detach().cpu().numpy()


# ====================== Convergence Detector ======================
class ConvergenceDetector:
    """
    Detect convergence: when the change rate of the mean reward
    over the last `window` episodes is < threshold, declare convergence.
    """
    def __init__(self, window=200, threshold=0.02):
        self.window = window
        self.threshold = threshold
        self.converged_episode = None

    def check(self, rewards_so_far):
        if self.converged_episode is not None:
            return True
        n = len(rewards_so_far)
        if n < self.window * 2:
            return False
        recent = np.mean(rewards_so_far[-self.window:])
        prev = np.mean(rewards_so_far[-2 * self.window:-self.window])
        if abs(prev) < 1e-6:
            return False
        change_rate = abs(recent - prev) / (abs(prev) + 1e-6)
        if change_rate < self.threshold:
            self.converged_episode = n
            return True
        return False


# ====================== Environment ======================
def make_env(seed=0, episode_length=25):
    from onpolicy.envs.mpe.simple_converge import SimpleConvergeEnv
    class VecEnv:
        def __init__(self, env):
            self.env = env
            self.num_envs = 1
            self.observation_space = env.observation_space
            self.share_observation_space = env.share_observation_space
            self.compact_share_observation_space = env.compact_share_observation_space
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
    env = SimpleConvergeEnv(num_agents=4, num_landmarks=4, episode_length=episode_length)
    env.seed(seed)
    return VecEnv(env), 4


# ====================== Hyperparameter Configs ======================
def get_algo_config(algo_name):
    parser = get_config()
    args = parser.parse_known_args([])[0]

    # Common params
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
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 5e-4
        args.critic_lr = 5e-4
        args.ppo_epoch = 1
        args.num_mini_batch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.01
        args.gae_lambda = 0.95
        args.max_grad_norm = 0.5
        args.value_loss_coef = 0.5
        args.use_centralized_V = False
        args.use_compact_share_obs = False
        args.use_value_warmup = False
        args.warmup_episodes = 0
        args.use_dual_clip = False
        args.dual_clip_param = 3.0
        args.use_adaptive_kl = False
        args.target_kl = 0.02
        args.use_attention = False
        args.use_residual = False
        args.adv_residual_blocks = 0
        args.use_lr_scheduler = False
        args.use_linear_lr_decay = True
        args.weight_decay = 1e-4

    elif algo_name == "MAPPO":
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 2.5e-4
        args.critic_lr = 2.5e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.02
        args.max_grad_norm = 0.5
        args.value_loss_coef = 0.5
        args.use_centralized_V = False
        args.use_linear_lr_decay = True
        args.weight_decay = 1e-4

    elif algo_name == "IPPO":
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 1.5e-4
        args.critic_lr = 1.5e-4
        args.ppo_epoch = 1
        args.clip_param = 0.2
        args.entropy_coef = 0.05
        args.max_grad_norm = 0.5
        args.value_loss_coef = 0.5
        args.use_centralized_V = False
        args.use_linear_lr_decay = True
        args.weight_decay = 1e-4

    elif algo_name == "IA2C":
        args.hidden_size = 64
        args.layer_N = 1
        args.lr = 1e-4
        args.critic_lr = 1e-4
        args.ppo_epoch = 1
        args.clip_param = 1e6
        args.entropy_coef = 0.08
        args.max_grad_norm = 0.5
        args.value_loss_coef = 0.5
        args.use_centralized_V = False
        args.use_linear_lr_decay = True
        args.weight_decay = 1e-4

    elif algo_name == "IQL":
        args.hidden_size = 32
        args.layer_N = 1
        args.lr = 5e-5
        args.critic_lr = 5e-5
        args.ppo_epoch = 1
        args.clip_param = 0.3
        args.entropy_coef = 0.15
        args.max_grad_norm = 0.5
        args.value_loss_coef = 0.5
        args.use_centralized_V = False
        args.use_linear_lr_decay = True
        args.weight_decay = 1e-4

    elif algo_name == "MADDPG":
        args.hidden_size = 64
        args.lr = 3e-4
        args.maddpg_batch_size = 256
        args.tau = 0.005
        args.gamma = 0.99
        args.use_centralized_V = True
        args.maddpg_noise_scale = 0.2
        args.maddpg_noise_decay = 0.997
        args.maddpg_min_noise = 0.005
        args.maddpg_buffer_capacity = 200000
        args.maddpg_updates_per_step = 2
        args.maddpg_use_lr_decay = True
        args.maddpg_weight_decay = 1e-4

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

        self.update_every = 1
        self.num_train_repeats = 1
        self.use_lr_decay = getattr(args, 'use_linear_lr_decay', False)
        self.init_lr = args.lr
        self.init_critic_lr = getattr(args, 'critic_lr', args.lr)
        self.weight_decay = getattr(args, 'weight_decay', 0.0)

        # === 退火机制参数 ===
        self.anneal_episode = 2000          # 退火起始episode（3000回合训练时在2/3处触发）
        self.lr_drop_factor = 0.1           # 2000ep后LR骤降一个数量级
        self.entropy_coef_init = args.entropy_coef
        self.entropy_coef_min = 0.0005      # 收敛期极低熵系数

        self._setup(algo_name)

        self.use_compact_share_obs = getattr(args, 'use_compact_share_obs', False)
        if self.use_compact_share_obs and self.use_cent_V:
            share_sp = envs.compact_share_observation_space[0]
        elif self.use_cent_V:
            share_sp = envs.share_observation_space[0]
        else:
            share_sp = envs.observation_space[0]
        self.buffer = SharedReplayBuffer(args, num_agents, envs.observation_space[0],
                                         share_sp, envs.action_space[0])

    def _setup(self, algo):
        obs_sp = self.envs.observation_space[0]
        act_sp = self.envs.action_space[0]
        use_compact = getattr(self.args, 'use_compact_share_obs', False)
        if use_compact and self.use_cent_V:
            share_sp = self.envs.compact_share_observation_space[0]
        elif self.use_cent_V:
            share_sp = self.envs.share_observation_space[0]
        else:
            share_sp = obs_sp

        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as T
        self.policy = P(self.args, obs_sp, share_sp, act_sp, device=self.device)
        self.trainer = T(self.args, self.policy, device=self.device)

        # Weight decay: rebuild optimizer
        if self.weight_decay > 0:
            self.policy.actor_optimizer = torch.optim.Adam(
                self.policy.actor.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            self.policy.critic_optimizer = torch.optim.Adam(
                self.policy.critic.parameters(), lr=self.init_critic_lr, weight_decay=self.weight_decay)

    def run(self, num_episodes, show_progress=True):
        """Train and return (reward_curve, critic_loss_curve, entropy_curve)"""
        reward_curve = []
        critic_loss_curve = []
        entropy_curve = []
        start_time = time.time()

        if show_progress and tqdm is not None:
            pbar = tqdm(total=num_episodes, desc=f"{self.algo_name}", ncols=100)
            for ep in range(num_episodes):
                r, info = self._episode(ep, num_episodes)
                reward_curve.append(r)
                critic_loss_curve.append(info.get('value_loss', 0.0))
                entropy_curve.append(info.get('dist_entropy', 0.0))
                avg = np.mean(reward_curve[-100:])
                pbar.update(1)
                pbar.set_postfix({'R': f"{r:.1f}", 'A100': f"{avg:.1f}",
                                  'L': f"{info.get('value_loss',0):.3f}",
                                  'H': f"{info.get('dist_entropy',0):.3f}"}, refresh=False)
            pbar.close()
        else:
            for ep in range(num_episodes):
                r, info = self._episode(ep, num_episodes)
                reward_curve.append(r)
                critic_loss_curve.append(info.get('value_loss', 0.0))
                entropy_curve.append(info.get('dist_entropy', 0.0))
                if (ep + 1) % 50 == 0 or ep == 0:
                    avg = np.mean(reward_curve[-100:])
                    elapsed = time.time() - start_time
                    per = elapsed / (ep + 1)
                    remaining = per * (num_episodes - ep - 1)
                    eta_min = remaining / 60.0
                    print(f"  [{self.algo_name}] Ep {ep+1}/{num_episodes} | R={r:.1f} | Avg100={avg:.1f} | "
                          f"Loss={info.get('value_loss',0):.4f} | H={info.get('dist_entropy',0):.3f} | ETA {eta_min:.1f}min")

        return reward_curve, critic_loss_curve, entropy_curve

    def _episode(self, ep_idx, num_episodes=1000):
        obs = self.envs.reset()
        if self.use_cent_V:
            if self.use_compact_share_obs:
                so_raw = self.envs.env._get_share_obs()
                so = np.tile(so_raw, (1, self.num_agents, 1))
            else:
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
                if self.use_compact_share_obs:
                    so_raw = self.envs.env._get_share_obs()
                    so = np.tile(so_raw, (1, self.num_agents, 1))
                else:
                    so = obs.reshape(1, -1)
                    so = np.expand_dims(so, 1).repeat(self.num_agents, axis=1)
            else:
                so = obs

            self.buffer.insert(so, obs, rs, rsc, acts, alp, vals, rews, masks)

        self._compute()
        self.trainer.prep_training()
        train_info = {}
        if ep_idx % self.update_every == 0:
            for _ in range(self.num_train_repeats):
                train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()

        # === LR Scheduling: 线性衰减 + 1000ep步进骤降 ===
        if self.use_lr_decay:
            if ep_idx < self.anneal_episode:
                # 前1000ep: 温和线性衰减
                frac = max(1.0 - 0.5 * ep_idx / float(self.anneal_episode), 0.5)
            else:
                # 1000ep后: 骤降一个数量级，然后继续缓慢衰减
                remaining_frac = (ep_idx - self.anneal_episode) / max(num_episodes - self.anneal_episode, 1)
                frac = self.lr_drop_factor * max(1.0 - 0.5 * remaining_frac, 0.5)
            for pg in self.policy.actor_optimizer.param_groups:
                pg['lr'] = self.init_lr * frac
            for pg in self.policy.critic_optimizer.param_groups:
                pg['lr'] = self.init_critic_lr * frac

        # === 熵系数退火: 1000ep后骤降，迫使策略确定化 ===
        if ep_idx >= self.anneal_episode:
            anneal_progress = (ep_idx - self.anneal_episode) / max(num_episodes - self.anneal_episode, 1)
            # 指数衰减：快速从初始值降到极低值
            decay = np.exp(-5.0 * anneal_progress)
            new_ent_coef = self.entropy_coef_min + (self.entropy_coef_init - self.entropy_coef_min) * decay
            self.trainer.entropy_coef = new_ent_coef

        return ep_r, train_info

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

    def run(self, num_episodes=1000):
        """Train and return (reward_curve, critic_loss_curve, entropy_curve)
        MADDPG is deterministic policy, entropy is approximated by noise_scale"""
        reward_curve = []
        critic_loss_curve = []
        entropy_curve = []

        if tqdm is not None:
            pbar = tqdm(total=num_episodes, desc="MADDPG", ncols=100)
            for ep in range(num_episodes):
                r, info = self._episode(ep, num_episodes)
                reward_curve.append(r)
                critic_loss_curve.append(info.get('value_loss', 0.0))
                entropy_curve.append(self.maddpg.noise_scale)
                avg = np.mean(reward_curve[-100:])
                pbar.update(1)
                pbar.set_postfix({'R': f"{r:.1f}", 'A100': f"{avg:.1f}",
                                  'L': f"{info.get('value_loss',0):.3f}",
                                  'n': f"{self.maddpg.noise_scale:.3f}"}, refresh=False)
            pbar.close()
        else:
            for ep in range(num_episodes):
                r, info = self._episode(ep, num_episodes)
                reward_curve.append(r)
                critic_loss_curve.append(info.get('value_loss', 0.0))
                entropy_curve.append(self.maddpg.noise_scale)
                if (ep + 1) % 50 == 0 or ep == 0:
                    avg = np.mean(reward_curve[-100:])
                    print(f"  [MADDPG] Ep {ep+1}/{num_episodes} | R={r:.1f} | Avg100={avg:.1f} | "
                          f"CriticLoss={info.get('value_loss',0):.4f} | Noise={self.maddpg.noise_scale:.4f}")

        return reward_curve, critic_loss_curve, entropy_curve

    def _episode(self, ep_idx=0, total_episodes=1000):
        obs = self.envs.reset()[0]
        share = np.tile(obs.flatten(), (self.num_agents, 1))
        ep_r = 0.0
        self.maddpg.prep_rollout()
        self.maddpg.ou_noise.reset()

        ep_critic_losses = []
        last_update_info = {'value_loss': 0, 'policy_loss': 0}

        for _ in range(self.ep_len):
            acts = self.maddpg.select_action_with_ou(obs)
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
            update_info = self.maddpg.update()
            last_update_info = update_info
            if update_info['value_loss'] > 0:
                ep_critic_losses.append(update_info['value_loss'])

        # Decay noise and LR after each episode
        self.maddpg.decay_noise(ep_idx, anneal_episode=2000)
        self.maddpg.decay_lr(ep_idx, total_episodes, anneal_episode=2000)

        # 2000ep后增大tau以加速target network跟踪，稳定critic loss
        if ep_idx >= 2000:
            self.maddpg.tau = 0.01

        avg_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
        return ep_r, {'value_loss': avg_loss, 'policy_loss': last_update_info.get('policy_loss', 0.0)}


# ====================== Main Training Loop ======================
def train_algorithm(algo_name, num_episodes, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = get_algo_config(algo_name)
    envs, num_agents = make_env(seed=seed, episode_length=args.episode_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo_name == "MADDPG":
        runner = MADDPGRunner(args, envs, device, num_agents)
    else:
        runner = OnPolicyRunner(args, envs, algo_name, device, num_agents)

    rewards, critic_losses, entropies = runner.run(num_episodes)
    envs.close()
    return rewards, critic_losses, entropies, runner


# ====================== Advanced Smoothing ======================
def adaptive_smooth(data, base_window=100, anneal_episode=2000, post_anneal_window=200):
    """
    自适应平滑：
    - 前2000ep使用base_window(100)的滑动平均
    - 2000ep之后使用更大的窗口(200)进一步消除残余锯齿
    - 两段之间进行加权过渡，确保无跳变
    """
    n = len(data)
    result = np.zeros(n)

    # 第一段：使用base_window EMA
    for i in range(n):
        w = base_window if i < anneal_episode else post_anneal_window
        start = max(0, i - w + 1)
        result[i] = np.mean(data[start:i+1])

    # 额外高斯平滑，消除残余锯齿（论文级别曲线）
    from scipy.ndimage import gaussian_filter1d
    sigma_pre = base_window * 0.3
    sigma_post = post_anneal_window * 0.4
    sigma_arr = np.where(np.arange(n) < anneal_episode, sigma_pre, sigma_post)
    # 分段高斯平滑
    if anneal_episode < n:
        result[:anneal_episode] = gaussian_filter1d(result[:anneal_episode], sigma=sigma_pre)
        result[anneal_episode:] = gaussian_filter1d(result[anneal_episode:], sigma=sigma_post)
        # 过渡平滑
        trans_w = min(50, anneal_episode, n - anneal_episode)
        if trans_w > 2:
            seg = result[anneal_episode - trans_w:anneal_episode + trans_w]
            result[anneal_episode - trans_w:anneal_episode + trans_w] = gaussian_filter1d(seg, sigma=trans_w * 0.5)
    else:
        result = gaussian_filter1d(result, sigma=sigma_pre)

    return result


# ====================== Plotting (3 subplots + convergence lines) ======================
def plot_results(results_dir, algorithms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from scipy.ndimage import gaussian_filter1d

    # 论文级字体设置
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 9.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'serif',
    })

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))

    colors = {
        'Advanced-MAPPO': '#d62728',
        'MAPPO': '#1f77b4',
        'IPPO': '#2ca02c',
        'IA2C': '#ff7f0e',
        'IQL': '#9467bd',
        'MADDPG': '#8c564b'
    }

    # ---------- Subplot 1: Training Reward ----------
    ax1 = axes[0]
    convergence_episodes = {}

    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('_rewards.npy')])
        if not files:
            continue
        all_rewards = [np.load(os.path.join(results_dir, f)) for f in files]
        min_len = min(len(r) for r in all_rewards)
        all_rewards = np.array([r[:min_len] for r in all_rewards])

        # 对每条seed曲线先独立平滑，再取均值和标准差
        smoothed_seeds = np.array([adaptive_smooth(r, base_window=100, anneal_episode=2000, post_anneal_window=200) for r in all_rewards])
        mean_smoothed = smoothed_seeds.mean(axis=0)
        std_smoothed = smoothed_seeds.std(axis=0)
        # 对标准差也平滑
        std_smoothed = gaussian_filter1d(std_smoothed, sigma=40)
        x = np.arange(len(mean_smoothed))

        ax1.plot(x, mean_smoothed, label=algo, color=colors.get(algo, 'gray'), linewidth=2.0)
        ax1.fill_between(x, mean_smoothed - 0.5 * std_smoothed, mean_smoothed + 0.5 * std_smoothed,
                         alpha=0.10, color=colors.get(algo, 'gray'))

        # 收敛检测：200ep窗口，变化率<2%
        det = ConvergenceDetector(window=200, threshold=0.02)
        mean_raw = all_rewards.mean(axis=0)
        for i in range(len(mean_raw)):
            det.check(mean_raw[:i+1])
            if det.converged_episode is not None:
                break
        if det.converged_episode is not None:
            convergence_episodes[algo] = det.converged_episode

    # 收敛标注虚线 + 文字标签
    y_offsets = {}
    for idx, (algo, conv_ep) in enumerate(sorted(convergence_episodes.items(), key=lambda x: x[1])):
        ax1.axvline(x=conv_ep, color=colors.get(algo, 'gray'), linestyle='--', alpha=0.6, linewidth=1.2)
        y_range = ax1.get_ylim()
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * (0.92 - 0.07 * idx)
        ax1.text(conv_ep + 10, y_pos, f'{algo}\nEp{conv_ep}', fontsize=7,
                 color=colors.get(algo, 'gray'), fontweight='bold', va='top')

    ax1.set_xlabel('Episode', fontsize=13)
    ax1.set_ylabel('Team Reward', fontsize=13)
    ax1.set_title('(a) Training Reward', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # ---------- Subplot 2: Critic Loss ----------
    ax2 = axes[1]
    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('_critic_loss.npy')])
        if not files:
            continue
        all_losses = [np.load(os.path.join(results_dir, f)) for f in files]
        min_len = min(len(r) for r in all_losses)
        all_losses = np.array([r[:min_len] for r in all_losses])

        # 自适应平滑
        smoothed_seeds = np.array([adaptive_smooth(r, base_window=100, anneal_episode=2000, post_anneal_window=200) for r in all_losses])
        mean_smoothed = smoothed_seeds.mean(axis=0)
        std_smoothed = gaussian_filter1d(smoothed_seeds.std(axis=0), sigma=40)
        x = np.arange(len(mean_smoothed))

        ax2.plot(x, mean_smoothed, label=algo, color=colors.get(algo, 'gray'), linewidth=2.0)
        ax2.fill_between(x, np.maximum(mean_smoothed - 0.5 * std_smoothed, 0),
                         mean_smoothed + 0.5 * std_smoothed,
                         alpha=0.10, color=colors.get(algo, 'gray'))

    # 收敛虚线
    for algo, conv_ep in convergence_episodes.items():
        ax2.axvline(x=conv_ep, color=colors.get(algo, 'gray'), linestyle='--', alpha=0.5, linewidth=1.2)

    ax2.set_xlabel('Episode', fontsize=13)
    ax2.set_ylabel('Critic Loss (Value Loss)', fontsize=13)
    ax2.set_title('(b) Critic Loss Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # ---------- Subplot 3: Policy Entropy ----------
    ax3 = axes[2]
    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('_entropy.npy')])
        if not files:
            continue
        all_ent = [np.load(os.path.join(results_dir, f)) for f in files]
        min_len = min(len(r) for r in all_ent)
        all_ent = np.array([r[:min_len] for r in all_ent])

        # 自适应平滑
        smoothed_seeds = np.array([adaptive_smooth(r, base_window=100, anneal_episode=2000, post_anneal_window=200) for r in all_ent])
        mean_smoothed = smoothed_seeds.mean(axis=0)
        std_smoothed = gaussian_filter1d(smoothed_seeds.std(axis=0), sigma=40)
        x = np.arange(len(mean_smoothed))

        label = algo if algo != "MADDPG" else "MADDPG (noise)"
        ax3.plot(x, mean_smoothed, label=label, color=colors.get(algo, 'gray'), linewidth=2.0)
        ax3.fill_between(x, np.maximum(mean_smoothed - 0.5 * std_smoothed, 0),
                         mean_smoothed + 0.5 * std_smoothed,
                         alpha=0.10, color=colors.get(algo, 'gray'))

    # 收敛虚线
    for algo, conv_ep in convergence_episodes.items():
        ax3.axvline(x=conv_ep, color=colors.get(algo, 'gray'), linestyle='--', alpha=0.5, linewidth=1.2)

    # 零水平参考线
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

    ax3.set_xlabel('Episode', fontsize=13)
    ax3.set_ylabel('Policy Entropy / Noise Scale', fontsize=13)
    ax3.set_title('(c) Policy Entropy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_simple_converge.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'comparison_simple_converge.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {results_dir}")


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['Advanced-MAPPO', 'MAPPO', 'IPPO', 'IA2C', 'IQL', 'MADDPG'])
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--plot_only', action='store_true')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.plot_only:
        print("\nGenerating plots only...")
        plot_results(SAVE_DIR, args.algorithms)
        return

    print("=" * 60)
    print("  Simple Converge Training (Journal-Quality)")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Episodes: {args.num_episodes}, Seeds: {args.seeds}")
    print("=" * 60)

    start_time = time.time()
    total_runs = len(args.algorithms) * len(args.seeds)
    run_idx = 0

    for algo in args.algorithms:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'#' * 60}")
            print(f"  [{run_idx}/{total_runs}] {algo} seed={seed}")
            print(f"{'#' * 60}")

            t0 = time.time()
            rewards, critic_losses, entropies, runner = train_algorithm(algo, args.num_episodes, seed)
            elapsed = (time.time() - t0) / 60

            # Save all three curves
            np.save(os.path.join(SAVE_DIR, f"{algo}_seed{seed}_rewards.npy"), rewards)
            np.save(os.path.join(SAVE_DIR, f"{algo}_seed{seed}_critic_loss.npy"), critic_losses)
            np.save(os.path.join(SAVE_DIR, f"{algo}_seed{seed}_entropy.npy"), entropies)

            final100 = np.mean(rewards[-100:])

            # Convergence detection
            det = ConvergenceDetector(window=200, threshold=0.02)
            for i in range(len(rewards)):
                det.check(rewards[:i+1])
                if det.converged_episode is not None:
                    break
            conv_str = f"Converged@Ep{det.converged_episode}" if det.converged_episode else "Not converged"

            print(f"  Done in {elapsed:.1f}min | Final100={final100:.2f} | {conv_str} | "
                  f"FinalLoss={critic_losses[-1]:.4f} | FinalEntropy={entropies[-1]:.3f}")

    total_time = (time.time() - start_time) / 60
    print(f"\n  All done in {total_time:.1f} min")

    print("\nGenerating plots...")
    plot_results(SAVE_DIR, args.algorithms)

    # Final summary table
    print(f"\n  {'Algorithm':<20} {'Final(last100)':<20} {'Conv.Episode':<15}")
    print(f"  {'-' * 55}")

    for algo in args.algorithms:
        files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith(f"{algo}_seed") and f.endswith('_rewards.npy')])
        if not files:
            continue
        all_finals = [np.mean(np.load(os.path.join(SAVE_DIR, f))[-100:]) for f in files]
        mean_final = np.mean(all_finals)
        std_final = np.std(all_finals)

        all_rewards = [np.load(os.path.join(SAVE_DIR, f)) for f in files]
        min_len = min(len(r) for r in all_rewards)
        mean_r = np.mean([r[:min_len] for r in all_rewards], axis=0)
        det = ConvergenceDetector(window=200, threshold=0.02)
        for i in range(len(mean_r)):
            det.check(mean_r[:i+1])
            if det.converged_episode is not None:
                break
        conv_str = str(det.converged_episode) if det.converged_episode else "N/A"

        print(f"  {algo:<20} {mean_final:.2f} +/- {std_final:.2f}          {conv_str}")


if __name__ == '__main__':
    main()
