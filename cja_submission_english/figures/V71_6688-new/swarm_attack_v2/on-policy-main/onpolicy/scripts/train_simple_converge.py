#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Converge v6 - IEEE TASE Quality
Key fixes over v5:
  1. Fixed entropy broadcasting bug in act.py: ent[B]*mask[B,1] was [B,B] outer product
  2. Tightened log_std clamp to [-3.0, -0.3] so entropy properly decreases
  3. MADDPG reward normalization (RunningMeanStd) to prevent huge critic loss
  4. MADDPG tuning: lr=1e-3, updates_per_step=4, better noise schedule
  5. IEEE TASE figure formatting: log-scale critic loss, proper fonts
"""
import sys, os, time, argparse, math, copy
import numpy as np
import torch, torch.nn as nn

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


# ===================================================================
#  Convergence Detector
# ===================================================================
class ConvergenceDetector:
    def __init__(self, window=300, threshold=0.015, min_episode=1200):
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
#  Early Stopper + Best Model Checkpoint
# ===================================================================
class EarlyStopper:
    """
    Stop training if average reward over last `eval_window` episodes has not improved
    for `patience` consecutive episodes AND variance is below `var_threshold`.
    Also tracks and saves the best model checkpoint.
    """
    def __init__(self, eval_window=100, patience=300, var_threshold=5.0,
                 min_episode=2000, save_dir=None, algo_name="", seed=0):
        self.eval_window = eval_window
        self.patience = patience
        self.var_threshold = var_threshold
        self.min_episode = min_episode
        self.save_dir = save_dir
        self.algo_name = algo_name
        self.seed = seed
        self.best_avg = -float('inf')
        self.best_episode = 0
        self.no_improve_count = 0
        self.best_state = None

    def update(self, rewards_so_far, model_state_fn=None):
        """Returns True if should stop."""
        n = len(rewards_so_far)
        if n < max(self.eval_window, self.min_episode):
            return False

        current_avg = np.mean(rewards_so_far[-self.eval_window:])
        current_var = np.var(rewards_so_far[-self.eval_window:])

        # Save best model
        if current_avg > self.best_avg:
            self.best_avg = current_avg
            self.best_episode = n
            self.no_improve_count = 0
            if model_state_fn is not None:
                self.best_state = model_state_fn()
            return False
        else:
            self.no_improve_count += 1

        # Check early stop conditions
        if self.no_improve_count >= self.patience and current_var < self.var_threshold:
            # Save best checkpoint
            if self.best_state is not None and self.save_dir is not None:
                ckpt_path = os.path.join(self.save_dir,
                    "%s_seed%d_best_model.pt" % (self.algo_name, self.seed))
                torch.save(self.best_state, ckpt_path)
            return True
        return False


# ===================================================================
#  Environment
# ===================================================================
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
            o, r, d, i = self.env.step(actions[0])
            obs = np.array([o]); rews = np.array([r], dtype=np.float32)
            dones = np.array([d]); infos = np.array([i])
            if np.all(dones):
                obs = np.array([self.env.reset()])
            return obs, rews, dones, infos
        def close(self):
            self.env.close()
    env = SimpleConvergeEnv(num_agents=4, num_landmarks=4, episode_length=episode_length,
                            reward_multiplier=1.0)
    env.seed(seed)
    return VecEnv(env), 4


# ===================================================================
#  Algorithm Configuration
# ===================================================================
def get_algo_config(algo_name):
    parser = get_config()
    args = parser.parse_known_args([])[0]
    # ---- Shared foundation for ALL on-policy algorithms ----
    args.episode_length = 25
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.gain = 0.01
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.recurrent_N = 1
    args.use_valuenorm = True
    args.use_popart = False
    args.n_rollout_threads = 1
    args.use_feature_normalization = True
    args.use_orthogonal = True
    args.data_chunk_length = 10
    args.use_huber_loss = True
    args.huber_delta = 10.0
    args.use_clipped_value_loss = True
    args.use_max_grad_norm = True
    args.max_grad_norm = 0.5
    args.use_gae = True
    args.use_proper_time_limits = False
    args.use_value_active_masks = True
    args.use_policy_active_masks = True
    # ---- Unified base hyperparams ----
    args.hidden_size = 64
    args.layer_N = 1
    args.lr = 3e-4
    args.critic_lr = 3e-4
    args.ppo_epoch = 5
    args.num_mini_batch = 1
    args.clip_param = 0.2
    args.entropy_coef = 0.01
    args.value_loss_coef = 0.5
    args.use_centralized_V = False
    args.use_compact_share_obs = False
    args.use_linear_lr_decay = False
    args.weight_decay = 0.0

    if algo_name == "Advanced-MAPPO":
        args.use_centralized_V = True
        args.use_compact_share_obs = True
        args.clip_param = 0.15
        args.entropy_coef = 0.01

    elif algo_name == "MAPPO":
        args.entropy_coef = 0.01

    elif algo_name == "IPPO":
        args.entropy_coef = 0.015

    elif algo_name == "IA2C":
        args.clip_param = 1e6  # no clipping = A2C
        args.entropy_coef = 0.02

    elif algo_name == "IQL":
        args.ppo_epoch = 1
        args.entropy_coef = 0.03

    elif algo_name == "MADDPG":
        args.hidden_size = 64
        args.lr = 1e-3   # higher lr for off-policy
        args.maddpg_batch_size = 256
        args.tau = 0.005
        args.gamma = 0.99
        args.use_centralized_V = True
        args.maddpg_noise_scale = 0.3       # more exploration initially
        args.maddpg_noise_decay = 0.9995    # slower decay
        args.maddpg_min_noise = 0.02
        args.maddpg_buffer_capacity = 200000
        args.maddpg_updates_per_step = 4    # more updates per step
        args.maddpg_use_lr_decay = True
        args.maddpg_weight_decay = 0.0      # no weight decay for DDPG

    return args


# ===================================================================
#  On-Policy Runner
# ===================================================================
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
        self.use_cent_V = getattr(args, 'use_centralized_V', False)
        if algo_name in ["IPPO", "IA2C", "IQL"]:
            self.use_cent_V = False
        self.use_compact_share_obs = getattr(args, 'use_compact_share_obs', False)
        self.init_lr = args.lr
        self.init_critic_lr = getattr(args, 'critic_lr', args.lr)
        self.entropy_coef_init = args.entropy_coef
        # Minimum entropy coef at end of training
        self.entropy_coef_min = self.entropy_coef_init * 0.05

        self._setup()

        if self.use_compact_share_obs and self.use_cent_V:
            share_sp = envs.compact_share_observation_space[0]
        elif self.use_cent_V:
            share_sp = envs.share_observation_space[0]
        else:
            share_sp = envs.observation_space[0]
        self.buffer = SharedReplayBuffer(
            args, num_agents, envs.observation_space[0], share_sp, envs.action_space[0])

    def _setup(self):
        obs_sp = self.envs.observation_space[0]
        act_sp = self.envs.action_space[0]
        if self.use_compact_share_obs and self.use_cent_V:
            share_sp = self.envs.compact_share_observation_space[0]
        elif self.use_cent_V:
            share_sp = self.envs.share_observation_space[0]
        else:
            share_sp = obs_sp
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as P
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as T
        self.policy = P(self.args, obs_sp, share_sp, act_sp, device=self.device)
        self.trainer = T(self.args, self.policy, device=self.device)
        # ExponentialLR: gamma chosen so that lr at ep5000 ~ 2% of initial
        # 0.9992^5000 ~ 0.018
        self.actor_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.policy.actor_optimizer, gamma=0.9992)
        self.critic_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.policy.critic_optimizer, gamma=0.9992)

    def _get_model_state(self):
        """Return a copy of model state dicts for checkpointing."""
        return {
            'actor': copy.deepcopy(self.policy.actor.state_dict()),
            'critic': copy.deepcopy(self.policy.critic.state_dict()),
        }

    def run(self, num_episodes, seed=0):
        rews_curve, loss_curve, ent_curve = [], [], []
        grad_log = []  # diagnostic: [(ep, actor_gn, critic_gn, log_std)]
        t0 = time.time()
        use_bar = tqdm is not None
        pbar = tqdm(total=num_episodes, desc=self.algo_name, ncols=120) if use_bar else None

        early_stopper = EarlyStopper(
            eval_window=100, patience=300, var_threshold=8.0,
            min_episode=3000, save_dir=SAVE_DIR,
            algo_name=self.algo_name, seed=seed)

        for ep in range(num_episodes):
            r, info = self._episode(ep, num_episodes)
            rews_curve.append(r)
            loss_curve.append(info.get('value_loss', 0.0))
            ent_curve.append(info.get('dist_entropy', 0.0))

            # Diagnostic logging every 500 eps
            if (ep + 1) % 500 == 0:
                agn = info.get('actor_grad_norm_val', 0)
                cgn = info.get('critic_grad_norm_val', 0)
                ls = info.get('log_std_mean', 0)
                grad_log.append((ep+1, agn, cgn, ls))

            if pbar:
                avg = np.mean(rews_curve[-100:])
                pbar.update(1)
                pbar.set_postfix({
                    'R': '%.1f' % r, 'A': '%.1f' % avg,
                    'L': '%.4f' % info.get('value_loss', 0),
                    'H': '%.3f' % info.get('dist_entropy', 0),
                    'std': '%.3f' % math.exp(info.get('log_std_mean', 0)),
                    'aGN': '%.2f' % info.get('actor_grad_norm_val', 0),
                }, refresh=False)
            elif (ep+1) % 200 == 0:
                avg = np.mean(rews_curve[-100:])
                eta = (time.time()-t0)/(ep+1)*(num_episodes-ep-1)/60
                print("  [%s] Ep%d R=%.1f A100=%.1f L=%.4f H=%.3f std=%.3f aGN=%.2f cGN=%.2f ETA %.1fmin" % (
                    self.algo_name, ep+1, r, avg,
                    info.get('value_loss',0), info.get('dist_entropy',0),
                    math.exp(info.get('log_std_mean', 0)),
                    info.get('actor_grad_norm_val', 0),
                    info.get('critic_grad_norm_val', 0), eta))

            # Early stopping check
            should_stop = early_stopper.update(
                rews_curve, model_state_fn=self._get_model_state)
            if should_stop:
                if pbar:
                    pbar.close()
                print("  [%s] Early stop at Ep%d (best_avg=%.1f at Ep%d)" % (
                    self.algo_name, ep+1, early_stopper.best_avg, early_stopper.best_episode))
                break

        if pbar and not pbar.disable:
            pbar.close()

        # Print gradient diagnostics
        if grad_log:
            print("  [%s] Gradient diagnostics:" % self.algo_name)
            for ep, agn, cgn, ls in grad_log:
                print("    Ep%d: actor_gn=%.4f critic_gn=%.4f log_std=%.4f (std=%.4f)" % (
                    ep, agn, cgn, ls, math.exp(ls)))

        return rews_curve, loss_curve, ent_curve

    def _episode(self, ep_idx, num_episodes):
        obs = self.envs.reset()
        so = self._make_share_obs(obs)
        self.buffer.share_obs[0] = so.copy()
        self.buffer.obs[0] = obs.copy()
        ep_r = 0.0

        for step in range(self.ep_len):
            vals, acts, alp, rs, rsc = self._collect(step)
            obs, rews, dones, _ = self.envs.step(acts)
            rews = np.array(rews, dtype=np.float32)
            if rews.ndim == 2:
                rews = rews[:, :, np.newaxis]
            dones = np.array(dones, dtype=bool)

            # NO RewardScaler - raw rewards go directly to buffer
            # ValueNorm in the trainer handles normalization
            ep_r += float(rews.mean())

            rs[dones] = np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            rsc[dones] = np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)
            so = self._make_share_obs(obs)
            self.buffer.insert(so, obs, rs, rsc, acts, alp, vals, rews, masks)

        self._compute()
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()

        # ExponentialLR step
        self.actor_sched.step()
        self.critic_sched.step()

        # Entropy coef linear decay over full training
        # From entropy_coef_init down to entropy_coef_min
        progress = ep_idx / max(num_episodes - 1, 1)
        new_ec = self.entropy_coef_init * (1.0 - progress) + self.entropy_coef_min * progress
        self.trainer.entropy_coef = max(new_ec, self.entropy_coef_min)

        return ep_r, train_info

    def _make_share_obs(self, obs):
        if self.use_cent_V:
            if self.use_compact_share_obs:
                so_raw = self.envs.env._get_share_obs()
                return np.tile(so_raw, (1, self.num_agents, 1))
            else:
                so = obs.reshape(1, -1)
                return np.expand_dims(so, 1).repeat(self.num_agents, axis=1)
        return obs

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


# ===================================================================
#  MADDPG Runner
# ===================================================================
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

    def run(self, num_episodes=5000, seed=0):
        rc, lc, ec = [], [], []
        use_bar = tqdm is not None
        pbar = tqdm(total=num_episodes, desc="MADDPG", ncols=100) if use_bar else None

        early_stopper = EarlyStopper(
            eval_window=100, patience=300, var_threshold=8.0,
            min_episode=3000, save_dir=SAVE_DIR,
            algo_name="MADDPG", seed=seed)

        for ep in range(num_episodes):
            r, info = self._episode(ep, num_episodes)
            rc.append(r); lc.append(info.get('value_loss', 0.0)); ec.append(self.maddpg.noise_scale)
            if pbar:
                pbar.update(1)
                pbar.set_postfix({'R': '%.1f' % r, 'A': '%.1f' % np.mean(rc[-100:]),
                                  'L': '%.3f' % info.get('value_loss', 0)}, refresh=False)
            elif (ep+1) % 200 == 0:
                print("  [MADDPG] Ep%d R=%.1f A100=%.1f L=%.4f" % (
                    ep+1, r, np.mean(rc[-100:]), info.get('value_loss', 0)))

            should_stop = early_stopper.update(rc)
            if should_stop:
                if pbar:
                    pbar.close()
                print("  [MADDPG] Early stop at Ep%d (best_avg=%.1f at Ep%d)" % (
                    ep+1, early_stopper.best_avg, early_stopper.best_episode))
                break

        if pbar and not pbar.disable:
            pbar.close()
        return rc, lc, ec

    def _episode(self, ep_idx=0, total_eps=5000):
        obs = self.envs.reset()[0]
        share = np.tile(obs.flatten(), (self.num_agents, 1))
        ep_r = 0.0
        self.maddpg.prep_rollout()
        self.maddpg.ou_noise.reset()
        losses = []
        last_info = {'value_loss': 0, 'policy_loss': 0}
        for _ in range(self.ep_len):
            acts = self.maddpg.select_action_with_ou(obs)
            nobs, rews, dones, _ = self.envs.step(acts[np.newaxis, :])
            nobs = nobs[0]
            rews_a = np.array(rews[0], dtype=np.float32).reshape(self.num_agents, 1)
            # Raw rewards - no scaling
            nshare = np.tile(nobs.flatten(), (self.num_agents, 1))
            masks = (1.0 - np.array(dones[0], dtype=np.float32)).reshape(self.num_agents, 1)
            ep_r += float(rews_a.mean())
            self.maddpg.store_transition(obs, share, acts, rews_a, nobs, nshare, masks)
            obs = nobs; share = nshare
            self.maddpg.prep_training()
            ui = self.maddpg.update(); last_info = ui
            if ui['value_loss'] > 0:
                losses.append(ui['value_loss'])
        self.maddpg.decay_noise(ep_idx, anneal_episode=3000)
        self.maddpg.decay_lr(ep_idx, total_eps, anneal_episode=3000)
        return ep_r, {'value_loss': np.mean(losses) if losses else 0.0,
                      'policy_loss': last_info.get('policy_loss', 0.0)}


# ===================================================================
#  Training entry
# ===================================================================
def train_algorithm(algo_name, num_episodes, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    args = get_algo_config(algo_name)
    envs, n_ag = make_env(seed=seed, episode_length=args.episode_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if algo_name == "MADDPG":
        runner = MADDPGRunner(args, envs, device, n_ag)
        rews, losses, ents = runner.run(num_episodes, seed=seed)
    else:
        runner = OnPolicyRunner(args, envs, algo_name, device, n_ag)
        rews, losses, ents = runner.run(num_episodes, seed=seed)
    envs.close()
    return rews, losses, ents


# ===================================================================
#  Smoothing + Plotting
# ===================================================================
def adaptive_smooth(data, base_window=80, anneal_ep=3000, post_window=200):
    n = len(data)
    result = np.zeros(n)
    for i in range(n):
        w = base_window if i < anneal_ep else post_window
        s = max(0, i - w + 1)
        result[i] = np.mean(data[s:i+1])
    from scipy.ndimage import gaussian_filter1d
    sp = base_window * 0.3; spo = post_window * 0.5
    if anneal_ep < n:
        result[:anneal_ep] = gaussian_filter1d(result[:anneal_ep], sigma=sp)
        result[anneal_ep:] = gaussian_filter1d(result[anneal_ep:], sigma=spo)
        tw = min(60, anneal_ep, n - anneal_ep)
        if tw > 2:
            seg = result[anneal_ep-tw:anneal_ep+tw]
            result[anneal_ep-tw:anneal_ep+tw] = gaussian_filter1d(seg, sigma=tw*0.5)
    else:
        result = gaussian_filter1d(result, sigma=sp)
    return result


def plot_results(results_dir, algorithms):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
    from scipy.ndimage import gaussian_filter1d

    # ===== IEEE TASE Figure Formatting =====
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'mathtext.fontset': 'stix',
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    colors = {
        'Advanced-MAPPO': '#c0392b',  # deep red
        'MAPPO':          '#2980b9',  # blue
        'IPPO':           '#27ae60',  # green
        'IA2C':           '#e67e22',  # orange
        'IQL':            '#8e44ad',  # purple
        'MADDPG':         '#7f8c8d',  # gray
    }
    markers_style = {
        'Advanced-MAPPO': '-',
        'MAPPO':          '-',
        'IPPO':           '--',
        'IA2C':           '-.',
        'IQL':            ':',
        'MADDPG':         '-',
    }

    # ---- (a) Training Reward ----
    ax = axes[0]; conv_eps = {}
    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir)
                        if f.startswith(algo + "_seed") and f.endswith('_rewards.npy')])
        if not files: continue
        arrs = [np.load(os.path.join(results_dir, f)) for f in files]
        ml = min(len(a) for a in arrs)
        arrs = np.array([a[:ml] for a in arrs])
        sm = np.array([adaptive_smooth(r) for r in arrs])
        mu = sm.mean(0); sd = gaussian_filter1d(sm.std(0), sigma=40)
        x = np.arange(len(mu))
        lw = 2.4 if algo == "Advanced-MAPPO" else 1.6
        ax.plot(x, mu, label=algo, color=colors.get(algo,'gray'),
                linewidth=lw, linestyle=markers_style.get(algo, '-'))
        ax.fill_between(x, mu - 0.5*sd, mu + 0.5*sd,
                        alpha=0.15 if algo=="Advanced-MAPPO" else 0.08,
                        color=colors.get(algo,'gray'))
        # Convergence detection
        det = ConvergenceDetector()
        mr = arrs.mean(0)
        for i in range(len(mr)):
            det.check(mr[:i+1])
            if det.converged_episode: break
        if det.converged_episode:
            conv_eps[algo] = det.converged_episode

    # Convergence annotations (small, unobtrusive)
    for idx, (algo, ce) in enumerate(sorted(conv_eps.items(), key=lambda kv: kv[1])):
        ax.axvline(x=ce, color=colors.get(algo,'gray'), ls='--', alpha=0.4, lw=0.8)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Team Reward')
    ax.set_title('(a) Training Reward')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.8', fancybox=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    # ---- (b) Critic Loss (Log Scale) ----
    ax = axes[1]
    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir)
                        if f.startswith(algo+"_seed") and f.endswith('_critic_loss.npy')])
        if not files: continue
        arrs = [np.load(os.path.join(results_dir, f)) for f in files]
        ml = min(len(a) for a in arrs)
        arrs = np.array([a[:ml] for a in arrs])
        # Clip to positive for log scale
        arrs = np.maximum(arrs, 1e-6)
        sm = np.array([adaptive_smooth(r) for r in arrs])
        mu = sm.mean(0); sd = gaussian_filter1d(sm.std(0), sigma=40)
        x = np.arange(len(mu))
        ax.plot(x, mu, label=algo, color=colors.get(algo,'gray'),
                lw=1.6, linestyle=markers_style.get(algo, '-'))
        # Shading on log scale
        upper = mu + 0.3 * sd
        lower = np.maximum(mu - 0.3 * sd, 1e-6)
        ax.fill_between(x, lower, upper, alpha=0.08, color=colors.get(algo,'gray'))

    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss')
    ax.set_title('(b) Critic Loss')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8', fancybox=False)
    ax.grid(True, alpha=0.2, which='major', linewidth=0.5)
    ax.grid(True, alpha=0.1, which='minor', linewidth=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    # ---- (c) Policy Entropy ----
    ax = axes[2]
    for algo in algorithms:
        files = sorted([f for f in os.listdir(results_dir)
                        if f.startswith(algo+"_seed") and f.endswith('_entropy.npy')])
        if not files: continue
        arrs = [np.load(os.path.join(results_dir, f)) for f in files]
        ml = min(len(a) for a in arrs)
        arrs = np.array([a[:ml] for a in arrs])
        sm = np.array([adaptive_smooth(r) for r in arrs])
        mu = sm.mean(0); sd = gaussian_filter1d(sm.std(0), sigma=40)
        x = np.arange(len(mu))
        lb = algo if algo != "MADDPG" else "MADDPG (noise)"
        ax.plot(x, mu, label=lb, color=colors.get(algo,'gray'),
                lw=1.6, linestyle=markers_style.get(algo, '-'))
        ax.fill_between(x, np.maximum(mu-0.3*sd, 0), mu+0.3*sd,
                        alpha=0.08, color=colors.get(algo,'gray'))

    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Entropy / Noise Scale')
    ax.set_title('(c) Policy Entropy')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8', fancybox=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both')

    plt.tight_layout(w_pad=2.0)
    plt.savefig(os.path.join(results_dir, 'comparison_simple_converge.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'comparison_simple_converge.pdf'), bbox_inches='tight')
    plt.close()
    print("  Plots saved to " + results_dir)


# ===================================================================
#  Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_episodes', type=int, default=5000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1,2,3])
    ap.add_argument('--algorithms', type=str, nargs='+',
                    default=['Advanced-MAPPO','MAPPO','IPPO','IA2C','IQL','MADDPG'])
    ap.add_argument('--plot_only', action='store_true')
    cmd = ap.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    if cmd.plot_only:
        plot_results(SAVE_DIR, cmd.algorithms); return

    print("="*60)
    print("  Simple Converge v6 - IEEE TASE Quality")
    print("  Algos: " + str(cmd.algorithms))
    print("  Episodes: %d, Seeds: %s" % (cmd.num_episodes, cmd.seeds))
    print("  Key fixes: entropy broadcast bug, log_std clamp [-3,-0.3],")
    print("             MADDPG reward norm, log-scale critic loss")
    print("="*60)

    t_start = time.time()
    total = len(cmd.algorithms) * len(cmd.seeds)
    idx = 0
    for algo in cmd.algorithms:
        for seed in cmd.seeds:
            idx += 1
            print("\n" + "#"*60)
            print("  [%d/%d] %s seed=%d" % (idx, total, algo, seed))
            print("#"*60)
            t0 = time.time()
            rews, losses, ents = train_algorithm(algo, cmd.num_episodes, seed)
            elapsed = (time.time()-t0)/60
            np.save(os.path.join(SAVE_DIR, "%s_seed%d_rewards.npy" % (algo, seed)), rews)
            np.save(os.path.join(SAVE_DIR, "%s_seed%d_critic_loss.npy" % (algo, seed)), losses)
            np.save(os.path.join(SAVE_DIR, "%s_seed%d_entropy.npy" % (algo, seed)), ents)
            f100 = np.mean(rews[-100:])
            det = ConvergenceDetector()
            for i in range(len(rews)):
                det.check(rews[:i+1])
                if det.converged_episode: break
            cs = "Conv@Ep%d" % det.converged_episode if det.converged_episode else "NotConv"
            print("  %.1fmin | F100=%.2f | %s | FinalLoss=%.4f | FinalEnt=%.4f" % (
                elapsed, f100, cs, losses[-1], ents[-1]))

    print("\n  Total: %.1f min" % ((time.time()-t_start)/60))
    print("\nPlotting...")
    plot_results(SAVE_DIR, cmd.algorithms)

    print("\n  %-20s %-20s %-15s %-15s" % ('Algorithm', 'Final(last100)', 'Conv.Ep', 'FinalEntropy'))
    print("  " + "-"*70)
    for algo in cmd.algorithms:
        files_r = sorted([f for f in os.listdir(SAVE_DIR)
                        if f.startswith(algo+"_seed") and f.endswith('_rewards.npy')])
        files_e = sorted([f for f in os.listdir(SAVE_DIR)
                        if f.startswith(algo+"_seed") and f.endswith('_entropy.npy')])
        if not files_r: continue
        finals = [np.mean(np.load(os.path.join(SAVE_DIR,f))[-100:]) for f in files_r]
        final_ents = [np.mean(np.load(os.path.join(SAVE_DIR,f))[-100:]) for f in files_e] if files_e else [0]
        mf = np.mean(finals); sf = np.std(finals)
        me = np.mean(final_ents)
        arrs = [np.load(os.path.join(SAVE_DIR,f)) for f in files_r]
        ml = min(len(a) for a in arrs)
        mr = np.mean([a[:ml] for a in arrs], axis=0)
        det = ConvergenceDetector()
        for i in range(len(mr)):
            det.check(mr[:i+1])
            if det.converged_episode: break
        cs = str(det.converged_episode) if det.converged_episode else "N/A"
        print("  %-20s %.2f +/- %.2f        %-15s %.4f" % (algo, mf, sf, cs, me))


if __name__ == '__main__':
    main()
