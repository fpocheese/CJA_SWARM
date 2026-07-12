#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Converge v7 - IEEE TASE Final
Changes over v6:
  1. Extended training to 10000 episodes (from 5000)
  2. Linear LR decay: lr_init -> 0 over full training
  3. Stronger entropy_coef decay: init -> 0 (not 5% of init)
  4. Removed MADDPG (deterministic policy, not comparable with stochastic Entropy)
  5. Only 5 on-policy algorithms: Advanced-MAPPO, MAPPO, IPPO, IA2C, IQL
  6. EarlyStopper min_episode raised to 5000 (give enough time before stopping)
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

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simple_converge_v7')

def _t2n(x):
    return x.detach().cpu().numpy()


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
#  Early Stopper
# ===================================================================
class EarlyStopper:
    def __init__(self, eval_window=200, patience=500, var_threshold=5.0,
                 min_episode=5000, save_dir=None, algo_name="", seed=0):
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

    def update(self, rewards_so_far):
        n = len(rewards_so_far)
        if n < max(self.eval_window, self.min_episode):
            return False
        current_avg = np.mean(rewards_so_far[-self.eval_window:])
        current_var = np.var(rewards_so_far[-self.eval_window:])
        if current_avg > self.best_avg:
            self.best_avg = current_avg
            self.best_episode = n
            self.no_improve_count = 0
            return False
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.patience and current_var < self.var_threshold:
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
    # ---- Shared foundation ----
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
    args.lr_end = 1e-6
    args.critic_lr_end = 1e-6
    args.entropy_coef_end = 0.0
    args.lr_hold_episode = 0
    args.lr_decay_episode = None
    args.entropy_hold_episode = 0
    args.entropy_decay_episode = None

    if algo_name == "Advanced-MAPPO":
        args.use_centralized_V = True
        args.use_compact_share_obs = True
        args.hidden_size = 64
        args.layer_N = 1
        args.ppo_epoch = 5
        args.clip_param = 0.15
        args.lr = 3.0e-4
        args.critic_lr = 3.0e-4
        args.entropy_coef = 0.01
        args.value_loss_coef = 0.5
        args.max_grad_norm = 0.5
        args.lr_hold_episode = 650
        args.lr_decay_episode = 1050
        args.entropy_hold_episode = 400
        args.entropy_decay_episode = 950
        args.use_attention = True
        args.use_residual = True
        args.adv_residual_blocks = 2

    elif algo_name == "MAPPO":
        args.use_centralized_V = True
        args.use_compact_share_obs = False
        args.hidden_size = 64
        args.ppo_epoch = 4
        args.clip_param = 0.15
        args.lr = 1.8e-4
        args.critic_lr = 2.8e-4
        args.entropy_coef = 0.006
        args.value_loss_coef = 0.6
        args.max_grad_norm = 0.45
        args.lr_decay_episode = 950
        args.entropy_decay_episode = 900

    elif algo_name == "IPPO":
        args.entropy_coef = 0.015

    elif algo_name == "IA2C":
        args.clip_param = 1e6   # no clipping = A2C
        args.entropy_coef = 0.02

    elif algo_name == "IQL":
        args.ppo_epoch = 1
        args.entropy_coef = 0.03

    return args


# ===================================================================
#  On-Policy Runner (with linear LR & entropy_coef decay)
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

        # LR decay config: linear from lr_init -> lr_end
        self.lr_init = args.lr
        self.critic_lr_init = getattr(args, 'critic_lr', args.lr)
        self.lr_end = getattr(args, 'lr_end', 1e-6)
        self.critic_lr_end = getattr(args, 'critic_lr_end', self.lr_end)
        self.lr_hold_episode = getattr(args, 'lr_hold_episode', 0)
        self.lr_decay_episode = getattr(args, 'lr_decay_episode', None)

        # Entropy coef decay: linear from init -> 0
        self.entropy_coef_init = args.entropy_coef
        self.entropy_coef_end = getattr(args, 'entropy_coef_end', 0.0)
        self.entropy_hold_episode = getattr(args, 'entropy_hold_episode', 0)
        self.entropy_decay_episode = getattr(args, 'entropy_decay_episode', None)

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
        # No scheduler - we do manual linear decay in _episode()

    def run(self, num_episodes, seed=0):
        rews_curve, loss_curve, ent_curve = [], [], []
        t0 = time.time()
        use_bar = tqdm is not None
        pbar = tqdm(total=num_episodes, desc=self.algo_name, ncols=120) if use_bar else None

        early_stopper = EarlyStopper(
            eval_window=200, patience=500, var_threshold=5.0,
            min_episode=5000, save_dir=SAVE_DIR,
            algo_name=self.algo_name, seed=seed)

        for ep in range(num_episodes):
            r, info = self._episode(ep, num_episodes)
            rews_curve.append(r)
            loss_curve.append(info.get('value_loss', 0.0))
            ent_curve.append(info.get('dist_entropy', 0.0))

            if pbar:
                avg = np.mean(rews_curve[-100:])
                pbar.update(1)
                pbar.set_postfix({
                    'R': '%.1f' % r, 'A': '%.1f' % avg,
                    'L': '%.4f' % info.get('value_loss', 0),
                    'H': '%.3f' % info.get('dist_entropy', 0),
                    'std': '%.3f' % math.exp(info.get('log_std_mean', -0.5)),
                    'lr': '%.1e' % info.get('current_lr', 0),
                    'ec': '%.4f' % info.get('current_ec', 0),
                }, refresh=False)
            elif (ep+1) % 500 == 0:
                avg = np.mean(rews_curve[-100:])
                eta = (time.time()-t0)/(ep+1)*(num_episodes-ep-1)/60
                print("  [%s] Ep%d R=%.1f A100=%.1f L=%.4f H=%.3f std=%.3f lr=%.1e ec=%.4f ETA %.1fmin" % (
                    self.algo_name, ep+1, r, avg,
                    info.get('value_loss',0), info.get('dist_entropy',0),
                    math.exp(info.get('log_std_mean', -0.5)),
                    info.get('current_lr', 0),
                    info.get('current_ec', 0), eta))

            should_stop = early_stopper.update(rews_curve)
            if should_stop:
                if pbar:
                    pbar.close()
                print("  [%s] Early stop at Ep%d (best_avg=%.1f at Ep%d)" % (
                    self.algo_name, ep+1, early_stopper.best_avg, early_stopper.best_episode))
                break

        if pbar and not pbar.disable:
            pbar.close()
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

        # ========== Linear LR Decay ==========
        # progress: 0.0 -> 1.0 over training
        lr_horizon = self.lr_decay_episode or num_episodes
        lr_start = min(self.lr_hold_episode, lr_horizon - 1)
        lr_progress = min(max((ep_idx - lr_start) / max(lr_horizon - lr_start - 1, 1), 0.0), 1.0)
        new_actor_lr = self.lr_init * (1.0 - lr_progress) + self.lr_end * lr_progress
        new_critic_lr = self.critic_lr_init * (1.0 - lr_progress) + self.critic_lr_end * lr_progress
        for pg in self.policy.actor_optimizer.param_groups:
            pg['lr'] = new_actor_lr
        for pg in self.policy.critic_optimizer.param_groups:
            pg['lr'] = new_critic_lr

        # ========== Linear Entropy Coef Decay ==========
        entropy_horizon = self.entropy_decay_episode or num_episodes
        entropy_start = min(self.entropy_hold_episode, entropy_horizon - 1)
        entropy_progress = min(max((ep_idx - entropy_start) / max(entropy_horizon - entropy_start - 1, 1), 0.0), 1.0)
        new_ec = self.entropy_coef_init * (1.0 - entropy_progress) + self.entropy_coef_end * entropy_progress
        self.trainer.entropy_coef = max(new_ec, 0.0)

        # Add scheduling info to train_info for logging
        train_info['current_lr'] = new_actor_lr
        train_info['current_ec'] = self.trainer.entropy_coef

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
#  Training entry
# ===================================================================
def train_algorithm(algo_name, num_episodes, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    args = get_algo_config(algo_name)
    envs, n_ag = make_env(seed=seed, episode_length=args.episode_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = OnPolicyRunner(args, envs, algo_name, device, n_ag)
    rews, losses, ents = runner.run(num_episodes, seed=seed)
    envs.close()
    return rews, losses, ents


# ===================================================================
#  Main
# ===================================================================
def main():
    global SAVE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_episodes', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5])
    ap.add_argument('--algorithms', type=str, nargs='+',
                    default=['Advanced-MAPPO','MAPPO','IPPO','IA2C','IQL'])
    ap.add_argument('--save_dir', type=str, default=SAVE_DIR,
                    help='Directory for reward/loss/entropy .npy files')
    cmd = ap.parse_args()
    SAVE_DIR = os.path.abspath(cmd.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("="*60)
    print("  Simple Converge v7 - IEEE TASE Final")
    print("  Algos: " + str(cmd.algorithms))
    print("  Episodes: %d, Seeds: %s" % (cmd.num_episodes, cmd.seeds))
    print("  Save dir: %s" % SAVE_DIR)
    print("  LR decay: linear -> 0")
    print("  Entropy coef decay: linear -> 0")
    print("  No MADDPG (deterministic, not comparable)")
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

    # Summary table
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

    print("\n  Data saved to: %s" % SAVE_DIR)
    print("  Run plot_results_swarm_attack.py to generate standalone IEEE-style figures.")


if __name__ == '__main__':
    main()
