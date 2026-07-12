"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
参考: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017

核心思想:
- 集中训练、分散执行 (CTDE)
- 确定性策略 + Ornstein-Uhlenbeck 噪声探索
- 集中式Critic，接收所有agent的obs和action
- 经验回放缓冲区 (off-policy)

本实现复用on-policy框架的接口，但内部使用off-policy的DDPG更新
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.utils.util import get_gard_norm
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check


# ====================== OU 噪声过程 ======================
class OUNoise:
    """Ornstein-Uhlenbeck 噪声过程，用于连续动作空间的时间相关探索"""
    def __init__(self, action_dim, n_agents, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.mu = mu
        self.theta = theta
        self.sigma_init = sigma
        self.sigma = sigma
        self.state = np.ones((n_agents, action_dim)) * mu

    def reset(self):
        self.state = np.ones((self.n_agents, self.action_dim)) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.n_agents, self.action_dim)
        self.state += dx
        return self.state.copy()

    def set_sigma(self, sigma):
        self.sigma = sigma


class MADDPGReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity, obs_dim, share_obs_dim, act_dim, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((capacity, n_agents, share_obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_share_obs = np.zeros((capacity, n_agents, share_obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, n_agents, 1), dtype=np.float32)

    def add(self, obs, share_obs, actions, rewards, next_obs, next_share_obs, masks):
        """存入单个transition"""
        self.obs[self.ptr] = obs
        self.share_obs[self.ptr] = share_obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.next_share_obs[self.ptr] = next_share_obs
        self.masks[self.ptr] = masks
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs], self.share_obs[idxs], self.actions[idxs],
            self.rewards[idxs], self.next_obs[idxs], self.next_share_obs[idxs],
            self.masks[idxs]
        )


class MADDPGActor(nn.Module):
    """MADDPG确定性Actor网络"""
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()  # action range [-1, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.modules())) - 2:  # last linear before Tanh
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        return self.net(obs)


class MADDPGCritic(nn.Module):
    """MADDPG集中式Critic: Q(s_all, a_all)"""
    def __init__(self, share_obs_dim, total_act_dim, hidden_size=64):
        super().__init__()
        input_dim = share_obs_dim + total_act_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, share_obs, all_actions):
        x = torch.cat([share_obs, all_actions], dim=-1)
        return self.net(x)


class MADDPGPolicy:
    """MADDPG策略类，管理Actor和Critic网络"""
    def __init__(self, obs_dim, share_obs_dim, act_dim, n_agents,
                 hidden_size=64, lr=1e-3, device=torch.device("cpu"), weight_decay=1e-4):
        self.device = device
        self.n_agents = n_agents
        self.act_dim = act_dim

        # Actor: 每个agent共享一个actor (parameter sharing)
        self.actor = MADDPGActor(obs_dim, act_dim, hidden_size).to(device)
        self.actor_target = MADDPGActor(obs_dim, act_dim, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic: 集中式
        total_act_dim = act_dim * n_agents
        self.critic = MADDPGCritic(share_obs_dim, total_act_dim, hidden_size).to(device)
        self.critic_target = MADDPGCritic(share_obs_dim, total_act_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)

    def select_action(self, obs, noise_scale=0.1):
        """选择动作 (带探索噪声)"""
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()
        noise = noise_scale * np.random.randn(*action.shape)
        return np.clip(action + noise, -1.0, 1.0)

    def select_action_deterministic(self, obs):
        """确定性动作 (无噪声)"""
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()
        return action

    def soft_update(self, tau=0.01):
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class RunningMeanStd:
    """Running mean/std for reward normalization."""
    def __init__(self, shape=(), clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)


class MADDPG:
    """MADDPG训练器 —— 适配on-policy框架接口"""
    def __init__(self, args, obs_dim, share_obs_dim, act_dim, n_agents, device=torch.device("cpu")):
        self.device = device
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.gamma = getattr(args, 'gamma', 0.95)
        self.tau = getattr(args, 'tau', 0.005)
        self.batch_size = getattr(args, 'maddpg_batch_size', 256)
        self.updates_per_step = getattr(args, 'maddpg_updates_per_step', 2)

        # OU 噪声配置
        self.noise_scale_init = getattr(args, 'maddpg_noise_scale', 0.2)
        self.noise_scale = self.noise_scale_init
        self.noise_decay = getattr(args, 'maddpg_noise_decay', 0.998)
        self.min_noise = getattr(args, 'maddpg_min_noise', 0.01)
        self.ou_noise = OUNoise(act_dim, n_agents,
                                mu=0.0, theta=0.15,
                                sigma=self.noise_scale_init)

        hidden_size = getattr(args, 'hidden_size', 64)
        lr = getattr(args, 'lr', 1e-3)
        self.init_lr = lr
        # LR decay
        self.use_lr_decay = getattr(args, 'maddpg_use_lr_decay', False)

        # Weight decay for stability
        weight_decay = getattr(args, 'maddpg_weight_decay', 1e-4)

        self.policy = MADDPGPolicy(obs_dim, share_obs_dim, act_dim, n_agents,
                                   hidden_size, lr, device, weight_decay=weight_decay)

        # 经验回放 — 更大的buffer
        buffer_capacity = getattr(args, 'maddpg_buffer_capacity', 200000)
        self.buffer = MADDPGReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=obs_dim,
            share_obs_dim=share_obs_dim,
            act_dim=act_dim,
            n_agents=n_agents
        )

        # Reward normalization for MADDPG (prevents huge critic loss)
        self.reward_rms = RunningMeanStd()

        # 记录最近一次更新的指标
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_td_error = 0.0

        # dummy value_normalizer for interface compatibility
        self.value_normalizer = None

    def store_transition(self, obs, share_obs, actions, rewards, next_obs, next_share_obs, masks):
        # Update reward running stats and normalize before storing
        self.reward_rms.update(rewards)
        norm_rewards = self.reward_rms.normalize(rewards).astype(np.float32)
        self.buffer.add(obs, share_obs, actions, norm_rewards, next_obs, next_share_obs, masks)

    def select_action_with_ou(self, obs):
        """使用 OU 噪声选择动作"""
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action = self.policy.actor(obs_t).cpu().numpy()
        noise = self.ou_noise.sample() * self.noise_scale
        return np.clip(action + noise, -1.0, 1.0)

    def decay_noise(self, ep_idx=0, anneal_episode=2000):
        """衰减噪声幅度 —— smooth exponential decay"""
        self.noise_scale = max(self.noise_scale * self.noise_decay, self.min_noise)
        self.ou_noise.set_sigma(self.noise_scale)

    def decay_lr(self, ep_idx, total_episodes, anneal_episode=2000):
        """LR衰减 —— anneal_episode后步进骤降一个数量级"""
        if not self.use_lr_decay:
            return
        if ep_idx < anneal_episode:
            frac = max(1.0 - 0.5 * ep_idx / float(anneal_episode), 0.5)
        else:
            remaining_frac = (ep_idx - anneal_episode) / max(total_episodes - anneal_episode, 1)
            frac = 0.1 * max(1.0 - 0.5 * remaining_frac, 0.5)
        new_lr = self.init_lr * frac
        for pg in self.policy.actor_optimizer.param_groups:
            pg['lr'] = new_lr
        for pg in self.policy.critic_optimizer.param_groups:
            pg['lr'] = new_lr

    def update(self):
        """执行多次MADDPG更新 (updates_per_step)"""
        if self.buffer.size < self.batch_size:
            return {'value_loss': 0, 'policy_loss': 0, 'td_error': 0}

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_td_error = 0.0

        for _ in range(self.updates_per_step):
            info = self._single_update()
            total_critic_loss += info['value_loss']
            total_actor_loss += info['policy_loss']
            total_td_error += info['td_error']

        n = self.updates_per_step
        self.last_critic_loss = total_critic_loss / n
        self.last_actor_loss = total_actor_loss / n
        self.last_td_error = total_td_error / n

        return {
            'value_loss': self.last_critic_loss,
            'policy_loss': self.last_actor_loss,
            'td_error': self.last_td_error,
        }

    def _single_update(self):
        """执行单次MADDPG更新"""
        obs, share_obs, actions, rewards, next_obs, next_share_obs, masks = \
            self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        share_obs_t = torch.FloatTensor(share_obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        next_share_obs_t = torch.FloatTensor(next_share_obs).to(self.device)
        masks_t = torch.FloatTensor(masks).to(self.device)

        B, N = obs_t.shape[0], obs_t.shape[1]

        # ===== Critic Update =====
        next_actions_all = []
        for i in range(N):
            next_actions_all.append(self.policy.actor_target(next_obs_t[:, i]))
        next_actions_cat = torch.cat(next_actions_all, dim=-1)

        next_share = next_share_obs_t[:, 0]
        share = share_obs_t[:, 0]

        target_q = self.policy.critic_target(next_share, next_actions_cat)
        avg_rewards = rewards_t.mean(dim=1)
        avg_masks = masks_t.min(dim=1)[0]
        target_q = avg_rewards + self.gamma * avg_masks * target_q.detach()
        # Clip target to prevent extreme values
        target_q = torch.clamp(target_q, -50.0, 50.0)

        actions_cat = actions_t.view(B, -1)
        current_q = self.policy.critic(share, actions_cat)
        td_error = (current_q - target_q).abs().mean().item()
        critic_loss = F.mse_loss(current_q, target_q)

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
        self.policy.critic_optimizer.step()

        # ===== Actor Update =====
        curr_actions_all = []
        for i in range(N):
            curr_actions_all.append(self.policy.actor(obs_t[:, i]))
        curr_actions_cat = torch.cat(curr_actions_all, dim=-1)

        actor_loss = -self.policy.critic(share, curr_actions_cat).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
        self.policy.actor_optimizer.step()

        # Soft update target networks
        self.policy.soft_update(self.tau)

        return {
            'value_loss': critic_loss.item(),
            'policy_loss': actor_loss.item(),
            'td_error': td_error,
        }

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
