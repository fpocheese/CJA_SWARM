#!/usr/bin/env python
"""
训练脚本：simple_world_comm 环境
FighterWorld: 20个防御无人机拦截8个进攻无人机
比较6种算法：MAPPO, Advanced-MAPPO, IPPO, IA2C, IQL, MADDPG
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mpe.MPE_env import MPEEnv

# ============================================================
# 算法配置
# ============================================================
def get_algo_config(algo_name):
    """获取算法专用配置"""
    base_config = {
        'hidden_size': 128,  # FighterWorld更复杂，用更大网络
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'ppo_epoch': 1,  # 关键：单环境用1
        'clip_param': 0.2,
        'use_centralized_V': True,
    }
    
    if algo_name == 'MAPPO':
        return {**base_config}
    
    elif algo_name == 'Advanced-MAPPO':
        return {
            **base_config,
            'use_attention': False,  # 单时间步attention无意义
            'use_residual': True,
            'use_dual_clip': False,
            'dual_clip_coef': 3.0,
            'use_adaptive_kl': False,
            'target_kl': 0.01,
        }
    
    elif algo_name == 'IPPO':
        return {
            **base_config,
            'use_centralized_V': False,  # 独立critic
        }
    
    elif algo_name == 'IA2C':
        return {
            **base_config,
            'use_centralized_V': False,
            'ppo_epoch': 1,
            'use_gae': False,  # A2C不用GAE
        }
    
    elif algo_name == 'IQL':
        return {
            'hidden_size': 128,
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 64,
            'target_update': 100,
        }
    
    elif algo_name == 'MADDPG':
        return {
            'hidden_size': 128,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.01,
            'buffer_size': 100000,
            'batch_size': 256,
            'noise_std': 0.1,
            'noise_decay': 0.9995,
        }
    
    return base_config

# ============================================================
# 网络模块
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x + residual)
        return torch.relu(x)

# ============================================================
# PPO系列 Actor-Critic
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, config, global_obs_dim=None):
        super().__init__()
        h = config['hidden_size']
        self.use_centralized_V = config.get('use_centralized_V', True)
        use_residual = config.get('use_residual', False)
        
        # Actor
        if use_residual:
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, h),
                nn.ReLU(),
                ResidualBlock(h),
                nn.Linear(h, act_dim)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, act_dim)
            )
        
        # Critic
        critic_input = global_obs_dim if (self.use_centralized_V and global_obs_dim) else obs_dim
        if use_residual:
            self.critic = nn.Sequential(
                nn.Linear(critic_input, h),
                nn.ReLU(),
                ResidualBlock(h),
                nn.Linear(h, 1)
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(critic_input, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 1)
            )
    
    def forward(self, obs, global_obs=None):
        logits = self.actor(obs)
        if self.use_centralized_V and global_obs is not None:
            value = self.critic(global_obs)
        else:
            value = self.critic(obs)
        return logits, value
    
    def get_action(self, obs, global_obs=None, deterministic=False):
        logits, value = self.forward(obs, global_obs)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, obs, global_obs, action):
        logits, value = self.forward(obs, global_obs)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)

# ============================================================
# IQL 网络
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)

# ============================================================
# MADDPG 网络
# ============================================================
class MADDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
    
    def forward(self, obs):
        return torch.softmax(self.net(obs), dim=-1)

class MADDPGCritic(nn.Module):
    def __init__(self, global_obs_dim, global_act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + global_act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        return self.net(x)

# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, n_agents):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
    
    def add(self, obs, actions, rewards, next_obs, dones):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.dones[idx])
        )

# ============================================================
# PPO Trainer
# ============================================================
class PPOTrainer:
    def __init__(self, obs_dim, act_dim, n_agents, config, algo_name='MAPPO'):
        self.config = config
        self.n_agents = n_agents
        self.algo_name = algo_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        use_centralized = config.get('use_centralized_V', True)
        global_obs_dim = obs_dim * n_agents if use_centralized else None
        
        self.policies = []
        self.optimizers = []
        for i in range(n_agents):
            policy = ActorCritic(obs_dim, act_dim, config, global_obs_dim).to(self.device)
            optimizer = optim.Adam(policy.parameters(), lr=config['lr'])
            self.policies.append(policy)
            self.optimizers.append(optimizer)
    
    def get_actions(self, obs_list):
        actions, log_probs, values = [], [], []
        global_obs = torch.FloatTensor(np.concatenate(obs_list)).to(self.device)
        
        for i, policy in enumerate(self.policies):
            obs = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            g_obs = global_obs.unsqueeze(0) if self.config.get('use_centralized_V', True) else None
            
            with torch.no_grad():
                a, lp, v = policy.get_action(obs, g_obs)
            
            actions.append(a.item())
            log_probs.append(lp.item())
            values.append(v.item())
        
        return actions, log_probs, values
    
    def update(self, trajectories):
        """更新所有agent的策略"""
        total_loss = 0
        
        for i, policy in enumerate(self.policies):
            obs = torch.FloatTensor(trajectories['obs'][:, i]).to(self.device)
            global_obs = torch.FloatTensor(trajectories['global_obs']).to(self.device)
            actions = torch.LongTensor(trajectories['actions'][:, i]).to(self.device)
            old_log_probs = torch.FloatTensor(trajectories['log_probs'][:, i]).to(self.device)
            returns = torch.FloatTensor(trajectories['returns'][:, i]).to(self.device)
            advantages = torch.FloatTensor(trajectories['advantages'][:, i]).to(self.device)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.config['ppo_epoch']):
                g_obs = global_obs if self.config.get('use_centralized_V', True) else None
                log_probs, entropy, values = policy.evaluate(obs, g_obs, actions)
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.config['clip_param'], 1+self.config['clip_param']) * advantages
                
                # Dual clip (可选)
                if self.config.get('use_dual_clip', False):
                    dual_coef = self.config.get('dual_clip_coef', 3.0)
                    surr3 = dual_coef * advantages
                    policy_loss = -torch.where(
                        advantages >= 0,
                        torch.min(surr1, surr2),
                        torch.max(torch.min(surr1, surr2), surr3)
                    ).mean()
                else:
                    policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                loss = (policy_loss + 
                        self.config['value_loss_coef'] * value_loss + 
                        self.config['entropy_coef'] * entropy_loss)
                
                self.optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.config['max_grad_norm'])
                self.optimizers[i].step()
                
                total_loss += loss.item()
        
        return total_loss / self.n_agents

# ============================================================
# IQL Trainer
# ============================================================
class IQLTrainer:
    def __init__(self, obs_dim, act_dim, n_agents, config):
        self.config = config
        self.n_agents = n_agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = config['epsilon_start']
        
        self.q_nets = []
        self.target_nets = []
        self.optimizers = []
        
        for i in range(n_agents):
            q_net = QNetwork(obs_dim, act_dim, config['hidden_size']).to(self.device)
            target_net = QNetwork(obs_dim, act_dim, config['hidden_size']).to(self.device)
            target_net.load_state_dict(q_net.state_dict())
            optimizer = optim.Adam(q_net.parameters(), lr=config['lr'])
            
            self.q_nets.append(q_net)
            self.target_nets.append(target_net)
            self.optimizers.append(optimizer)
        
        self.buffer = ReplayBuffer(config['buffer_size'], obs_dim, act_dim, n_agents)
        self.update_count = 0
    
    def get_actions(self, obs_list):
        actions = []
        for i, q_net in enumerate(self.q_nets):
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, q_net.net[-1].out_features)
            else:
                obs = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = q_net(obs)
                action = q_values.argmax(dim=-1).item()
            actions.append(action)
        return actions, [0]*self.n_agents, [0]*self.n_agents
    
    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)
    
    def update(self):
        if self.buffer.size < self.config['batch_size']:
            return 0
        
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config['batch_size'])
        obs, actions, rewards = obs.to(self.device), actions.to(self.device), rewards.to(self.device)
        next_obs, dones = next_obs.to(self.device), dones.to(self.device)
        
        total_loss = 0
        for i in range(self.n_agents):
            q_values = self.q_nets[i](obs[:, i]).gather(1, actions[:, i].unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q = self.target_nets[i](next_obs[:, i]).max(dim=-1)[0]
                target = rewards[:, i] + self.config['gamma'] * next_q * (1 - dones[:, i])
            
            loss = nn.MSELoss()(q_values, target)
            
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            
            total_loss += loss.item()
        
        self.update_count += 1
        if self.update_count % self.config['target_update'] == 0:
            for i in range(self.n_agents):
                self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
        
        self.epsilon = max(self.config['epsilon_end'], 
                          self.epsilon * self.config['epsilon_decay'])
        
        return total_loss / self.n_agents

# ============================================================
# MADDPG Trainer  
# ============================================================
class MADDPGTrainer:
    def __init__(self, obs_dim, act_dim, n_agents, config):
        self.config = config
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_std = config['noise_std']
        
        global_obs_dim = obs_dim * n_agents
        global_act_dim = act_dim * n_agents
        
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            actor = MADDPGActor(obs_dim, act_dim, config['hidden_size']).to(self.device)
            critic = MADDPGCritic(global_obs_dim, global_act_dim, config['hidden_size']).to(self.device)
            target_actor = MADDPGActor(obs_dim, act_dim, config['hidden_size']).to(self.device)
            target_critic = MADDPGCritic(global_obs_dim, global_act_dim, config['hidden_size']).to(self.device)
            
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config['actor_lr']))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=config['critic_lr']))
        
        self.buffer = ReplayBuffer(config['buffer_size'], obs_dim, act_dim, n_agents)
    
    def get_actions(self, obs_list):
        actions = []
        for i, actor in enumerate(self.actors):
            obs = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = actor(obs).squeeze()
            
            # 添加噪声探索
            probs = probs.cpu().numpy()
            noise = np.random.randn(self.act_dim) * self.noise_std
            probs = probs + noise
            probs = np.exp(probs) / np.exp(probs).sum()
            action = np.random.choice(self.act_dim, p=probs)
            actions.append(action)
        
        return actions, [0]*self.n_agents, [0]*self.n_agents
    
    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.add(obs, actions, rewards, next_obs, dones)
    
    def update(self):
        if self.buffer.size < self.config['batch_size']:
            return 0
        
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config['batch_size'])
        obs, actions, rewards = obs.to(self.device), actions.to(self.device), rewards.to(self.device)
        next_obs, dones = next_obs.to(self.device), dones.to(self.device)
        
        batch_size = obs.shape[0]
        global_obs = obs.view(batch_size, -1)
        global_next_obs = next_obs.view(batch_size, -1)
        
        # One-hot actions
        actions_onehot = torch.zeros(batch_size, self.n_agents, self.act_dim).to(self.device)
        for i in range(self.n_agents):
            actions_onehot[:, i].scatter_(1, actions[:, i].unsqueeze(1), 1)
        global_actions = actions_onehot.view(batch_size, -1)
        
        total_loss = 0
        
        for i in range(self.n_agents):
            # Critic update
            with torch.no_grad():
                next_actions = []
                for j in range(self.n_agents):
                    next_act_probs = self.target_actors[j](next_obs[:, j])
                    next_actions.append(next_act_probs)
                next_actions = torch.stack(next_actions, dim=1).view(batch_size, -1)
                target_q = self.target_critics[i](global_next_obs, next_actions).squeeze()
                target = rewards[:, i] + self.config['gamma'] * target_q * (1 - dones[:, i])
            
            current_q = self.critics[i](global_obs, global_actions).squeeze()
            critic_loss = nn.MSELoss()(current_q, target)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # Actor update
            curr_actions = []
            for j in range(self.n_agents):
                if j == i:
                    curr_actions.append(self.actors[j](obs[:, j]))
                else:
                    curr_actions.append(actions_onehot[:, j])
            curr_actions = torch.stack(curr_actions, dim=1).view(batch_size, -1)
            
            actor_loss = -self.critics[i](global_obs, curr_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            total_loss += (critic_loss.item() + actor_loss.item())
        
        # Soft update targets
        tau = self.config['tau']
        for i in range(self.n_agents):
            for tp, p in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for tp, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        
        self.noise_std = max(0.01, self.noise_std * self.config['noise_decay'])
        
        return total_loss / self.n_agents

# ============================================================
# 环境创建
# ============================================================
def make_env(args):
    """创建 simple_world_comm 环境"""
    class EnvArgs:
        def __init__(self):
            self.scenario_name = 'simple_world_comm'
            self.num_agents = 20  # 防御无人机数量
            self.num_landmarks = 4  # food + aaa + bbb + ccc
            self.episode_length = 100  # 每episode步数
            self.use_discrete_action = True
    
    env_args = EnvArgs()
    env = MPEEnv(env_args)
    
    # 获取实际的观测和动作维度
    obs = env.reset()
    n_agents = len(obs)
    obs_dim = obs[0].shape[0]
    act_dim = env.action_space[0].n if hasattr(env.action_space[0], 'n') else 5
    
    return env, n_agents, obs_dim, act_dim

# ============================================================
# 训练函数
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算GAE"""
    n_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return returns, advantages

def train_ppo_episode(env, trainer, config, n_agents):
    """训练一个episode (PPO系列)"""
    obs = env.reset()
    episode_reward = 0
    
    # 收集轨迹
    traj = {
        'obs': [], 'global_obs': [], 'actions': [],
        'log_probs': [], 'values': [], 'rewards': [], 'dones': []
    }
    
    for step in range(config.get('episode_length', 100)):
        actions, log_probs, values = trainer.get_actions(obs)
        
        # 将离散动作转换为连续动作 (FighterWorld需要连续控制)
        continuous_actions = []
        for a in actions:
            # 将离散动作映射到连续空间 [-1, 1]
            u0 = (a // 5 - 2) * 0.5  # 第一个控制量
            u1 = (a % 5 - 2) * 0.5   # 第二个控制量
            continuous_actions.append([u0, u1])
        
        next_obs, rewards, dones, infos = env.step(continuous_actions, step)
        
        # 所有agent共享team reward
        team_reward = np.mean(rewards)
        episode_reward += team_reward
        
        traj['obs'].append(np.array(obs))
        traj['global_obs'].append(np.concatenate(obs))
        traj['actions'].append(actions)
        traj['log_probs'].append(log_probs)
        traj['values'].append(values)
        traj['rewards'].append([team_reward] * n_agents)
        traj['dones'].append([any(dones)] * n_agents)
        
        obs = next_obs
        
        if any(dones):
            break
    
    # 转换为numpy数组
    for k in traj:
        traj[k] = np.array(traj[k])
    
    # 计算returns和advantages
    returns_list = []
    advantages_list = []
    for i in range(n_agents):
        ret, adv = compute_gae(
            traj['rewards'][:, i],
            traj['values'][:, i],
            traj['dones'][:, i],
            config['gamma'],
            config.get('gae_lambda', 0.95)
        )
        returns_list.append(ret)
        advantages_list.append(adv)
    
    traj['returns'] = np.array(returns_list).T
    traj['advantages'] = np.array(advantages_list).T
    
    # 更新策略
    trainer.update(traj)
    
    return episode_reward

def train_iql_episode(env, trainer, config, n_agents):
    """训练一个episode (IQL)"""
    obs = env.reset()
    episode_reward = 0
    
    for step in range(config.get('episode_length', 100)):
        actions, _, _ = trainer.get_actions(obs)
        
        # 将离散动作转换为连续动作
        continuous_actions = []
        for a in actions:
            u0 = (a // 5 - 2) * 0.5
            u1 = (a % 5 - 2) * 0.5
            continuous_actions.append([u0, u1])
        
        next_obs, rewards, dones, infos = env.step(continuous_actions, step)
        
        team_reward = np.mean(rewards)
        episode_reward += team_reward
        
        trainer.store(
            np.array(obs),
            np.array(actions),
            np.array([team_reward] * n_agents),
            np.array(next_obs),
            np.array([float(any(dones))] * n_agents)
        )
        
        trainer.update()
        obs = next_obs
        
        if any(dones):
            break
    
    return episode_reward

def train_maddpg_episode(env, trainer, config, n_agents):
    """训练一个episode (MADDPG)"""
    obs = env.reset()
    episode_reward = 0
    
    for step in range(config.get('episode_length', 100)):
        actions, _, _ = trainer.get_actions(obs)
        
        # 将离散动作转换为连续动作
        continuous_actions = []
        for a in actions:
            u0 = (a // 5 - 2) * 0.5
            u1 = (a % 5 - 2) * 0.5
            continuous_actions.append([u0, u1])
        
        next_obs, rewards, dones, infos = env.step(continuous_actions, step)
        
        team_reward = np.mean(rewards)
        episode_reward += team_reward
        
        trainer.store(
            np.array(obs),
            np.array(actions),
            np.array([team_reward] * n_agents),
            np.array(next_obs),
            np.array([float(any(dones))] * n_agents)
        )
        
        trainer.update()
        obs = next_obs
        
        if any(dones):
            break
    
    return episode_reward

# ============================================================
# 主训练循环
# ============================================================
def train_algorithm(algo_name, num_episodes, seed, args):
    """训练单个算法"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = get_algo_config(algo_name)
    config['episode_length'] = 100
    
    env, n_agents, obs_dim, act_dim = make_env(args)
    
    print(f"\n  Environment: simple_world_comm")
    print(f"  Agents: {n_agents}, Obs: {obs_dim}, Act: {act_dim}")
    
    # 创建trainer
    if algo_name in ['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C']:
        trainer = PPOTrainer(obs_dim, act_dim, n_agents, config, algo_name)
        train_fn = train_ppo_episode
    elif algo_name == 'IQL':
        trainer = IQLTrainer(obs_dim, act_dim, n_agents, config)
        train_fn = train_iql_episode
    elif algo_name == 'MADDPG':
        trainer = MADDPGTrainer(obs_dim, act_dim, n_agents, config)
        train_fn = train_maddpg_episode
    
    # 训练循环
    all_rewards = []
    for ep in range(1, num_episodes + 1):
        reward = train_fn(env, trainer, config, n_agents)
        all_rewards.append(reward)
        
        if ep % 10 == 0 or ep == 1:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"  [{algo_name}] Ep {ep}/{num_episodes} | R={reward:.2f} | Avg100={avg_reward:.2f}")
    
    env.close()
    return np.array(all_rewards)

# ============================================================
# 绘图函数
# ============================================================
def plot_results(results_dir, algorithms):
    """绘制对比图"""
    plt.figure(figsize=(14, 5))
    
    colors = {
        'MAPPO': 'blue',
        'Advanced-MAPPO': 'red',
        'IPPO': 'green',
        'IA2C': 'orange',
        'IQL': 'purple',
        'MADDPG': 'brown'
    }
    
    # 子图1: 训练曲线
    plt.subplot(1, 2, 1)
    for algo in algorithms:
        files = [f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('.npy')]
        if not files:
            continue
        
        all_rewards = []
        for f in files:
            rewards = np.load(os.path.join(results_dir, f))
            all_rewards.append(rewards)
        
        all_rewards = np.array(all_rewards)
        mean_rewards = all_rewards.mean(axis=0)
        std_rewards = all_rewards.std(axis=0)
        
        # 平滑
        window = 10
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed))
        
        plt.plot(x, smoothed, label=algo, color=colors.get(algo, 'gray'), linewidth=2)
        
        smoothed_std = np.convolve(std_rewards, np.ones(window)/window, mode='valid')
        plt.fill_between(x, smoothed - smoothed_std, smoothed + smoothed_std,
                        alpha=0.2, color=colors.get(algo, 'gray'))
    
    plt.xlabel('Episode')
    plt.ylabel('Team Reward')
    plt.title('Simple World Comm: 20 Defense UAVs vs 8 Attack UAVs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 最终性能条形图
    plt.subplot(1, 2, 2)
    final_means = []
    final_stds = []
    algo_names = []
    
    for algo in algorithms:
        files = [f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('.npy')]
        if not files:
            continue
        
        all_finals = []
        for f in files:
            rewards = np.load(os.path.join(results_dir, f))
            all_finals.append(np.mean(rewards[-100:]))
        
        final_means.append(np.mean(all_finals))
        final_stds.append(np.std(all_finals))
        algo_names.append(algo)
    
    x_pos = np.arange(len(algo_names))
    bars = plt.bar(x_pos, final_means, yerr=final_stds, capsize=5,
                   color=[colors.get(a, 'gray') for a in algo_names])
    plt.xticks(x_pos, algo_names, rotation=45, ha='right')
    plt.ylabel('Final Reward (last 100 eps)')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_world_comm.png'), dpi=150)
    plt.savefig(os.path.join(results_dir, 'comparison_world_comm.pdf'))
    plt.close()
    
    print(f"  ✓ comparison_world_comm.pdf/png")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['MAPPO', 'Advanced-MAPPO', 'IPPO', 'IA2C', 'IQL', 'MADDPG'])
    parser.add_argument('--plot_only', action='store_true')
    args = parser.parse_args()
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'simple_world_comm')
    os.makedirs(results_dir, exist_ok=True)
    
    if args.plot_only:
        print("\nGenerating plots only...")
        plot_results(results_dir, args.algorithms)
        return
    
    print("="*60)
    print("  Simple World Comm Training")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Episodes: {args.num_episodes}, Seeds: {args.seeds}")
    print("="*60)
    
    start_time = time.time()
    total_runs = len(args.algorithms) * len(args.seeds)
    run_idx = 0
    
    for algo in args.algorithms:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'#'*60}")
            print(f"  [{run_idx}/{total_runs}] {algo} seed={seed}")
            print(f"{'#'*60}")
            
            print(f"\n{'='*60}")
            print(f"  {algo} | seed={seed} | {args.num_episodes} episodes")
            print(f"{'='*60}")
            
            t0 = time.time()
            rewards = train_algorithm(algo, args.num_episodes, seed, args)
            elapsed = (time.time() - t0) / 60
            
            # 保存结果
            save_path = os.path.join(results_dir, f"{algo}_seed{seed}_rewards.npy")
            np.save(save_path, rewards)
            
            final100 = np.mean(rewards[-100:])
            print(f"  Done in {elapsed:.1f}min | Final100={final100:.2f} | Saved: {save_path}")
    
    total_time = (time.time() - start_time) / 60
    print(f"\n  All done in {total_time:.1f} min")
    
    # 绘图
    print("\nGenerating plots...")
    plot_results(results_dir, args.algorithms)
    
    # 打印汇总
    print(f"\n  {'Algorithm':<20} {'Final(last100)':<20} {'Best':<10}")
    print(f"  {'-'*50}")
    
    for algo in args.algorithms:
        files = [f for f in os.listdir(results_dir) if f.startswith(f"{algo}_seed") and f.endswith('.npy')]
        if not files:
            continue
        
        all_finals = []
        all_best = []
        for f in files:
            rewards = np.load(os.path.join(results_dir, f))
            all_finals.append(np.mean(rewards[-100:]))
            all_best.append(np.max(rewards))
        
        mean_final = np.mean(all_finals)
        std_final = np.std(all_finals)
        best = np.max(all_best)
        
        print(f"  {algo:<20} {mean_final:.2f} ± {std_final:.2f}          {best:.2f}")

if __name__ == '__main__':
    main()
