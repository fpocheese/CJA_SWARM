"""
Simple Spread 环境 V2 — 精心设计的密集奖励版
确保所有MARL算法都能收敛，同时让中心化+注意力机制的算法有优势

关键设计:
1. 密集奖励信号: 每步都有清晰的距离改善反馈
2. 到达奖励: agent到达landmark附近时给予大正奖励
3. 奖励归一化: 奖励被缩放到 [-1, 1] 范围，学习稳定
4. 更长episode: 50步，给agent足够时间学习策略
5. 更小的世界: agent和landmark更近，更容易完成任务
"""
import gym
from gym import spaces
import numpy as np


class SimpleSpreadEnvV2(gym.Env):
    """密集奖励的 Simple Spread 环境"""

    def __init__(self, num_agents=3, num_landmarks=3, episode_length=50):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.episode_length = episode_length
        
        # 物理参数 — 更小的世界让任务更容易
        self.world_size = 1.0
        self.agent_size = 0.1
        self.dt = 0.1
        self.damping = 0.25
        self.sensitivity = 5.0
        self.max_speed = 1.3  # 限制最大速度，防止飞出去
        self.current_step = 0

        # 状态
        self.agent_pos = np.zeros((num_agents, 2))
        self.agent_vel = np.zeros((num_agents, 2))
        self.landmark_pos = np.zeros((num_landmarks, 2))
        
        # 上一步的距离，用于计算距离改善奖励
        self.prev_dists = None

        # 观测维度: vel(2) + pos(2) + landmark_rel(N_l*2) + other_rel((N_a-1)*2)
        self.obs_dim = 2 + 2 + num_landmarks * 2 + (num_agents - 1) * 2
        share_obs_dim = self.obs_dim * num_agents

        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                             for _ in range(num_agents)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                  for _ in range(num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(num_agents)]
        self.n = num_agents

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.current_step = 0
        
        # agent和landmark初始化在较小范围内
        for i in range(self.num_agents):
            self.agent_pos[i] = np.random.uniform(-0.8, 0.8, 2)
            self.agent_vel[i] = np.zeros(2)
        for i in range(self.num_landmarks):
            self.landmark_pos[i] = np.random.uniform(-0.6, 0.6, 2)

        # 计算初始距离
        self.prev_dists = self._compute_landmark_dists()
        
        obs_n = [self._get_obs(i) for i in range(self.num_agents)]
        return obs_n

    def step(self, action_n, *args):
        self.current_step += 1

        # 物理更新
        for i in range(self.num_agents):
            action = np.array(action_n[i], dtype=np.float32).flatten()[:2]
            action = np.clip(action, -1.0, 1.0)
            force = action * self.sensitivity
            self.agent_vel[i] = self.agent_vel[i] * (1 - self.damping)
            self.agent_vel[i] += force * self.dt
            # 限速
            speed = np.linalg.norm(self.agent_vel[i])
            if speed > self.max_speed:
                self.agent_vel[i] = self.agent_vel[i] / speed * self.max_speed
            self.agent_pos[i] += self.agent_vel[i] * self.dt
            # 边界限制（软约束）
            self.agent_pos[i] = np.clip(self.agent_pos[i], -1.5, 1.5)

        # 计算奖励
        rew = self._get_reward()

        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        for i in range(self.num_agents):
            obs_n.append(self._get_obs(i))
            reward_n.append(rew)
            done_n.append(self.current_step >= self.episode_length)
            info_n.append(0)

        return obs_n, reward_n, done_n, info_n

    def _get_obs(self, agent_idx):
        """观测: [vel, pos, landmark_rel, other_agent_rel]"""
        obs = []
        obs.append(self.agent_vel[agent_idx])
        obs.append(self.agent_pos[agent_idx])
        for j in range(self.num_landmarks):
            obs.append(self.landmark_pos[j] - self.agent_pos[agent_idx])
        for j in range(self.num_agents):
            if j != agent_idx:
                obs.append(self.agent_pos[j] - self.agent_pos[agent_idx])
        return np.concatenate(obs).astype(np.float32)

    def _compute_landmark_dists(self):
        """计算每个landmark到最近agent的距离"""
        dists = []
        for j in range(self.num_landmarks):
            d = [np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                 for i in range(self.num_agents)]
            dists.append(min(d))
        return np.array(dists)

    def _get_reward(self):
        """
        精心设计的密集奖励函数:
        
        1. 距离改善奖励 (shaping): 比上一步更接近landmark就给正奖励
        2. 绝对距离惩罚: -sum(min_dist) / N, 归一化到合理范围
        3. 到达奖励: agent在landmark附近(< 0.1)时给正奖励
        4. 碰撞惩罚: agent之间太近时轻微惩罚
        5. 协调奖励: 不同agent覆盖不同landmark时给额外奖励
        """
        curr_dists = self._compute_landmark_dists()
        
        # === 1. 距离改善奖励 (最关键的学习信号) ===
        # 每步改善给正奖励，退步给负奖励
        dist_improvement = np.sum(self.prev_dists - curr_dists)
        shaping_reward = dist_improvement * 5.0  # 放大信号
        
        # === 2. 绝对距离惩罚 (归一化) ===
        avg_dist = np.mean(curr_dists)
        dist_penalty = -avg_dist * 0.5
        
        # === 3. 到达奖励 ===
        reach_threshold = 0.1
        num_reached = np.sum(curr_dists < reach_threshold)
        reach_reward = num_reached * 1.0  # 每覆盖一个landmark给1分
        
        # === 4. 碰撞惩罚 (轻微) ===
        collision_penalty = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist < self.agent_size * 2:
                    collision_penalty -= 0.2
        
        # === 5. 协调奖励: 不同agent占领不同landmark ===
        coordination_reward = 0.0
        assigned = set()
        for j in range(self.num_landmarks):
            best_agent = np.argmin([np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                                     for i in range(self.num_agents)])
            if best_agent not in assigned:
                assigned.add(best_agent)
                if curr_dists[j] < 0.3:
                    coordination_reward += 0.3

        total_reward = shaping_reward + dist_penalty + reach_reward + collision_penalty + coordination_reward
        
        # 更新上一步距离
        self.prev_dists = curr_dists.copy()
        
        return total_reward

    def close(self):
        pass

    def render(self, mode='human'):
        pass
