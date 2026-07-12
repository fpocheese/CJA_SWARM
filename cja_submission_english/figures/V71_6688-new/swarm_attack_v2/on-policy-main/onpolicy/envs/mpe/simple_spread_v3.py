"""
Simple Spread 环境 V3 — 最终稳定版
设计原则:
1. 全局奖励 (团队奖励): 让centralized critic更容易学习
2. 密集shaping: 每步距离改善
3. 协调bonus: 鼓励不同agent占领不同landmark
4. 奖励scale适中: 不会太大导致不稳定
"""
import gym
from gym import spaces
import numpy as np


class SimpleSpreadEnvV3(gym.Env):
    """最终版 Simple Spread — 面向收敛性优化"""

    def __init__(self, num_agents=3, num_landmarks=3, episode_length=25):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.episode_length = episode_length

        # 物理参数
        self.world_size = 1.0
        self.agent_size = 0.15
        self.dt = 0.1
        self.damping = 0.25
        self.sensitivity = 5.0
        self.max_speed = 1.0
        self.current_step = 0

        # 状态
        self.agent_pos = np.zeros((num_agents, 2))
        self.agent_vel = np.zeros((num_agents, 2))
        self.landmark_pos = np.zeros((num_landmarks, 2))
        self.prev_min_dists = None

        # 观测: vel(2) + pos(2) + landmark_rel(N_l*2) + other_rel((N_a-1)*2)
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
        # 随机初始化 — 适中的范围
        for i in range(self.num_agents):
            self.agent_pos[i] = np.random.uniform(-0.8, 0.8, 2)
            self.agent_vel[i] = np.zeros(2)
        for i in range(self.num_landmarks):
            self.landmark_pos[i] = np.random.uniform(-0.8, 0.8, 2)

        self.prev_min_dists = self._min_dists_per_landmark()
        return [self._get_obs(i) for i in range(self.num_agents)]

    def step(self, action_n, *args):
        self.current_step += 1

        # 物理更新
        for i in range(self.num_agents):
            a = np.array(action_n[i], dtype=np.float32).flatten()[:2]
            a = np.clip(a, -1.0, 1.0)
            force = a * self.sensitivity
            self.agent_vel[i] = self.agent_vel[i] * (1 - self.damping) + force * self.dt
            speed = np.linalg.norm(self.agent_vel[i])
            if speed > self.max_speed:
                self.agent_vel[i] = self.agent_vel[i] / speed * self.max_speed
            self.agent_pos[i] += self.agent_vel[i] * self.dt
            self.agent_pos[i] = np.clip(self.agent_pos[i], -1.5, 1.5)

        # 全局团队奖励 — 所有agent拿相同的r
        reward = self._compute_reward()

        obs_n, reward_n, done_n, info_n = [], [], [], []
        done = self.current_step >= self.episode_length
        for i in range(self.num_agents):
            obs_n.append(self._get_obs(i))
            reward_n.append(reward)
            done_n.append(done)
            info_n.append({})
        return obs_n, reward_n, done_n, info_n

    def _get_obs(self, idx):
        obs = []
        obs.append(self.agent_vel[idx])
        obs.append(self.agent_pos[idx])
        for j in range(self.num_landmarks):
            obs.append(self.landmark_pos[j] - self.agent_pos[idx])
        for j in range(self.num_agents):
            if j != idx:
                obs.append(self.agent_pos[j] - self.agent_pos[idx])
        return np.concatenate(obs).astype(np.float32)

    def _min_dists_per_landmark(self):
        """每个landmark到其最近agent的距离"""
        dists = np.zeros(self.num_landmarks)
        for j in range(self.num_landmarks):
            d = [np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                 for i in range(self.num_agents)]
            dists[j] = min(d)
        return dists

    def _compute_reward(self):
        """
        奖励 = shaping + 覆盖bonus + 碰撞惩罚
        所有agent共享同一个team reward (有利于centralized critic)
        """
        curr_dists = self._min_dists_per_landmark()

        # 1. 距离shaping: 鼓励agent靠近landmark
        #    -sum(min_dist) 是标准MPE的奖励
        dist_rew = -np.sum(curr_dists)

        # 2. 距离改善shaping: 每步比上一步更好就给正信号
        improvement = np.sum(self.prev_min_dists - curr_dists)
        shaping_rew = improvement * 10.0  # 放大让学习信号更强

        # 3. 覆盖bonus: 每个landmark被"覆盖"（dist<0.1）给大奖励
        cover_rew = 0.0
        for j in range(self.num_landmarks):
            if curr_dists[j] < 0.1:
                cover_rew += 5.0

        # 4. 碰撞惩罚
        collision_pen = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if d < self.agent_size * 2:
                    collision_pen -= 1.0

        total = dist_rew + shaping_rew + cover_rew + collision_pen

        self.prev_min_dists = curr_dists.copy()
        return total

    def close(self):
        pass

    def render(self, mode='human'):
        pass
