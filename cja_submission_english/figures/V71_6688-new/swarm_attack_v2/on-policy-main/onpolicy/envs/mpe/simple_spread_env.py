"""
自包含的 Simple Spread 环境
不依赖被修改过的 environment.py 和 core.py (那些已被改为FighterWorld专用)

标准 MPE Simple Spread:
- N 个 agent 合作覆盖 N 个 landmark
- 连续动作空间 Box(2,), 范围 [-1, 1]
- 奖励 = -sum(min_dist_to_each_landmark) - collision_penalty
- 观测: [vel, pos, landmark_rel_pos, other_agent_rel_pos]
"""
import gym
from gym import spaces
import numpy as np


class SimpleSpreadEnv(gym.Env):
    """自包含的 Simple Spread 多智能体环境"""

    def __init__(self, num_agents=3, num_landmarks=3, episode_length=25,
                 world_size=1.0, agent_size=0.15, collision_penalty=1.0):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.episode_length = episode_length
        self.world_size = world_size
        self.agent_size = agent_size
        self.collision_penalty = collision_penalty
        self.dt = 0.1
        self.damping = 0.25
        self.max_speed = None  # no speed limit
        self.sensitivity = 5.0
        self.current_step = 0

        # Agent state: [pos_x, pos_y, vel_x, vel_y]
        self.agent_pos = np.zeros((num_agents, 2))
        self.agent_vel = np.zeros((num_agents, 2))
        self.landmark_pos = np.zeros((num_landmarks, 2))

        # Observation: vel(2) + pos(2) + landmark_rel(num_landmarks*2) + other_rel((num_agents-1)*2)
        self.obs_dim = 2 + 2 + num_landmarks * 2 + (num_agents - 1) * 2
        share_obs_dim = self.obs_dim * num_agents

        # 动作空间: 连续 [-1,1]^2
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                             for _ in range(num_agents)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                  for _ in range(num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(num_agents)]

        # 兼容属性
        self.n = num_agents

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.current_step = 0
        for i in range(self.num_agents):
            self.agent_pos[i] = np.random.uniform(-self.world_size, self.world_size, 2)
            self.agent_vel[i] = np.zeros(2)
        for i in range(self.num_landmarks):
            self.landmark_pos[i] = np.random.uniform(-0.8 * self.world_size, 0.8 * self.world_size, 2)

        obs_n = [self._get_obs(i) for i in range(self.num_agents)]
        return obs_n

    def step(self, action_n, *args):
        """
        action_n: list of (2,) arrays, 连续动作 [-1,1]^2
        兼容 DummyVecEnv 的 step(a, env) 调用
        """
        self.current_step += 1

        # 应用动作 (力) -> 更新速度 -> 更新位置
        for i in range(self.num_agents):
            action = np.array(action_n[i], dtype=np.float32).flatten()[:2]
            force = action * self.sensitivity
            # 阻尼
            self.agent_vel[i] = self.agent_vel[i] * (1 - self.damping)
            # 加速度 (mass=1)
            self.agent_vel[i] += force * self.dt
            # 更新位置
            self.agent_pos[i] += self.agent_vel[i] * self.dt

        # 计算观测、奖励、done
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        rew = self._get_reward()  # shared reward

        for i in range(self.num_agents):
            obs_n.append(self._get_obs(i))
            reward_n.append(rew)
            done_n.append(self.current_step >= self.episode_length)
            info_n.append(0)

        return obs_n, reward_n, done_n, info_n

    def _get_obs(self, agent_idx):
        """观测: [vel, pos, landmark_rel_pos, other_agent_rel_pos]"""
        obs = []
        obs.append(self.agent_vel[agent_idx])
        obs.append(self.agent_pos[agent_idx])
        # relative landmark positions
        for j in range(self.num_landmarks):
            obs.append(self.landmark_pos[j] - self.agent_pos[agent_idx])
        # relative other agent positions
        for j in range(self.num_agents):
            if j != agent_idx:
                obs.append(self.agent_pos[j] - self.agent_pos[agent_idx])
        return np.concatenate(obs).astype(np.float32)

    def _get_reward(self):
        """共享奖励: -sum(min_dist_to_each_landmark) - collision_penalty"""
        rew = 0.0
        # landmark coverage
        for j in range(self.num_landmarks):
            dists = [np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                     for i in range(self.num_agents)]
            rew -= min(dists)

        # collision penalty
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist < self.agent_size * 2:
                    rew -= self.collision_penalty

        return rew

    def close(self):
        pass

    def render(self, mode='human'):
        pass
