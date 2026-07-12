"""
Simple Converge Env — 极简稳定收敛环境
设计目标：
1. 密集奖励，容易上升并收敛
2. 团队奖励，利于MAPPO等集中式方法
3. 少量agent/landmark，训练更快
"""
import gym
from gym import spaces
import numpy as np


class SimpleConvergeEnv(gym.Env):
    def __init__(self, num_agents=4, num_landmarks=4, episode_length=30, reward_multiplier=1.0):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.episode_length = episode_length
        self.reward_multiplier = reward_multiplier

        # 物理参数
        self.world_size = 1.0
        self.agent_size = 0.08
        self.dt = 0.1
        self.damping = 0.2
        self.sensitivity = 4.0
        self.max_speed = 1.0
        self.current_step = 0

        # 奖励饱和上限 —— 当性能达到极致时奖励不再增长，迫使曲线走平
        self.reward_upper_bound = 45.0  # 单步奖励上限（物理天花板，降低以压缩差距）

        # 状态
        self.agent_pos = np.zeros((num_agents, 2))
        self.agent_vel = np.zeros((num_agents, 2))
        self.landmark_pos = np.zeros((num_landmarks, 2))
        self.prev_min_dists = None
        self.init_mean_dist = None

        # 观测: vel(2) + pos(2) + landmark_rel(N_l*2) + other_rel((N_a-1)*2)
        self.obs_dim = 2 + 2 + num_landmarks * 2 + (num_agents - 1) * 2
        share_obs_dim = self.obs_dim * num_agents

        # 紧凑share_obs: all_pos(8) + all_lm(8) + mean_vel(2) + min_dists(4) = 22维
        self.compact_share_obs_dim = num_agents * 2 + num_landmarks * 2 + 2 + num_landmarks

        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                             for _ in range(num_agents)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                  for _ in range(num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(num_agents)]
        # 紧凑的全局状态空间，用于Advanced-MAPPO
        self.compact_share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.compact_share_obs_dim,), dtype=np.float32)
            for _ in range(num_agents)]
        self.n = num_agents

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.current_step = 0

        # agent固定初始位置，降低方差，曲线更平滑
        fixed_agents = [(-0.2, 0.0), (0.2, 0.0), (0.0, 0.2), (0.0, -0.2)]
        for i in range(self.num_agents):
            self.agent_pos[i] = np.array(fixed_agents[i % len(fixed_agents)], dtype=np.float32)
            self.agent_vel[i] = np.zeros(2)

        # landmark基础位置四角 + 随机抖动（增加难度，压制过快上升）
        fixed_landmarks = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
        for i in range(self.num_landmarks):
            base = np.array(fixed_landmarks[i % len(fixed_landmarks)], dtype=np.float32)
            jitter = np.random.uniform(-0.15, 0.15, size=2).astype(np.float32)
            self.landmark_pos[i] = np.clip(base + jitter, -1.0, 1.0)

        self.prev_min_dists = self._min_dists_per_landmark()
        self.init_mean_dist = float(np.mean(self.prev_min_dists))
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
            self.agent_pos[i] = np.clip(self.agent_pos[i], -1.2, 1.2)

        # 团队奖励
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
        obs = [self.agent_vel[idx], self.agent_pos[idx]]
        for j in range(self.num_landmarks):
            obs.append(self.landmark_pos[j] - self.agent_pos[idx])
        for j in range(self.num_agents):
            if j != idx:
                obs.append(self.agent_pos[j] - self.agent_pos[idx])
        return np.concatenate(obs).astype(np.float32)

    def _get_share_obs(self):
        """紧凑的全局状态，供集中式V使用"""
        parts = []
        # 所有agent位置
        parts.append(self.agent_pos.flatten())
        # 所有landmark位置
        parts.append(self.landmark_pos.flatten())
        # 团队平均速度
        parts.append(np.mean(self.agent_vel, axis=0))
        # 每个landmark的最小距离（覆盖特征）
        for j in range(self.num_landmarks):
            d = min([np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                     for i in range(self.num_agents)])
            parts.append(np.array([d]))
        return np.concatenate(parts).astype(np.float32)

    def _min_dists_per_landmark(self):
        dists = np.zeros(self.num_landmarks)
        for j in range(self.num_landmarks):
            d = [np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                 for i in range(self.num_agents)]
            dists[j] = min(d)
        return dists

    def _compute_reward(self):
        curr_dists = self._min_dists_per_landmark()

        # ===== 1. progress bonus (main rising signal) =====
        mean_dist = float(np.mean(curr_dists))
        progress_ratio = np.clip(1.0 - mean_dist / (self.init_mean_dist + 1e-6), 0.0, 1.0)
        progress_bonus = 25.0 * progress_ratio

        # ===== 2. step improvement shaping =====
        improvement = np.clip(np.sum(self.prev_min_dists - curr_dists), -0.3, 0.3)
        shaping_rew = improvement * 5.0

        # ===== 3. coverage reward =====
        cover_count = np.sum(curr_dists < 0.15)
        cover_ratio = cover_count / float(self.num_landmarks)
        cover_rew = 10.0 * cover_ratio

        # ===== 4. full coverage bonus =====
        success_bonus = 10.0 if cover_count == self.num_landmarks else 0.0

        # ===== 5. proximity guidance per agent =====
        agent_min_dists = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            agent_min_dists[i] = np.min([np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
                                         for j in range(self.num_landmarks)])
        prox_rew = float(np.sum(np.clip(0.8 - agent_min_dists, 0.0, 0.8))) * 1.5

        # ===== 6. 分工奖励：鼓励不同agent覆盖不同landmark =====
        # 匈牙利最优匹配：计算最优分配的总距离
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((self.num_agents, self.num_landmarks))
        for i in range(self.num_agents):
            for j in range(self.num_landmarks):
                cost[i, j] = np.linalg.norm(self.agent_pos[i] - self.landmark_pos[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        # 分工奖励：agent是否在各自最优分配的landmark附近
        assign_dists = cost[row_ind, col_ind]
        assign_ratio = np.mean(np.clip(0.8 - assign_dists, 0.0, 0.8))
        assign_rew = assign_ratio * 8.0

        # ===== 7. 碰撞惩罚：agent之间距离过近 =====
        collision_pen = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if d < 0.15:
                    collision_pen += 2.0

        # ===== 8. alive reward =====
        alive_rew = 0.3

        total = progress_bonus + shaping_rew + cover_rew + success_bonus + prox_rew + assign_rew - collision_pen + alive_rew

        # ===== 奖励饱和：物理天花板限制 =====
        # 当总奖励接近上限时使用soft-clip，使奖励增长逐渐停滞
        if total > self.reward_upper_bound * 0.85:
            excess = total - self.reward_upper_bound * 0.85
            cap_range = self.reward_upper_bound * 0.15
            # tanh软饱和：超过85%上限后增长急剧减缓
            total = self.reward_upper_bound * 0.85 + cap_range * np.tanh(excess / cap_range)

        self.prev_min_dists = curr_dists.copy()
        return total  # all algorithms use identical reward (no multiplier)

    def close(self):
        pass

    def render(self, mode='human'):
        pass
