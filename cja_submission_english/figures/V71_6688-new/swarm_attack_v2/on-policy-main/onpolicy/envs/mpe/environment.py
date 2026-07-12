import gym
import math
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import random


# environment for all agents in the multiagent world创造多智能体的环境，也是对基类的一个继承
# currently code assumes that no agents will be created/destroyed at runtime!
## 当前代码假定在运行时不会创建/销毁智能体！
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.agents
        
        self.n = len(world.agents)
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters  环境的一些参数
        self.discrete_action_space = True
        # self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        ##如果为真，动作是一个数字 0...N，否则动作是一个一热的 N 维向量
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        ##如果为真，即使动作是连续的，动作也会离散执行
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        ##如果为真，则每个智能体都有相同的奖励
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.dt = 0.05

        # configure spaces  配置空间  一个是观测值得空间，一个是动作的空间
        self.action_space = []

        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim =0
        for agent in self.world.policy_agents:
            total_action_space = []

            action_dim = getattr(world, "action_dim", world.dim_p)
            u_action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])

            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            
            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]



        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]

        self._reset_render()

    def seed(self, seed=None):
        self.world.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def jieshu(self):
        self.agents123 = self.world.policy_agents
        self.agents111 = self.world.scripted_agents
        fanhui = False
        if self.is_collision(self.agents111[0],self.world.landmarks[0]) and self.is_collision(self.agents111[1],self.world.landmarks[0]):
            fanhui = True
        return fanhui


    def step(self, action_n, step=0):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        # info_n = {'n': []}

        self.agents123 = self.world.policy_agents
        self.agents111 = self.world.scripted_agents
        self.agents = self.world.agents
        # set action for each agent
        for i, agent in enumerate(self.agents123):
            self._set_action(action_n[i], agent, self.action_space[i])
        for i, agent in enumerate(self.agents111):
            if getattr(self.world, "use_agent_script_callback", False) and agent.action_callback is not None:
                agent.action = agent.action_callback(agent, self.world)
            else:
                self.action_callback1017(agent, self.world)
        # advance world state这就是我们那个场景里的step
        self.world.step()
        # record observation for each agent
        for agent in self.agents123:
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent,step))
            reward_n.append(self._get_reward(agent))

            if all(agent.doneflag for agent in self.agents123):
                info_n.append(1)
            else:
                info_n.append(0)

        return obs_n, reward_n, done_n , info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        # self.agents = self.world.policy_agents
        self.agents = self.world.policy_agents
        # for agent in self.world.policy_agents:
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        # obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    #我如果改成action_callback的话，那么返回的return就能是动作的列表
    def _get_done(self, agent,step):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)


    # def _get_action(self, agent):
    #     if agent.action_callback is None:
    #         return
    #     else:
    #         return self.action_callback(agent, self.world)

    # # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        action_dim = getattr(self.world, "action_dim", self.world.dim_p)
        agent.action.u = np.zeros(action_dim)
        agent.action.c = np.zeros(self.world.dim_c)

        # print(action)
        # print(action)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        for idx in range(min(len(action), len(agent.action.u))):
            agent.action.u[idx] = action[idx]
        sensitivity = 1.0

        agent.action.u *= sensitivity
        # process action
        # if isinstance(action_space, MultiDiscrete):
        #     act = []
        #     size = action_space.high - action_space.low + 1
        #     index = 0
        #     for s in size:
        #         act.append(action[index:(index + s)])
        #         index += s
        #     action = act
        # else:
        #     action = [action]
        #
        # if agent.movable:
        #     # physical action
        #     if self.discrete_action_input:
        #         agent.action.u = np.zeros(self.world.dim_p)
        #         # process discrete action
        #         if action[0] == 1:
        #             agent.action.u[0] = -1.0
        #         if action[0] == 2:
        #             agent.action.u[0] = +1.0
        #         if action[0] == 3:
        #             agent.action.u[1] = -1.0
        #         if action[0] == 4:
        #             agent.action.u[1] = +1.0
        #     else:
        #         if self.force_discrete_action:
        #             d = np.argmax(action[0])
        #             action[0][:] = 0.0
        #             action[0][d] = 1.0
        #         if self.discrete_action_space:
        #             agent.action.u[0] += action[0][1] - action[0][2]
        #             agent.action.u[1] += action[0][3] - action[0][4]
        #         else:
        #             agent.action.u = action[0]
        #     sensitivity = 5.0
        #     if agent.accel is not None:
        #         sensitivity = agent.accel
        #     agent.action.u *= sensitivity
        #     action = action[1:]
        # # if not agent.silent:
        # #     # communication action
        # #     if self.discrete_action_input:
        # #         agent.action.c = np.zeros(self.world.dim_c)
        # #         agent.action.c[action[0]] = 1.0
        # #     else:
        # #         agent.action.c = action[0]
        # #     action = action[1:]
        # # make sure we used all elements of action
        # assert len(action) == 0

    # # set env action for a particular agent为每一个智能体设置动作
    # #这里有一些改动
    # def _set_action(self, action, agent, action_space, time=None):
    #     # print(action)
    #     agent.action.u = np.zeros(self.world.dim_p)
    #     agent.action.c = np.zeros(self.world.dim_c)
    #     agent.action.u = action
    #     if action[0] < 0:
    #         agent.action.u[0] = 0
    #     # if action[0] > 3:
    #     #     agent.action.u[0] = 3
    #
    #     # process action
    #
    #     # if isinstance(action_space, MultiDiscrete):
    #     #     act = []
    #     #     size = action_space.high - action_space.low + 1
    #     #     index = 0
    #     #     for s in size:
    #     #         act.append(action[index:(index + s)])
    #     #         index += s
    #     #     action = act
    #     #     print("1010101010110")
    #     # else:
    #     #     action = [action]
    #
    #
    #     # if agent.movable:
    #     #
    #     #     agent.action.u[0] += action[1] - action[2]
    #     #     agent.action.u[1] += action[3] - action[4]
    #     #
    #     #     sensitivity =1 #灵敏度等于四
    #     #
    #     #     agent.action.u *= sensitivity
    #     #
    #     #     if agent.action.u[0] < -0.01:
    #     #         agent.action.u[0] = -0.01
    #     #     if agent.action.u[0] > 3:
    #     #         agent.action.u[0] = 3
    #     #     if agent.action.u[1] < -1:
    #     #         agent.action.u[1] = -1
    #     #     if agent.action.u[1] > 1:
    #     #         agent.action.u[1] = 1
    #     #
    #     #     action = action[1:]
    #     #
    #     #
    #     # if not agent.silent:
    #     #     # communication action
    #     #     if self.discrete_action_input:
    #     #         agent.action.c = np.zeros(self.world.dim_c)
    #     #         agent.action.c[action[0]] = 1.0
    #     #     else:
    #     #         # agent.action.c = action[0]
    #     #         action = action[1:]
    #     # make sure we used all elements of action
    #     # assert len(action) == 0

    def get_sign(self, value):
        if value == 0:
            return 0
        return int(math.copysign(1, value))

    def evasion_acceleration(self):
        # 计算规避加速度
        a_evasion = np.zeros(2)
        k_e = 1e6
        for inter in self.interceptors:
            r_mi = self.pos_m - inter['pos']  # 飞行器到拦截器的相对位置
            dist_mi = np.linalg.norm(r_mi)
            if dist_mi < 1e-6:  # 防止除零
                continue
            # 规避加速度：远离拦截器，权重与距离倒数平方成正比
            a_evasion += -self.k_e * r_mi / (dist_mi ** 3)
        return a_evasion

    def action_callback1017(self, agent, world ):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        # add 计算规避加速度v1
        # auav_duav_dist = []
        # vel_m = agent.state.p_vel
        # vel_m_norm = np.linalg.norm(vel_m)
        # if vel_m_norm < 1e-6:
        #     return
        # vel_m_unit = vel_m / vel_m_norm
        # normal_unit = np.array([-vel_m_unit[1], vel_m_unit[0]])
        # k_e = 800
        # a_evasion = 0
        # for duav in self.world.agents :
        #     if duav.adversary :
        #         if duav.target == agent.namenumber :
        #             r_mi = (duav.state.p_pos - agent.state.p_pos) / 1
        #             dist_mi = np.linalg.norm(r_mi)
        #             if dist_mi < 1e-6:
        #                 continue
        #                 # 规避加速度投影到法向
        #             a_evasion_i = -k_e * r_mi / (dist_mi ** 3)
        #             a_evasion += np.dot(a_evasion_i, normal_unit) / 9.81

        # 计算规避加速度v5
        auav_duav_dist = []
        vel_m = agent.state.p_vel
        vel_m_norm = np.linalg.norm(vel_m)
        if vel_m_norm < 1e-6:
            return
        vel_m_unit = vel_m / vel_m_norm
        normal_unit = np.array([-vel_m_unit[1], vel_m_unit[0]])
        k_e0 = 10  # 基础规避系数（增强）
        d0 = 80  # 平滑距离参数 (m)
        alpha = 0.3  # 平滑因子（加快响应）
        d_max = 3000  # 目标距离上限 (m)
        d_threshold = 250  # 规避距离阈值 (m)
        a_max = 0.5  # 最大加速度 (g)
        delta_a_max = 0.8  # 加速度变化率限制 (g/s)
        N = 5  # 比例导引常数（适度增强）

        # 计算与目标的距离
        r_target = self.world.landmarks[0].state.p_pos - agent.state.p_pos
        dist_target = np.linalg.norm(r_target)

        # 收集所有敌对无人机的距离
        for duav in self.world.agents:
            if duav.adversary and duav.target == agent.namenumber:
                r_mi = duav.state.p_pos - agent.state.p_pos
                dist_mi = np.linalg.norm(r_mi)
                auav_duav_dist.append(dist_mi)

        # 动态调整 k_e
        min_dist = min(auav_duav_dist + [1e6])
        k_e = k_e0 * min(dist_target, d_max) / (min_dist + 50) if min_dist < d_threshold else 0
        k_e = np.clip(k_e, 2, 20)  # 放宽 k_e 上限

        # 计算规避加速度
        a_evasion = 0
        for duav in self.world.agents:
            if duav.adversary and duav.target == agent.namenumber:
                r_mi = duav.state.p_pos - agent.state.p_pos
                dist_mi = np.linalg.norm(r_mi)
                if dist_mi < 1e-6 or dist_mi >= d_threshold:
                    continue
                rel_vel = np.linalg.norm(vel_m - duav.state.p_vel)
                delta_t_lead = min(dist_mi / max(rel_vel, 1e-6), 1.0)  # 限制提前时间
                pred_pos = duav.state.p_pos + duav.state.p_vel * delta_t_lead
                # 适度平滑预测位置
                if not hasattr(duav, 'pred_pos_prev'):
                    duav.pred_pos_prev = pred_pos
                pred_pos = 0.5 * pred_pos + 0.5 * duav.pred_pos_prev
                duav.pred_pos_prev = pred_pos
                r_mi_pred = pred_pos - agent.state.p_pos
                dist_mi_pred = np.linalg.norm(r_mi_pred)
                if dist_mi_pred < 1e-6:
                    continue
                # 规避方向检查（逆轨优化）
                target_dir = r_target / max(dist_target, 1e-6)

                # 这个是规避方向的
                # evasion_dir = -r_mi_pred / max(dist_mi_pred, 1e-6)
                # cos_angle = np.dot(evasion_dir, target_dir)
                # k_e_adjusted = k_e * 0.1 if cos_angle < 0 else k_e  # 偏离过多时减弱规避
                # 这个是速度方向和视线方向的夹角
                cos_angle = np.dot(vel_m_unit, target_dir)
                k_e_adjusted = k_e * 0.0 if cos_angle < 0.5 else k_e  # 速度方向偏离目标大于60度时减弱规避

                # 平滑规避加速度
                a_evasion_i = -k_e_adjusted * r_mi_pred / (dist_mi_pred + d0)
                a_evasion += np.dot(a_evasion_i, normal_unit) / 9.81

        # 平滑加速度（指数移动平均）
        if not hasattr(self, 'a_evasion_prev'):
            self.a_evasion_prev = 0
        a_evasion_new = a_evasion
        a_evasion = alpha * a_evasion_new + (1 - alpha) * self.a_evasion_prev
        # 适度限制加速度变化率
        delta_a = a_evasion - self.a_evasion_prev
        delta_a = np.clip(delta_a, -delta_a_max * 0.01, delta_a_max * 0.01)  # 假设 dt=0.01s
        a_evasion = self.a_evasion_prev + delta_a
        self.a_evasion_prev = a_evasion


        if agent.movable:
            target = self.world.landmarks[0]
            r_pos = (target.state.p_pos - agent.state.p_pos) / 1
            dist = np.sqrt(np.sum(np.square(r_pos)))
            vel_r = (target.state.p_vel - agent.state.p_vel) / 1
            # q = np.arctan(r_pos[1]/r_pos[0])
            # agent.action.u[0] = -2 * (q - agent.state.v_vel[1]) * agent.state.v_vel[0] / 9.8
            # q_dot = (vel_r[1] * np.cos(q) - vel_r[0] * np.sin(q)) / dist
            # q_dot = (vel_r[1] * r_pos[0] - vel_r[0] * r_pos[1]) / dist
            q_dot = np.cross(r_pos, vel_r) / dist
            q_dot = q_dot / dist

            # A = random.uniform(1.5, 1.8) # 振幅（可调整，单位 g）
            # omega = (2 * np.pi / 50)  # 计算角频率
            # phase_offset = np.random.uniform(0, 2 * np.pi)  # 随机相位
            #
            # agent.action.u[1] = 3 * q_dot * agent.state.v_vel[0] / 9.8 + A * np.sin(
            #     omega * agent.state.timestep) / 9.8
            # # agent.action.u[1] = 3 * q_dot * agent.state.v_vel[0] / 9.8
            # # agent.action.u[0] = 3 * q_dot * agent.state.v_vel[0] / 9.8 + 0.02 * np.sin((2 * np.pi / 50) * agent.state.timestep + 0.01)
            # agent.action.u[0] = 0




            # if agent.state.timestep < 10 :
            #     # agent.action.u[1] = self.get_sign(q_dot) * 0.5
            #     agent.action.u[1] = 0.02
            #     agent.action.u[0] = 0
            # else:
            #     agent.action.u[1] = 3 * q_dot * agent.state.v_vel[0] / 9.81
            #     agent.action.u[0] = 0

            agent.action.u[1] = 3 * q_dot * agent.state.v_vel[0] / 9.81 + a_evasion
            agent.action.u[0] = 0

            # ###----------------------加个一阶滞后环节------------------###
            # T = 0.1  # 滞后时间常数
            # alpha = T / (T + self.dt)
            # load_y = alpha * agent.state.load[1] + (1 - alpha) * agent.action.u[1]
            # agent.action.u[1] = max(-1, min(1, load_y))
            # ###-------------------------------------------------------###

            load_y = agent.action.u[1]
            agent.action.u[1] = max(-2, min(2, load_y))

            agent.state.load[1] = agent.action.u[1]
            

        if not agent.silent:
            # communication action
            agent.action.c = np.zeros(self.world.dim_c)
            
    # def action_callback1113(self, agent, world):
    #     agent.action.u = np.zeros(self.world.dim_p)
    #     agent.action.c = np.zeros(self.world.dim_c)
    #
    #     if agent.movable:
    #         target = self.world.agents[agent.target]
    #         r_pos = (target.state.p_pos - agent.state.p_pos) / 1
    #         dist = np.sqrt(np.sum(np.square(r_pos)))
    #         vel_r = (target.state.p_vel - agent.state.p_vel) / 1
    #
    #         q = np.arctan2(r_pos[1],r_pos[0])
    #         q_dot2 = (q - agent.q_old)/0.1
    #         agent.q_old = q
    #
    #         q_dot = np.cross(r_pos, vel_r) / dist
    #         q_dot1 = q_dot / dist
    #
    #         agent.action.u[1] =  3 * q_dot1 * agent.state.v_vel[0] / 9.8
    #         agent.action.u[0] = 0
    #
    #         agent.state.qddd[0] = q_dot1*57.3
    #         agent.state.qddd[1] = q_dot2*57.3
    #         agent.state.qddd1[0] = q*57.3
    #         agent.state.qddd1[1] = 0
    #
    #
    #     if not agent.silent:
    #         # communication action
    #
    #         agent.action.c = np.zeros(self.world.dim_c)


    # added in win at 20200422 22:01
    # for record pos vel gamma and done of all agents
    # 用于记录位置，速度，伽马参数和所有智能体的存活状态
    def record_pos(self):
        agents_pos = []
        for agent in self.agents:
            agents_pos.append(agent.state.p_pos)
        return np.array(agents_pos)

    def record_eval(self):
        agents_pos = []
        for agent in self.agents:
            if agent.adversary :
                agents_pos.append(agent.state.eval)
        agents_pos = np.array(agents_pos)
        column_means = np.mean(agents_pos, axis=0)
        return np.array(column_means)


    def record_qddd(self):
        agents_pos1 = []
        for agent in self.agents:
            agents_pos1.append(agent.state.k3)
        return np.array(agents_pos1)
    
    def record_qddd1(self):
        agents_pos1 = []
        for agent in self.agents:
            agents_pos1.append(agent.state.k4)
        return np.array(agents_pos1)

    def record_qddd2(self):
        agents_pos1 = []
        for agent in self.agents:
            agents_pos1.append(agent.state.k5)
        return np.array(agents_pos1)

    def record_pvel(self):
        agents_vel = []
        for agent in self.agents:
            agents_vel.append(agent.state.p_vel)
        return np.array(agents_vel)

    def record_vel(self):
        agents_vel = []
        for agent in self.agents: 
            agents_vel.append(agent.state.v_vel)
        return np.array(agents_vel)
    
    def record_q(self):
        agents_vel = []
        for agent in self.agents: 
            agents_vel.append(agent.state.kq)
        return np.array(agents_vel)
    def record_time_tgo(self):
        agents_vel = []
        for agent in self.agents:
            agents_vel.append(agent.state.time_tgo)
        return np.array(agents_vel)
    def record_time_tgo_dist(self):
        agents_vel = []
        for agent in self.agents:
            agents_vel.append(agent.state.time_tgo_dist)
        return np.array(agents_vel)

    def record_done(self):
        agents_done = np.zeros(self.n, dtype=bool)
        if self.done_callback is not None:
            for i, agent in enumerate(self.agents):
                agents_done[i] = self.done_callback(agent, self.world)
        return agents_done
    
    # def record_rewrew1(self):
    #     agents_rewrew = np.zeros(self.n, dtype=float)
    #     if self.reward_callback is not None:
    #         for i, agent in enumerate(self.agents):
    #             agents_rewrew[i] =  self.reward_callback(agent, self.world)
    #     return np.array(agents_rewrew)

    def record_rewrew(self):
        agents_rewrew = []
        for agent in self.agents:
            agents_rewrew.append(self.reward_callback(agent, self.world))
        return np.array(agents_rewrew)


    def record_action(self):
        agents_uact = []
        for agent in self.agents:
            agents_uact.append(agent.state.load)
        return agents_uact

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(900, 900)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                # changed in win at 20200422 20:50
                ##能动的弄成三角，不能动的弄成圆形，给出大小画图形，位置就是你位置数组的0号位
                if entity.movable:
                    geom = rendering.make_fixwing(entity.size)
                    # geom = rendering.make_circle(entity.size)
                    # geom = rendering.SimpleImageViewer(1)
                else:
                    if entity.food:
                        geom = rendering.make_circle(entity.size,30,True)
                    else:
                        geom = rendering.make_circle(entity.size,30)

                xform = rendering.Transform()
                if 'agent' in entity.name:
                    if entity.doneflag:
                        entity.color = np.array([0, 0, 0])
                        geom.set_color(0, 0, 0, 0)
                    else:
                        geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                if entity.movable:
                    self.render_geoms.append(geom)
                    # self.render_geoms.append(geom2)
                else:
                    self.render_geoms.append(geom)



                # self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            # 将几何图形添加到查看器
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 2
            cam_range = self.world.scale
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos / (2 * self.world.scale)
            # print(pos)


            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                if 'agent' in entity.name:
                    if entity.doneflag:
                        self.render_geoms[e].set_color(0, 0, 0, 0)

                nposx = entity.state.p_pos[0]
                nposy = entity.state.p_pos[1]
                self.render_geoms_xform[e].set_translation(nposx, nposy)

                
                if entity.movable:

                    self.render_geoms_xform[e].set_rotation(entity.state.v_vel[1] - 0.5 * np.pi)




            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    # 在局部坐标系中创建受体场位置
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        # 圆形接受的区域
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            # 增加起源
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        # 网格接受的区域
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx








# vectorized wrapper for a batch of multi-agent environments
# 一批多智能体环境的矢量化包装器
# assumes all environments have the same observation and action space
# 假设所有环境都有相同的观察和行动空间
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    # 绘制环境
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
