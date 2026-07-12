import numpy as np
# from multiagent.core import World, Agent, Landmark
from onpolicy.envs.mpe.core import FighterWorld, FighterAgent,Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


def _arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


class Scenario(BaseScenario):
    def make_world(self,args):
        self.reward_w_dist = _arg(args, "reward_w_dist", 0.10)
        self.reward_w_angle = _arg(args, "reward_w_angle", 1.00)
        self.reward_w_hit = _arg(args, "reward_w_hit", 1.00)
        self.reward_w_coord = _arg(args, "reward_w_coord", 1.00)
        self.reward_w_energy = _arg(args, "reward_w_energy", 1.00)
        self.reward_alpha_dist = _arg(args, "reward_alpha_dist", 1e-3)
        self.reward_alpha_angle = _arg(args, "reward_alpha_angle", 1e-2)
        self.reward_alpha_coord = _arg(args, "reward_alpha_coord", 5e-3)
        self.reward_alpha_energy = _arg(args, "reward_alpha_energy", 5e-2)
        self.reward_hit_bonus = _arg(args, "reward_hit_bonus", 3.0)
        self.reward_coord_bonus = _arg(args, "reward_coord_bonus", 0.1)
        self.reward_coord_tol = _arg(args, "reward_coord_tol", 0.1)
        self.reward_angle_power = _arg(args, "reward_angle_power", 0.3)
        self.reward_coord_power = _arg(args, "reward_coord_power", 0.3)
        self.reward_use_progress = _arg(args, "reward_use_progress", False)

        world = FighterWorld()
        # set any world properties first
        world.dim_c = 4
        #world.damping = 1
        num_good_agents = 8    ##进攻无人机的数量
        num_adversaries = 20   ##防御无人机的数量
        num_agents = num_adversaries + num_good_agents
        # num_landmarks = 1
        num_food = 1
        # num_forests = 1
        num_aaa = 1
        num_bbb = 1
        num_ccc = 1
        # add agents
        world.agents = [FighterAgent() for i in range(num_agents)]


        for i, agent in enumerate(world.agents):
            # agent.timestep = agent.timestep + 1
            # print("count", agent.timestep)
            agent.name = 'agent %d' % i
            agent.namenumber = i
            agent.doneflag = False
            agent.collide = True
            agent.leader = True if i == 0 else False  ##第一个飞行器就是leader
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 13 if agent.adversary else 15
            agent.accel = 3.0 if agent.adversary else 4.5
            agent.food = False
            agent.state.target = 10
            agent.state.done = False
            agent.target = 20
            agent.q_old = 0
            agent.state.timestep = 0
            # agent.have = 0
            if i == 0:
                agent.target = 20
            if i == 1:
                agent.target = 21
            if i == 2:
                agent.target = 22
            if i == 3:
                agent.target = 23
            if i == 4:
                agent.target = 24
            if i == 5:
                agent.target = 25
            if i == 6:
                agent.target = 26
            if i == 7:
                agent.target = 27
            if i == 8:
                agent.target = 20
            if i == 9:
                agent.target = 21
            if i == 10:
                agent.target = 22
            if i == 11:
                agent.target = 23
            if i == 12:
                agent.target = 24
            if i == 13:
                agent.target = 25
            if i == 14:
                agent.target = 26
            if i == 15:
                agent.target = 27
            if i == 16:
                agent.target = 20
            if i == 17:
                agent.target = 21
            if i == 18:
                agent.target = 22
            if i == 19:
                agent.target = 23


            # agent.target = 3
            #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.5
        # add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = True
        #     landmark.movable = False
        #     landmark.food = False
        #     landmark.size = 2
        #     landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.food = True
            landmark.size = 5
            landmark.boundary = False
        world.aaa = [Landmark() for i in range(num_aaa)]
        for i, landmark in enumerate(world.aaa):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.food = False
            landmark.size = 2100
            landmark.boundary = False

        world.bbb = [Landmark() for i in range(num_bbb)]
        for i, landmark in enumerate(world.bbb):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.food = False
            landmark.size = 2000
            landmark.boundary = False

        world.ccc = [Landmark() for i in range(num_ccc)]
        for i, landmark in enumerate(world.ccc):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.food = False
            landmark.size = 20
            landmark.boundary = False


        world.landmarks += world.food
        # world.landmarks += world.forests
        world.landmarks += world.aaa
        world.landmarks += world.bbb
        world.landmarks += world.ccc

        ##9.7  21.04从defense里添加的循环
        ##就是给agent的赋值了action_callback函数
        for i, agent in enumerate(world.agents):
            if not agent.adversary:
                agent.action_callback = self.action_callback


        # world.landmarks += self.set_boundaries(world)  # world boundaries now penalized with negative reward
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):   ##设定界限
        boundary_list = []  ##界限的列表
        landmark_size = 1  ##地标的大小
        edge = 1 + landmark_size   ##边是地标大小+1
        num_landmarks = int(edge * 2 / landmark_size)  ##地标的数量
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list  ##边界的list


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.doneflag = False
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            # agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.ccc):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.bbb):
            landmark.color = np.array([0.6, 0.9, 0.6])
        for i, landmark in enumerate(world.aaa):
            landmark.color = np.array([0.35, 0.45, 0.65])


        for i, landmark in enumerate(world.food):

            landmark.state.p_pos = np.random.uniform(-90, +90, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            # print(landmark.state.p_pos)

        for i, landmark in enumerate(world.bbb):
            bobo = world.landmarks[0]

            landmark.state.p_pos = bobo.state.p_pos
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.aaa):
            bobo = world.landmarks[0]

            landmark.state.p_pos = bobo.state.p_pos
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.ccc):
            bobo = world.landmarks[0]

            landmark.state.p_pos = bobo.state.p_pos
            landmark.state.p_vel = np.zeros(world.dim_p)





        for agent in world.agents:
            ######
            agent.action.u = np.zeros(world.dim_p)
            agent.action.c = np.zeros(world.dim_c)
            agent.state.time_tgo = np.zeros(world.dim_p)
            agent.state.time_tgo_dist = np.zeros(world.dim_p)
            agent.state.load = np.zeros(world.dim_p)
            agent.state.doneflag_me_target = np.zeros(world.dim_p)
            agent.state.last_loadx = 0
            agent.state.last_loady = 0
            agent.state.load_delt_all = 0
            agent.state.last_q_dot = 0
            agent.state.kalman_p_last = 1.0
            agent.state.dist_target = 1000.0
            agent.state.eval = np.zeros(5)
            agent.state.eval_flag = 0

            if agent.adversary:
                lr = np.random.uniform(0, 10, 1)
                # ltheta = np.random.uniform(-0.3, +0.3, 1) + (2*i+0.5)*np.pi/3
                ltheta = np.random.uniform(0, 2*np.pi, 1)
                posx =world.landmarks[0].state.p_pos[0] + lr * np.cos(ltheta)
                posy =world.landmarks[0].state.p_pos[1] + lr * np.sin(ltheta)
                agent.state.p_pos = np.concatenate([posx]+[posy])
                agent.state.qddd = np.zeros(world.dim_p)
                agent.state.qddd1 = np.zeros(world.dim_p)
                agent.state.k3 = np.zeros(world.dim_p)
                agent.state.k4 = np.zeros(world.dim_p)
                agent.state.k5 = np.zeros(world.dim_p)
                agent.state.kq = np.zeros(world.dim_p)

                
            else:
                lr = np.random.uniform(2000, 2100, 1)
                # ltheta = np.random.uniform(-0.8, +0.8, 1) + np.pi/2   + np.pi / 2
                ltheta = np.random.uniform(0, 2*np.pi, 1)
                # ltheta = np.random.uniform(+0.3, +0.8, 1) + (2 * i + 0.5) * np.pi / 3
                posx =world.landmarks[0].state.p_pos[0] + lr * np.cos(ltheta)
                posy =world.landmarks[0].state.p_pos[1] + lr * np.sin(ltheta)
                agent.state.p_pos = np.concatenate([posx] + [posy])
                # agent.state.p_vel = np.random.uniform(10 , 20, 2)
                # agent.state.v_vel = np.zeros(world.dim_p)
                agent.state.v_vel = np.zeros(world.dim_p)
                r_pos = (world.landmarks[0].state.p_pos - agent.state.p_pos)
                jiaodu = np.arctan2(r_pos[1], r_pos[0])
                agent.q_old = jiaodu
                # self.q_old = jiaodu
                agent.state.v_vel[0] = np.random.uniform(30, 50, 1)
                # agent.state.v_vel[0] = 0
                agent.state.v_vel[1] = jiaodu + np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, 1)
                px_dot = agent.state.v_vel[0] * np.cos(agent.state.v_vel[1])
                py_dot = agent.state.v_vel[0] * np.sin(agent.state.v_vel[1])
                agent.state.p_vel = np.array([px_dot, py_dot])
                agent.state.qddd = np.zeros(world.dim_p)
                agent.state.qddd1 = np.zeros(world.dim_p)
                agent.state.k3 = np.zeros(world.dim_p)
                agent.state.k4 = np.zeros(world.dim_p)
                agent.state.k5 = np.zeros(world.dim_p)
                agent.state.kq = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)


        for agent in world.agents:
            if agent.adversary:
                agent.state.v_vel = np.zeros(world.dim_p)
                r_pos = (world.agents[agent.target].state.p_pos - agent.state.p_pos)
                if r_pos[0] == 0:
                    if r_pos[1] > 0:
                        q = 0.5 * np.pi
                    if r_pos[1] < 0:
                        q = 1.5 * np.pi
                if not r_pos[0] == 0:
                    q = np.arctan2(r_pos[1] , r_pos[0])
                agent.state.lamda0 = q
                jiaodu = np.arctan2(r_pos[1],r_pos[0])
                agent.q_old = jiaodu
                agent.state.v_vel[0] = np.random.uniform(20, 30, 1)
                agent.state.v_vel[1] = jiaodu + np.random.uniform(-0.05*np.pi, 0.05*np.pi, 1)
                # agent.state.v_vel[1] = np.random.uniform(0, 2 * np.pi, 1)
                px_dot = agent.state.v_vel[0] * np.cos(agent.state.v_vel[1])
                py_dot = agent.state.v_vel[0] * np.sin(agent.state.v_vel[1])
                agent.state.p_vel = np.array([px_dot, py_dot])

                agent.state.dist0 = np.sqrt(np.sum(np.square(r_pos)))
                if agent.state.dist0 == 0:
                    agent.state.dist0 = 0.01
                agent.state.dist1 = agent.state.dist0
            else:
                continue





    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = 5
        return True if dist < dist_min else False



    # return all agents that are not adversaries
    # 返回所有不是对手的智能体
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # 智能体根据到每个地标的最小代理距离获得奖励
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward


    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

##我们是一个循环，对每个agent循环一次去求奖励，那么放进去的agent就是单个的智能体，然后根据每个的情况去判断用的是哪个函数，然后把这个agent放到这个函数中

    def agent_reward(self, agent, world):
        #######
        # if agent.doneflag:
        #     return 0
        # # Agents are rewarded based on minimum agent distance to each landmark
        # rew = 0
        # adversaries = self.adversaries(world)
        # agents = self.good_agents(world)
        # 
        # if agent.collide:
        #     for a in adversaries:
        #         if self.is_collision(a, agent):
        #             rew -= 5
        # 
        # 
        # 
        # if self.is_collision(agent, world.landmarks[0]):
        #     rew += 3
        # 
        # rew -= 0.06 * np.sqrt(np.sum(np.square(world.landmarks[0].state.p_pos - agent.state.p_pos)))

        rew = 0
        return rew

    def adversary_reward(self, agent, world):
        if agent.doneflag:
            return 0
        # take friend as pursuit
        rew = 0
        rewshuchu = []
        bogys = self.good_agents(world)
        friends = self.adversaries(world)

        ###------------------------------------引导距离接近的部分-----------------------------------------#####
        dist = np.sqrt(np.sum(np.square(world.agents[agent.target].state.p_pos - agent.state.p_pos)))
        if dist == 0:
            dist = 0.01

        LZJ = (agent.state.dist1 - dist)/1000
        if self.reward_use_progress:
            r_dist = LZJ
        else:
            r_dist = np.exp(-self.reward_alpha_dist * dist)
        rew += self.reward_w_dist * r_dist
        agent.state.dist1 = dist
        ###------------------------------------引导距离接近的部分-----------------------------------------#####

        r_pos = (world.agents[agent.target].state.p_pos - agent.state.p_pos)
        q = np.arctan2(r_pos[1], r_pos[0])
        if r_pos[0] == 0:
            if r_pos[1] > 0:
                q = 0.5 * np.pi
            if r_pos[1] < 0:
                q = 1.5 * np.pi
        if not r_pos[0] == 0:
            q = np.arctan2(r_pos[1], r_pos[0])
        qian = abs(agent.state.v_vel[1] - q)
        r_angle = -self.reward_alpha_angle * (qian ** self.reward_angle_power)
        rew += self.reward_w_angle * r_angle
        # if qian < 0.01:
        #     rew += 2
        q_dot = (world.agents[agent.target].state.v_vel[0] * np.sin(world.agents[agent.target].state.v_vel[1] - q) - agent.state.v_vel[0] * np.sin(agent.state.v_vel[1] - q)) / dist

        if agent.collide:
            if self.is_collision(agent, world.agents[agent.target]):
                rew += self.reward_w_hit * self.reward_hit_bonus

        # N = (agent.action.u[1]*9.8)/(agent.state.v_vel[0]*q_dot)
        # N_power_sum = agent.action.u[1] + agent.action.u[0]
        r_energy = -self.reward_alpha_energy * (agent.action.u[1] ** 2 + agent.action.u[0] ** 2) * world.dt
        rew += self.reward_w_energy * r_energy

        ###-------------------过载变化率的惩罚--------------------------####
        loadx_delt = abs(agent.state.load[0] - agent.state.last_loadx)
        loady_delt = abs(agent.state.load[1] - agent.state.last_loady)
        load_delt = loadx_delt + loady_delt
        agent.state.load_delt_all = load_delt
        # rew -= 0.05 * load_delt
        agent.state.last_loadx = agent.state.load[0]
        agent.state.last_loady = agent.state.load[1]
        ###-------------------------------------------------------------###


        #######################TIME################
        N = 3
        time0 = (dist / agent.state.v_vel[0]) * (1 + (np.square(np.sin(agent.state.v_vel[1] - q)) / (4*N-2)))
        timeall = 0
        timeave = []
        number_i = 0
        for i , adv in enumerate(friends) :
            if adv.target == agent.target:
                number_i = number_i + 1
                dist111 = np.sqrt(np.sum(np.square(world.agents[adv.target].state.p_pos - adv.state.p_pos)))
                timeadv = (dist111 / adv.state.v_vel[0]) * (1 + (np.square(np.sin(adv.state.v_vel[1] - q)) / (4*N-2)))
                # timecha = abs(time0-timeadv)
                timeave.append(timeadv)
        timealll = sum(timeave)
        timeavenumber = timealll / number_i
        time_max = max(timeave)
        time_min = min(timeave)
        vel_desire = dist / time_min
        vel_ave_desire = dist / timeavenumber
        vel_desire_delt = abs(vel_desire - agent.state.v_vel[0])
        vel_ave_desire_delt = abs(vel_ave_desire - agent.state.v_vel[0])
        time_abs = abs(time0 - timeavenumber)
        r_coord = -self.reward_alpha_coord * (time_abs ** self.reward_coord_power)
        rew += self.reward_w_coord * r_coord
        agent.state.time_tgo[0] = time0
        agent.state.time_tgo[1] = dist
        agent.state.time_tgo_dist[0] = time_abs
        agent.state.time_tgo_dist[1] = vel_desire
        if time_abs <= self.reward_coord_tol:
            rew += self.reward_w_coord * self.reward_coord_bonus
        else:
            rew += 0

        # timeave = timeall/(number_i)
        # time_abs = abs(timeave - time0)
        ##############-------------------timerew---------------------------------###########
        # rew -= 0.0005 * (time_abs ** 0.3)  # canshu is 0.0005  rew is -1.19643
        # if time_abs <= 0.5 and dist < 100:
        #     rew -= 0.005 * (time_abs ** 0.3)  # canshu is 0.0005  rew is -0.030522
        #     if time_abs <= 0.05 and dist < 10:
        #         rew += 1
        # else:
        #     rew += 0


        # if time_abs < 0.05 and time_abs > 0:
        #     rew += 0.003
        # elif time_abs < 0.1 and time_abs > 0.05:
        #     rew += 0.001
        # elif time_abs < 0.5 and time_abs > 0.1:
        #     rew -= 0.001
        # else:
        #     rew -= 0.005
        ##############-------------------timerew---------------------------------###########
        ##--------------记录评价---------------##
        if agent.state.doneflag_me_target[0] == 0:
            agent.state.eval_flag = 1
        if agent.state.eval_flag == 1 :
            agent.state.eval_flag == 0
            # agent.state.eval[0] = abs(time0 - timeavenumber)
            agent.state.eval[0] = agent.state.timestep
            agent.state.eval[1] = abs(agent.state.load[1])
            agent.state.eval[2] = dist
            agent.state.eval[3] += (agent.state.load[0] ** 2 + agent.state.load[1] ** 2) * world.dt

        rewshuchu.append(rew)
        return np.concatenate([rewshuchu])

    def kalman_filter(self, q_dot, last_q_dot, p_last, q=1e-6, r=1e-1):
        """
        卡尔曼滤波器（函数形式）

        Args:
            q_dot (float): 当前测量值
            last_q_dot (float): 上一次平滑值（状态估计值）
            p_last (float): 上一次估计误差协方差
            q (float): 过程噪声协方差，表示系统动态变化的不确定性
            r (float): 测量噪声协方差，表示测量值的不确定性

        Returns:
            tuple: (滤波后的值, 当前估计误差协方差)
        """
        # 预测步骤
        p_prior = p_last + q  # 更新预测误差协方差

        # 更新步骤
        k = p_prior / (p_prior + r)  # 计算卡尔曼增益
        smoothed_value = last_q_dot + k * (q_dot - last_q_dot)  # 更新状态估计值
        p_current = (1 - k) * p_prior  # 更新误差协方差

        return smoothed_value, p_current



    def observation(self, agent, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        others_qdot = []
        r = []
        r_dot = []
        qchuan = []
        qian = []
        next = []
        other_pos = []
        other_vel = []
        ziji_pos = []
        ziji_vel = []
        action_power = []
        power_delt = []
        q = 0

        act = agent.action.u[1] + agent.action.u[0]
        power = 0.005 * (act ** 2) * 0.05
        action_power.append(power)

        power_delt_power = 0.01 * agent.state.load_delt_all
        power_delt.append(power_delt_power)

        pos_r = world.agents[agent.target].state.p_pos - agent.state.p_pos
        dist_r = np.sqrt(np.sum(np.square(pos_r)))
        agent.state.dist_target = dist_r
        if dist_r == 0:
            dist_r = 0.01
        if pos_r[0] == 0:
            if pos_r[1] > 0:
                q = 0.5*np.pi
            if pos_r[1] < 0:
                q = 1.5 * np.pi
        if not pos_r[0] == 0:
            q = np.arctan2(pos_r[1], pos_r[0])
        q_dot = (world.agents[agent.target].state.v_vel[0] * np.sin(world.agents[agent.target].state.v_vel[1] - q) - agent.state.v_vel[0] * np.sin(agent.state.v_vel[1] - q)) / dist_r
        r_dot1 = world.agents[agent.target].state.v_vel[0] * np.cos(world.agents[agent.target].state.v_vel[1] - q) - agent.state.v_vel[0] * np.cos(agent.state.v_vel[1] - q)

        # 平滑 q_dot
        # smoothed_q_dot, agent.state.kalman_p_last = self.kalman_filter(q_dot, agent.state.last_q_dot, agent.state.kalman_p_last, q=1e-5, r=1e-2)
        if dist_r > 600:
            smoothed_q_dot = q_dot
            agent.state.last_q_dot = smoothed_q_dot
        else:
            # if abs(q_dot - agent.state.last_q_dot) > 0.01:
            #     smoothed_q_dot = agent.state.last_q_dot
            # else:
            #     smoothed_q_dot = q_dot
            # onefilter_arg = 0.1
            # smoothed_q_dot = onefilter_arg * q_dot + (1 - onefilter_arg) * agent.state.last_q_dot
            smoothed_q_dot = q_dot
            agent.state.last_q_dot = smoothed_q_dot


        vqg = smoothed_q_dot * agent.state.v_vel[0]/9.8
        agent.state.k3[0] = 5 * vqg
        qianxiangjiao = abs(agent.state.v_vel[1] - q)
        qianxiangjiao = 0.0001 * (qianxiangjiao ** 0.3)
        qqq = q / agent.state.lamda0
        rll = dist_r / agent.state.dist0
        others_qdot.append(vqg)
        r.append(rll)
        r_dot.append(r_dot1)
        qchuan.append(qqq)
        qian.append(qianxiangjiao)

        ######################TIME#############################
        timealll = 0
        timeavenumber = 0
        dist_ave = 0
        t = 0
        # N = (agent.action.u[1] * 9.8) / (agent.state.v_vel[0] * q_dot)
        N = 3
        timeself = []
        timeave=[]
        timeself1 = (dist_r / agent.state.v_vel[0]) * (1 + (np.square(np.sin(agent.state.v_vel[1] - q)) / (4*N-2)))
        timeself.append(timeself1)
        time=[]
        number_i = 0
        for i,adv in enumerate(adversaries):
            if adv.target == agent.target:
                number_i = number_i + 1
                pos_adv_r = world.agents[adv.target].state.p_pos - adv.state.p_pos
                dist_adv_r = np.sqrt(np.sum(np.square(pos_adv_r)))
                # N1 = (adv.action.u[1] * 9.8) / (adv.state.v_vel[0] * q_dot)
                N1 = 3
                adv_t = (dist_adv_r / adv.state.v_vel[0]) * (1 + (np.square(np.sin(adv.state.v_vel[1] - q)) / (4*N1-2)))
                # timealll += adv_t
                timeave.append(adv_t)
        timealll = sum(timeave)
        timeavenumber = timealll / number_i
        time_max = max(timeave)
        time_min = min(timeave)
        vel_max_desire = dist_r / time_min
        vel_ave_desire = dist_r / timeavenumber
        vel_max_desire_delt = vel_max_desire - agent.state.v_vel[0]
        vel_ave_desire_delt = vel_ave_desire - agent.state.v_vel[0]
        time.append(vel_ave_desire_delt * 0.001)


        # time_delt = 0.001 * abs(timeself1 - timeavenumber)
        # # time_delt_end = 0
        # if time_delt <= 0.5 and dist_r < 100:
        #     time_delt_end = time_delt
        # else:
        #     time_delt_end = 0
        # # print(time_delt)
        # # timeave.append(timeave1)
        # time.append(time_delt_end)
        ########################################################
        # print('others_qdot',others_qdot)
        # print('qian', qian)
        # print('action_power', action_power)
        # print('time', time)
        # print('power_delt', power_delt)

        # return np.concatenate([qian] + [action_power] + [time])
        return np.concatenate([others_qdot] + [qian] + [time] + [action_power])
        # return np.concatenate([others_qdot] + [r] + [qian] + [time] + [timeave] + [timeself])







    def done_callback(self, agent, world):
        done_flag_zhanglian = []

        done_flag = False
        num_food = len(world.food)
        agents = self.good_agents(world)        ##进攻无人机
        adversaries = self.adversaries(world)   ##防御无人机
        if agent.doneflag:
            return True

        timeoverlist = []
        if agent.state.doneflag_me_target[1] == 1 and world.agents[agent.target].doneflag == False:
            for adv in adversaries:
                if adv.target == agent.target:
                    timeoverlist.append(adv.state.eval[0])
            timeoverlistave = np.mean(timeoverlist)
            for adv in adversaries:
                if adv.target == agent.target:
                    adv.state.eval[0] = abs(adv.state.eval[0] - timeoverlistave)

        if agent.state.doneflag_me_target[1] == 1:
            world.agents[agent.target].doneflag = True

        if agent.state.doneflag_me_target[0] == 1:
            agent.doneflag = True
        else:
            attack_number = 0
            die_number = 0
            for adv in adversaries:
                if adv.target == agent.target :
                    attack_number = attack_number + 1
                else:
                    pass

            if agent.adversary:  ##如果是防御无人机
                if self.is_collision(agent, world.agents[agent.target]) :
                    agent.state.doneflag_me_target[0] = 1
                    # agent.doneflag = True

            for adv in adversaries:
                if adv.target == agent.target :
                    # if adv.doneflag :
                    #     die_number = die_number + 1
                    if adv.state.doneflag_me_target[0] == 1:
                        die_number = die_number + 1
                else:
                    pass

            if die_number == attack_number :
                agent.state.doneflag_me_target[1] = 1
                for adv in adversaries:
                    if adv.target == agent.target:
                        adv.state.doneflag_me_target[1] = 1
                # world.agents[agent.target].doneflag = True


        done_flag_zhanglian.append(done_flag)

        return done_flag  ##这是总的到底是否结束
        # return np.concatenate([done_flag_zhanglian])




    def action_callback(self, agent, world):
        # if agent.bogy:
        #     agent.action.u[0] = 0.5
        #     agent.action.u[1] = 0
        #     return agent.action

        target = world.landmarks[0]
        r_pos = (target.state.p_pos - agent.state.p_pos) / 1
        dist = np.sqrt(np.sum(np.square(r_pos)))
        vel_r = (target.state.p_vel - agent.state.p_vel) / 1

        # q = np.arctan(r_pos[1]/r_pos[0])
        # agent.action.u[0] = -2 * (q - agent.state.v_vel[1]) * agent.state.v_vel[0] / 9.8
        # q_dot = (vel_r[1] * np.cos(q) - vel_r[0] * np.sin(q)) / dist
        # q_dot = (vel_r[1] * r_pos[0] - vel_r[0] * r_pos[1]) / dist

        q_dot = np.cross(r_pos, vel_r) / dist
        q_dot = q_dot / dist
        agent.action.u[0] = 3 * q_dot * agent.state.v_vel[0] / 9.8
        agent.action.u[1] = 0



        return agent.action

