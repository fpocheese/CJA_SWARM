import numpy as np

from onpolicy.envs.mpe.core import FighterWorld, FighterAgent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


def _arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


def _unit(v, fallback=None):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        if fallback is None:
            return np.zeros_like(v)
        return fallback.copy()
    return v / norm


def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Scenario(BaseScenario):
    """3D cooperative air-defense scenario.

    Defenders are policy-controlled fixed-wing agents launched from the ground.
    Attackers are scripted high-altitude incoming UAVs. The horizontal geometry
    follows the two paper cases while adding altitude and vertical guidance.
    """

    def make_world(self, args):
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
        self.reward_coord_tol = _arg(args, "reward_coord_tol", 0.5)
        self.reward_angle_power = _arg(args, "reward_angle_power", 0.3)
        self.reward_coord_power = _arg(args, "reward_coord_power", 0.3)
        self.reward_use_progress = _arg(args, "reward_use_progress", False)
        self.case = _arg(args, "case_3d", "case1")
        self.hit_radius = _arg(args, "hit_radius_3d", 20.0)
        self.protected_asset = np.array([0.0, 0.0, 0.0])
        self.attack_maneuver_gain = _arg(args, "attack_maneuver_gain", 2.10)
        self.attack_maneuver_offset_gain = _arg(args, "attack_maneuver_offset_gain", 1.25)
        self.case1_lateral_base = _arg(args, "case1_lateral_base", 0.95)
        self.case1_lateral_tail = _arg(args, "case1_lateral_tail", 0.40)
        self.case1_vertical_amp = _arg(args, "case1_vertical_amp", 0.35)
        self.case2_lateral_amp = _arg(args, "case2_lateral_amp", 1.05)
        self.case2_vertical_amp = _arg(args, "case2_vertical_amp", 0.50)
        self.attack_maneuver_freq = _arg(args, "attack_maneuver_freq", 1.35)

        world = FighterWorld()
        world.dim_p = 3
        world.dim_c = 4
        world.action_dim = 2
        world.use_agent_script_callback = True

        num_good_agents = 8      # attackers
        num_adversaries = 20     # defenders
        world.agents = [FighterAgent() for _ in range(num_adversaries + num_good_agents)]
        world.food = [Landmark()]
        world.aaa = []
        world.bbb = []
        world.ccc = []
        world.landmarks = world.food

        assignment = [20, 21, 22, 23, 24, 25, 26, 27,
                      20, 21, 22, 23, 24, 25, 26, 27,
                      20, 21, 22, 23]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.namenumber = i
            agent.doneflag = False
            agent.collide = True
            agent.leader = i == 0
            agent.silent = i > 0
            agent.adversary = i < num_adversaries
            agent.size = 13 if agent.adversary else 15
            agent.accel = 3.0 if agent.adversary else 4.5
            agent.state.target = 10
            agent.state.done = False
            agent.target = assignment[i] if agent.adversary else 0
            agent.q_old = 0.0
            agent.state.timestep = 0.0
            if not agent.adversary:
                agent.action_callback = self.action_callback

        world.food[0].name = "protected asset"
        world.food[0].collide = False
        world.food[0].movable = False
        world.food[0].food = True
        world.food[0].size = 5
        world.food[0].boundary = False
        world.attacker_load_limit = _arg(args, "attacker_load_limit", 1.75)
        world.attacker_yaw_scale = _arg(args, "attacker_yaw_scale", 1.55)
        world.attacker_pitch_scale = _arg(args, "attacker_pitch_scale", 1.55)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        rng = np.random
        world.landmarks[0].state.p_pos = self.protected_asset.copy()
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.doneflag = False
            agent.action.u = np.zeros(world.action_dim)
            agent.action.c = np.zeros(world.dim_c)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.time_tgo = np.zeros(world.dim_p)
            agent.state.time_tgo_dist = np.zeros(world.dim_p)
            agent.state.load = np.zeros(world.action_dim)
            agent.state.doneflag_me_target = np.zeros(world.dim_p)
            agent.state.last_loadx = 0.0
            agent.state.last_loady = 0.0
            agent.state.load_delt_all = 0.0
            agent.state.last_q_dot = 0.0
            agent.state.kalman_p_last = 1.0
            agent.state.dist_target = 1000.0
            agent.state.eval = np.zeros(5)
            agent.state.eval_flag = 0
            agent.state.timestep = 0.0
            agent.state.timeover = False
            agent.state.actual_hit = False
            agent.state.hit_time = np.nan
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])

        attackers = self.good_agents(world)
        defenders = self.adversaries(world)

        att_angles = np.linspace(0.0, 2 * np.pi, len(attackers), endpoint=False) + rng.normal(0.0, 0.08, len(attackers))
        for j, agent in enumerate(attackers):
            radius = rng.uniform(1200.0, 1500.0)
            altitude = rng.uniform(650.0, 850.0)
            agent.state.p_pos = np.array([radius * np.cos(att_angles[j]), radius * np.sin(att_angles[j]), altitude])
            direction = _unit(self.protected_asset - agent.state.p_pos, np.array([0.0, 0.0, -1.0]))
            speed = rng.uniform(24.0, 30.0)
            agent.state.p_vel = speed * direction
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.phase = rng.uniform(0.0, 2 * np.pi)
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)

        def_angles = np.linspace(0.0, 2 * np.pi, len(defenders), endpoint=False) + rng.normal(0.0, 0.12, len(defenders))
        for i, agent in enumerate(defenders):
            radius = rng.uniform(20.0, 95.0)
            agent.state.p_pos = np.array([radius * np.cos(def_angles[i]), radius * np.sin(def_angles[i]), 0.0])
            target = world.agents[agent.target]
            direction = _unit(target.state.p_pos - agent.state.p_pos + np.array([0.0, 0.0, 35.0]))
            speed = rng.uniform(16.0, 22.0)
            agent.state.p_vel = speed * direction
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.lamda0 = agent.state.v_vel[2]
            agent.state.dist0 = max(np.linalg.norm(target.state.p_pos - agent.state.p_pos), 0.01)
            agent.state.dist1 = agent.state.dist0
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)

    def _vel_to_flight_state(self, vel):
        speed = max(np.linalg.norm(vel), 1e-6)
        yaw = np.arctan2(vel[1], vel[0])
        pitch = np.arcsin(np.clip(vel[2] / speed, -1.0, 1.0))
        return np.array([speed, pitch, yaw])

    def _flight_dir(self, agent):
        speed, pitch, yaw = agent.state.v_vel
        return np.array([np.cos(pitch) * np.cos(yaw),
                         np.cos(pitch) * np.sin(yaw),
                         np.sin(pitch)])

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def is_collision(self, agent1, agent2):
        return np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos) < self.hit_radius

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else np.array([0.0])

    def adversary_reward(self, agent, world):
        if agent.doneflag:
            return np.array([0.0])
        target = world.agents[agent.target]
        rel = target.state.p_pos - agent.state.p_pos
        dist = max(np.linalg.norm(rel), 0.01)

        if self.reward_use_progress:
            r_dist = (agent.state.dist1 - dist) / 1000.0
        else:
            r_dist = np.exp(-self.reward_alpha_dist * dist)
        agent.state.dist1 = dist

        los = _unit(rel)
        flight_dir = self._flight_dir(agent)
        angle_err = np.arccos(np.clip(np.dot(flight_dir, los), -1.0, 1.0))
        r_angle = -self.reward_alpha_angle * (angle_err ** self.reward_angle_power)

        r_hit = self.reward_hit_bonus if self.is_collision(agent, target) else 0.0
        r_energy = -self.reward_alpha_energy * np.sum(np.square(agent.action.u)) * world.dt

        tgo = dist / max(agent.state.v_vel[0], 1.0)
        group_tgo = []
        for other in self.adversaries(world):
            if other.target == agent.target and not other.doneflag:
                d = np.linalg.norm(world.agents[other.target].state.p_pos - other.state.p_pos)
                group_tgo.append(d / max(other.state.v_vel[0], 1.0))
        tgo_mean = np.mean(group_tgo) if group_tgo else tgo
        time_abs = abs(tgo - tgo_mean)
        r_coord = -self.reward_alpha_coord * (time_abs ** self.reward_coord_power)
        if time_abs <= self.reward_coord_tol:
            r_coord += self.reward_coord_bonus

        agent.state.dist_target = dist
        agent.state.time_tgo[0] = tgo
        agent.state.time_tgo[1] = dist
        agent.state.time_tgo_dist[0] = time_abs
        agent.state.eval[0] = agent.state.timestep
        agent.state.eval[1] = np.linalg.norm(agent.state.load)
        agent.state.eval[2] = dist
        agent.state.eval[3] += np.sum(np.square(agent.state.load)) * world.dt

        rew = (self.reward_w_dist * r_dist +
               self.reward_w_angle * r_angle +
               self.reward_w_hit * r_hit +
               self.reward_w_coord * r_coord +
               self.reward_w_energy * r_energy)
        return np.array([rew], dtype=np.float32)

    def observation(self, agent, world):
        target = world.agents[agent.target]
        rel = target.state.p_pos - agent.state.p_pos
        dist = max(np.linalg.norm(rel), 0.01)
        los = rel / dist
        rel_vel = target.state.p_vel - agent.state.p_vel
        closing = np.dot(rel_vel, los)
        flight_dir = self._flight_dir(agent)
        angle_err = np.arccos(np.clip(np.dot(flight_dir, los), -1.0, 1.0))
        tgo = dist / max(agent.state.v_vel[0], 1.0)

        group_tgo = []
        for other in self.adversaries(world):
            if other.target == agent.target and not other.doneflag:
                d = np.linalg.norm(world.agents[other.target].state.p_pos - other.state.p_pos)
                group_tgo.append(d / max(other.state.v_vel[0], 1.0))
        tgo_mean = np.mean(group_tgo) if group_tgo else tgo

        obs = np.array([
            dist / 2000.0,
            rel[2] / 1000.0,
            closing / 100.0,
            angle_err / np.pi,
            agent.state.v_vel[0] / 65.0,
            target.state.v_vel[0] / 65.0,
            agent.state.v_vel[1] / np.pi,
            _wrap_pi(agent.state.v_vel[2]) / np.pi,
            los[0],
            los[1],
            los[2],
            (tgo - tgo_mean) / 50.0,
            agent.state.load[0],
            agent.state.load[1],
        ], dtype=np.float32)
        agent.state.dist_target = dist
        return obs

    def done_callback(self, agent, world):
        if agent.doneflag:
            return True

        target = world.agents[agent.target]
        if agent.adversary and self.is_collision(agent, world.agents[agent.target]):
            agent.state.doneflag_me_target[0] = 1
            agent.doneflag = True
            agent.state.actual_hit = True
            agent.state.hit_time = agent.state.timestep

            same_target = [other for other in self.adversaries(world) if other.target == agent.target]
            if same_target and all(getattr(other.state, "actual_hit", False) for other in same_target):
                target.doneflag = True
            return True
        return False

    def action_callback(self, agent, world):
        target = world.landmarks[0]
        rel = target.state.p_pos - agent.state.p_pos
        los = _unit(rel, np.array([0.0, 0.0, -1.0]))
        speed = max(agent.state.v_vel[0], 1.0)
        flight_dir = self._flight_dir(agent)

        vertical = np.array([0.0, 0.0, 1.0])
        horizontal_perp = _unit(np.cross(vertical, los), np.array([1.0, 0.0, 0.0]))
        phase = getattr(agent.state, "phase", 0.0)
        t = agent.state.timestep
        if self.case == "case2":
            maneuver = self.case2_lateral_amp * np.sin(self.attack_maneuver_freq * t + phase) * horizontal_perp
            maneuver += self.case2_vertical_amp * np.sin(1.55 * t + 0.7 * phase) * vertical
        else:
            if t < 15.0:
                sign = 1.0 if (agent.namenumber % 2 == 0) else -1.0
                maneuver = sign * self.case1_lateral_base * horizontal_perp
                maneuver += self.case1_vertical_amp * np.sin(self.attack_maneuver_freq * t + phase) * vertical
            else:
                maneuver = self.case1_lateral_tail * np.sin(0.30 * t + phase) * horizontal_perp

        offset = self.attack_maneuver_offset_gain * (
            0.03 * np.sin(0.22 * t) * horizontal_perp
            + 0.03 * np.sin(0.19 * t) * vertical
        )
        maneuver = (maneuver + offset) * self.attack_maneuver_gain

        desired_dir = _unit(los + maneuver, los)
        desired_vel = np.clip(speed, 20.0, 32.0) * desired_dir
        acc_cmd = (desired_vel - agent.state.p_vel) / 1.2

        vhat = _unit(agent.state.p_vel, flight_dir)
        axial = np.dot(acc_cmd, vhat) / 9.81
        normal = acc_cmd - np.dot(acc_cmd, vhat) * vhat
        horizontal_vel = np.array([agent.state.p_vel[0], agent.state.p_vel[1], 0.0])
        hhat = _unit(horizontal_vel, np.array([1.0, 0.0, 0.0]))
        lhat = np.array([-hhat[1], hhat[0], 0.0])

        agent.action.u = np.array([
            np.clip(np.dot(normal, lhat) / 9.81, -1.0, 1.0),
            np.clip(normal[2] / 9.81, -1.0, 1.0),
        ], dtype=np.float32)
        return agent.action

    def info(self, agent, world):
        return {}
