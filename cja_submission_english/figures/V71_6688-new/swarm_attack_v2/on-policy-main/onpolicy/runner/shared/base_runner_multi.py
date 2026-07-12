"""
多算法统一Runner
支持: MAPPO, Advanced-MAPPO, IPPO, IA2C, IQL
在base_runner基础上修改算法选择逻辑
"""
import time
import wandb
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space

def _t2n(x):
    return x.detach().cpu().numpy()


class MultiAlgoRunner(object):
    """
    支持多种算法的Runner, 用于对比训练
    """
    def __init__(self, config, algo_name="MAPPO"):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.algo_name = algo_name

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = False
        self.use_render = False
        self.recurrent_N = self.all_args.recurrent_N
        self.save_interval = self.all_args.save_interval
        self.use_eval = False
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.model_dir = None

        # dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 根据算法名选择不同的policy和trainer
        self._setup_algorithm(algo_name)

        # buffer (所有算法共用)
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0]
        )

    def _setup_algorithm(self, algo_name):
        """根据算法名创建对应的policy和trainer"""
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]

        if algo_name == "MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, share_observation_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        elif algo_name == "Advanced-MAPPO":
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_advanced import R_MAPPOPolicy_Advanced as Policy
            from onpolicy.algorithms.r_mappo.r_mappo_advanced import R_MAPPO_Advanced as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, share_observation_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        elif algo_name == "IPPO":
            # IPPO: Independent PPO，每个agent用自己的obs作为critic输入（去中心化V）
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            # IPPO用obs替代share_obs
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
            self.use_centralized_V = False  # IPPO是去中心化的

        elif algo_name == "IA2C":
            # Independent A2C: 类似IPPO但不用PPO clip，用A2C loss
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.ia2c.ia2c import IA2C as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
            self.use_centralized_V = False

        elif algo_name == "IQL":
            # Independent Q-Learning (adapted for continuous/Box action space via soft Q)
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from onpolicy.algorithms.iql.iql import IQL as TrainAlgo
            self.policy = Policy(self.all_args, obs_space, obs_space, act_space, device=self.device)
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
            self.use_centralized_V = False

        else:
            raise ValueError(f"不支持的算法: {algo_name}")

    def run(self):
        """主训练循环, 返回奖励曲线"""
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        reward_curve = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.warmup()

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            avg_reward = np.mean(self.buffer.rewards) * self.episode_length
            reward_curve.append(avg_reward)

            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start)) if (end - start) > 0 else 0
                print(f"[{self.algo_name}] Ep {episode}/{episodes} | "
                      f"Steps {total_num_steps}/{self.num_env_steps} | "
                      f"FPS {fps} | "
                      f"Reward {avg_reward:.2f}")

                train_infos["average_episode_rewards"] = avg_reward
                self.log_train(train_infos, total_num_steps)

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

        self.writter.export_scalars_to_json(str(self.log_dir + '/summary.json'))
        self.writter.close()

        return reward_curve

    def warmup(self):
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]))

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = actions
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # Ensure rewards/dones are proper float arrays (handle mixed scalar/array from env)
        rewards = np.array(rewards, dtype=np.float32)
        if rewards.ndim == 2:
            rewards = rewards[:, :, np.newaxis]  # (threads, agents) -> (threads, agents, 1)
        dones = np.array(dones, dtype=bool)

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                          actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        time_now = time.strftime('%y%m_%d%H%M')
        policy_actor = self.trainer.policy.actor
        model_file_dir = os.path.join(self.save_dir, f'{self.algo_name}_{time_now}')
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir)
        torch.save(policy_actor.state_dict(), str(model_file_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(model_file_dir) + "/critic.pt")

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
