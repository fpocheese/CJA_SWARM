"""
改进版MPE Runner - 用于训练Advanced MAPPO
"""
import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    """
    改进的Runner类，用于Advanced MAPPO训练
    """
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        """主训练循环"""
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        print(f"\n开始训练 - 总共 {episodes} 个episodes")
        print(f"每个episode长度: {self.episode_length}")
        print(f"并行环境数: {self.n_rollout_threads}")
        print(f"总训练步数: {self.num_env_steps}\n")

        for episode in range(episodes):
            # 学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # 热身采样
            self.warmup()

            # Episode训练
            for step in range(self.episode_length):
                # 采样动作
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # 环境交互
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                # 存储数据
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            train_infos = self.train()
            
            # 后处理
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # 保存模型
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # 日志记录
            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start))
                
                # 打印训练信息
                print("\n" + "="*80)
                print(f"Scenario: {self.all_args.scenario_name} | Algo: {self.algorithm_name}")
                print(f"Episode: {episode}/{episodes} | Steps: {total_num_steps}/{self.num_env_steps}")
                print(f"FPS: {fps}")
                
                # 计算平均回报
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print(f"Average Episode Rewards: {train_infos['average_episode_rewards']:.2f}")
                
                # 打印改进算法的额外信息
                if 'kl_divergence' in train_infos:
                    print(f"KL Divergence: {train_infos['kl_divergence']:.6f}")
                if 'kl_coef' in train_infos:
                    print(f"KL Coefficient: {train_infos['kl_coef']:.6f}")
                if 'is_warmup' in train_infos:
                    print(f"Value Warmup: {train_infos['is_warmup']}")
                if 'current_episode' in train_infos:
                    print(f"Current Episode: {train_infos['current_episode']}")
                
                print("="*80 + "\n")
                
                # 记录到wandb或tensorboard
                self.log_train(train_infos, total_num_steps)

            # 评估
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        """预热buffer"""
        # 重置环境
        obs = self.envs.reset()

        # 初始化buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """采集数据"""
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step])
            )
        
        # 分割数据
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # 转换为环境动作
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        """插入数据到buffer"""
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # 重置RNN状态
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), 
            dtype=np.float32
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), 
            dtype=np.float32
        )
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # 集中式观察
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # 插入buffer
        self.buffer.insert(
            share_obs, 
            obs, 
            rnn_states, 
            rnn_states_critic, 
            actions, 
            action_log_probs, 
            values, 
            rewards, 
            masks
        )

    @torch.no_grad()
    def compute(self):
        """计算returns"""
        self.trainer.prep_rollout()
        
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1])
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """训练策略"""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """保存模型"""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """恢复模型"""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """记录训练信息"""
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """记录环境信息"""
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """评估策略"""
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), 
            dtype=np.float32
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), 
            dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # 转换动作
            eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)

            # 环境step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # 重置mask
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), 
                dtype=np.float32
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), 
                dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), 
                dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        
        print(f"Evaluation average episode rewards: {eval_average_episode_rewards:.2f}")
        
        self.log_env(eval_env_infos, total_num_steps)
