import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_advanced import R_Actor_Advanced, R_Critic_Advanced
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy_Advanced:
    """
    改进的MAPPO策略类，包含先进的Actor和Critic网络
    
    主要改进:
    1. 使用带注意力机制的Actor和Critic
    2. 添加残差连接增强特征传递
    3. 支持更灵活的学习率调度
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 使用改进的Actor和Critic
        self.actor = R_Actor_Advanced(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_Advanced(args, self.share_obs_space, self.device)

        # 使用AdamW优化器（比Adam更好的权重衰减）
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.lr, 
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器 - 使用余弦退火
        self.use_lr_scheduler = getattr(args, 'use_lr_scheduler', False)
        if self.use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, 
                T_max=getattr(args, 'num_env_steps', 10000000) // getattr(args, 'episode_length', 25),
                eta_min=self.lr * 0.1
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer,
                T_max=getattr(args, 'num_env_steps', 10000000) // getattr(args, 'episode_length', 25),
                eta_min=self.critic_lr * 0.1
            )

    def lr_decay(self, episode, episodes):
        """
        学习率衰减
        """
        if self.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        else:
            update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
            update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        计算动作和价值函数预测
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        获取价值函数预测
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        评估动作，获取对数概率、熵和价值函数预测
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks
        )

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        仅获取动作（用于推理）
        """
        actions, _, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic
        )
        return actions, rnn_states_actor
