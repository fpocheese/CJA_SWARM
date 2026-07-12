#!/bin/bash
# 改进版MAPPO训练脚本 - 针对固定翼集群拦截任务

env="MPE"
scenario="simple_world_comm"  # 修改为你的场景名称
num_landmarks=3
num_agents=20  # 固定翼数量（拦截方+目标方）
algo="rmappo"  # 或 "mappo"
exp="advanced_mappo_uav_interception"
seed=1

# 训练参数
n_training_threads=1
n_rollout_threads=32
num_mini_batch=1
episode_length=600  # 根据你的任务调整
num_env_steps=10000000
ppo_epoch=10
save_interval=25
log_interval=5

# 网络参数
hidden_size=256  # 增大网络容量
layer_N=2
use_ReLU=True
use_feature_normalization=True
use_orthogonal=True
gain=0.01

# PPO参数
lr=5e-4
critic_lr=5e-4
opti_eps=1e-5
weight_decay=0
clip_param=0.2
entropy_coef=0.01
value_loss_coef=1
use_max_grad_norm=True
max_grad_norm=10.0
use_clipped_value_loss=True
use_huber_loss=True
use_value_active_masks=True
use_policy_active_masks=True
huber_delta=10.0

# GAE参数
use_gae=True
gamma=0.99
gae_lambda=0.95

# 改进的算法参数
use_attention=True  # 使用注意力机制
use_residual=True  # 使用残差连接
use_dual_clip=True  # 使用Dual-clip PPO
dual_clip_param=3.0
use_adaptive_kl=True  # 使用自适应KL
target_kl=0.02
use_value_warmup=True  # 使用价值函数热身
warmup_episodes=100
use_lr_scheduler=False  # 是否使用学习率调度器

# 其他参数
use_recurrent_policy=True
use_naive_recurrent_policy=False
recurrent_N=1
data_chunk_length=10
use_valuenorm=True
use_popart=False
use_linear_lr_decay=False

# wandb参数（可选）
use_wandb=False
user_name="uav_research"

echo "================================="
echo "Training Advanced MAPPO for UAV Interception"
echo "Scenario: ${scenario}"
echo "Num Agents: ${num_agents}"
echo "Improvements: Attention + Residual + Dual-clip + Adaptive KL"
echo "================================="

python ../train/train_mpe_advanced.py \
--env_name ${env} \
--algorithm_name ${algo} \
--experiment_name ${exp} \
--scenario_name ${scenario} \
--num_agents ${num_agents} \
--num_landmarks ${num_landmarks} \
--seed ${seed} \
--n_training_threads ${n_training_threads} \
--n_rollout_threads ${n_rollout_threads} \
--num_mini_batch ${num_mini_batch} \
--episode_length ${episode_length} \
--num_env_steps ${num_env_steps} \
--ppo_epoch ${ppo_epoch} \
--save_interval ${save_interval} \
--log_interval ${log_interval} \
--hidden_size ${hidden_size} \
--layer_N ${layer_N} \
--lr ${lr} \
--critic_lr ${critic_lr} \
--opti_eps ${opti_eps} \
--weight_decay ${weight_decay} \
--clip_param ${clip_param} \
--entropy_coef ${entropy_coef} \
--value_loss_coef ${value_loss_coef} \
--max_grad_norm ${max_grad_norm} \
--huber_delta ${huber_delta} \
--gamma ${gamma} \
--gae_lambda ${gae_lambda} \
--use_attention ${use_attention} \
--use_residual ${use_residual} \
--use_dual_clip ${use_dual_clip} \
--dual_clip_param ${dual_clip_param} \
--use_adaptive_kl ${use_adaptive_kl} \
--target_kl ${target_kl} \
--use_value_warmup ${use_value_warmup} \
--warmup_episodes ${warmup_episodes} \
--use_lr_scheduler ${use_lr_scheduler} \
--use_recurrent_policy ${use_recurrent_policy} \
--use_ReLU ${use_ReLU} \
--use_feature_normalization ${use_feature_normalization} \
--use_orthogonal ${use_orthogonal} \
--gain ${gain} \
--use_max_grad_norm ${use_max_grad_norm} \
--use_clipped_value_loss ${use_clipped_value_loss} \
--use_huber_loss ${use_huber_loss} \
--use_value_active_masks ${use_value_active_masks} \
--use_policy_active_masks ${use_policy_active_masks} \
--use_gae ${use_gae} \
--use_valuenorm ${use_valuenorm} \
--use_popart ${use_popart} \
--use_linear_lr_decay ${use_linear_lr_decay} \
--use_naive_recurrent_policy ${use_naive_recurrent_policy} \
--recurrent_N ${recurrent_N} \
--data_chunk_length ${data_chunk_length} \
--use_wandb ${use_wandb} \
--user_name ${user_name} \
--cuda
