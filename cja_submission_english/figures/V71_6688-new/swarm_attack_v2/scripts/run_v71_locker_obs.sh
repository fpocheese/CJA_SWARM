#!/bin/bash
# V71 (experiment name v71_locker_obs):
# New obs channel: primary_locker block (7 dims, positions 19:26).
# obs_dim: 23 -> 30. Teaches each attacker to perceive and break through
# the specific interceptor currently locking it.
#
# Architecture change (obs_dim 23->30) is incompatible with v70 weights, so
# training starts from scratch with slightly higher lr for faster early convergence.
#
# Frozen env constants (det=500, lock=500, pn_gain=3, lock_persist=200,
# policies_interceptor.py canonical) are NOT touched.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

PPO_EPOCH="${PPO_EPOCH:-1}"
LR="${LR:-3.0e-5}"
CRITIC_LR="${CRITIC_LR:-3.0e-4}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
CLIP_PARAM="${CLIP_PARAM:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
RUN_SEED="${RUN_SEED:-7101}"

export FOV_REWARD_PROFILE=v69teamsurvive
export FOV_OBS_PHASE_MASK=v65_strict_los
export FOV_TERMINAL_GUIDANCE=pn_los
export FOV_TERMINAL_PN_GAIN=3.0
export FOV_TERMINAL_PN_MAX_ACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V71] start=$(date) reward=$FOV_REWARD_PROFILE obs_mask=$FOV_OBS_PHASE_MASK lr=$LR entropy=$ENTROPY_COEF obs_dim=30 (primary_locker +7)"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v71_locker_obs \
  --scenario scenario_1 \
  --ap_config v28 \
  --obs_phase_mask v65_strict_los \
  --terminal_guidance pn_los \
  --terminal_pn_gain 3.0 \
  --terminal_pn_max_action 0.8 \
  --cuda \
  --n_rollout_threads 40 \
  --n_training_threads 4 \
  --n_eval_rollout_threads 10 \
  --num_env_steps 200000000 \
  --episode_length 8000 \
  --ppo_epoch "$PPO_EPOCH" \
  --use_value_active_masks \
  --hidden_size 256 \
  --layer_N 3 \
  --lr "$LR" \
  --critic_lr "$CRITIC_LR" \
  --entropy_coef "$ENTROPY_COEF" \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param "$CLIP_PARAM" \
  --value_loss_coef 0.5 \
  --num_mini_batch 4 \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --use_max_grad_norm \
  --use_eval \
  --eval_interval 1 \
  --eval_episodes 3 \
  --log_interval 1 \
  --save_interval 1 \
  --user_name fov_team \
  --use_ReLU \
  --use_feature_normalization \
  --use_orthogonal \
  --gain 0.01 \
  --data_chunk_length 25 \
  --use_recurrent_policy \
  --std_x_coef 1.0 \
  --std_y_coef 0.2 \
  --seed "$RUN_SEED" \
  2>&1 | tee logs_remote/v71_locker_obs.log
