#!/bin/bash
# V59: Minimal-change resume from v45.
# After V58 stage-gated reward regressed v45 from closed_mean=1670m to 563m (and
# killed agents mid-flight), revert to V45 base reward and only add a small
# near-strike bonus (lambda=10, sigma=30, active<=200m). No stage scaling, no
# lateral-miss, no role split — keeps the v45 gradient intact and only nudges
# the terminal 200m basin toward an actual hit.
# Environment is frozen (dt=0.01, max_steps=8000, hit/collision thresholds=5.0).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

V45_MODEL_DIR="outputs/results/fov_penetration/mappo/v45_kill_heading_freebie/run1/models"
if [[ ! -d "$V45_MODEL_DIR" ]] || [[ ! -f "$V45_MODEL_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] V45 run1 model dir incomplete: $V45_MODEL_DIR"
    exit 1
fi

export FOV_REWARD_PROFILE=v59
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V59] start=$(date) model_dir=$V45_MODEL_DIR profile=$FOV_REWARD_PROFILE"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v59_minimal_strike \
  --scenario scenario_1 \
  --ap_config v28 \
  --cuda \
  --n_rollout_threads 40 \
  --n_training_threads 4 \
  --n_eval_rollout_threads 10 \
  --num_env_steps 200000000 \
  --episode_length 8000 \
  --ppo_epoch 3 \
  --use_value_active_masks \
  --hidden_size 256 \
  --layer_N 3 \
  --lr 1.5e-5 \
  --critic_lr 4.0e-5 \
  --entropy_coef 0.006 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param 0.10 \
  --value_loss_coef 1.0 \
  --num_mini_batch 4 \
  --max_grad_norm 0.5 \
  --use_max_grad_norm \
  --use_eval \
  --eval_interval 5 \
  --eval_episodes 10 \
  --log_interval 1 \
  --save_interval 5 \
  --user_name fov_team \
  --use_ReLU \
  --use_feature_normalization \
  --use_orthogonal \
  --gain 0.01 \
  --data_chunk_length 25 \
  --use_recurrent_policy \
  --std_x_coef 1.0 \
  --std_y_coef 0.2 \
  --seed 5901 \
  --model_dir "$V45_MODEL_DIR"
