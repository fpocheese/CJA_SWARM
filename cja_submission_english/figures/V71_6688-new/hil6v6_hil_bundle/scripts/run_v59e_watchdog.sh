#!/bin/bash
# V59E: extreme-conservative protective fine-tune from v59b u22 SOTA.
# v59d (pure v45 reward, lr=3e-6, ppo_epoch=2, clip=0.05) already drifted
# closed_mean from 1225m -> 1036m by u22. Even tiny PPO updates shift the
# policy away from u22's edge attractor.
# v59e: smallest possible step that still updates: lr=1e-6, ppo_epoch=1,
# clip=0.03, entropy=0.001, value_loss_coef=0.5 (don't let critic drag actor).
# This is a "watchdog" run — main goal is to confirm whether u22 is truly
# the asymptote of the v45 reward, or merely a transient peak.
# Environment is frozen.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

RESUME_DIR="outputs/results/fov_penetration/mappo/v59b_gentle_strike/run1/models_u22_snapshot"
if [[ ! -d "$RESUME_DIR" ]] || [[ ! -f "$RESUME_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] resume snapshot incomplete: $RESUME_DIR"
    exit 1
fi

unset FOV_REWARD_PROFILE
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V59E] start=$(date) model_dir=$RESUME_DIR profile=<default v45>"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v59e_watchdog_u22 \
  --scenario scenario_1 \
  --ap_config v28 \
  --cuda \
  --n_rollout_threads 40 \
  --n_training_threads 4 \
  --n_eval_rollout_threads 10 \
  --num_env_steps 200000000 \
  --episode_length 8000 \
  --ppo_epoch 1 \
  --use_value_active_masks \
  --hidden_size 256 \
  --layer_N 3 \
  --lr 1.0e-6 \
  --critic_lr 5.0e-6 \
  --entropy_coef 0.001 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param 0.03 \
  --value_loss_coef 0.5 \
  --num_mini_batch 4 \
  --max_grad_norm 0.3 \
  --use_max_grad_norm \
  --use_eval \
  --eval_interval 5 \
  --eval_episodes 10 \
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
  --seed 5905 \
  --model_dir "$RESUME_DIR"
