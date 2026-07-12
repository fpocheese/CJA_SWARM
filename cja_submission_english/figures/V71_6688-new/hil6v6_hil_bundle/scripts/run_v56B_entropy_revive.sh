#!/bin/bash
# V56B: resume from v45 run1 with HIGH entropy + small lr to avoid the
#       v53-style actor collapse (avg_act became constant, herr=91deg).
#   Reward block: UNCHANGED v45 (no FOV_REWARD_PROFILE set).
#   Hyperparams: lr 1e-5, clip 0.10, entropy 0.02 (~7x v45's 0.003).
#   Designed to run in parallel with v56A on the same GPU.
#   Env still frozen (dt=0.01, hit=5m, max_steps=8000).
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

echo "=========================================="
echo " V56B: entropy revive resume from v45 run1"
echo " Reward profile: (none, vanilla v45)"
echo " Resume from: $V45_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

unset FOV_REWARD_PROFILE
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v56B_entropy_revive \
    --scenario scenario_1 \
    --ap_config v28 \
    --cuda \
    --n_rollout_threads 40 \
    --n_training_threads 4 \
    --n_eval_rollout_threads 10 \
    --num_env_steps 200000000 \
    --episode_length 8000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 1.0e-5 \
    --critic_lr 5.0e-5 \
    --entropy_coef 0.02 \
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
    --seed 5602 \
    --model_dir "$V45_MODEL_DIR"
