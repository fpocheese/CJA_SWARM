#!/bin/bash
# V51: resume from the best surviving compatible checkpoint (v45 run1), not the
# later-regressed v45 run3.
#   Evidence under the current frozen env + current reward config:
#     - v44 run3: min_d_min ~= 361 m
#     - v45 run1: min_d_min ~= 306 m
#     - v45 run3: min_d_min ~= 1141 m
#   Hypothesis:
#     v45's heading-freebie cut helped at first, but continued high-aggression
#     PPO updates drifted the actor away from the better run1 regime.
#   Strategy:
#     - keep the current reward config as-is
#     - resume from v45 run1
#     - reduce actor update aggressiveness versus original v45
#     - eval/save every update so the first regression is visible immediately
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
echo " V51: slow recover from v45 run1"
echo " Resume from: $V45_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v51_run1_slow_recover \
    --scenario scenario_1 \
    --ap_config v28 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 8000 \
    --ppo_epoch 2 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 5.0e-6 \
    --critic_lr 2.0e-5 \
    --entropy_coef 0.003 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.06 \
    --value_loss_coef 1.0 \
    --num_mini_batch 8 \
    --max_grad_norm 0.5 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 1 \
    --eval_episodes 20 \
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
    --seed 5101 \
    --model_dir "$V45_MODEL_DIR" \
    2>&1 | tee outputs/v51_run1_slow_recover.log