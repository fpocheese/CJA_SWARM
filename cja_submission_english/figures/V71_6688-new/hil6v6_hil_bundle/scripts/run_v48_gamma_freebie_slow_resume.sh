#!/bin/bash
# V48: keep V47 reward change, but make resume update much gentler.
#   Evidence:
#   1) v46 and v47 both collapsed from the v45 run3 ~180m regime to >1km min_d
#      after the very first update.
#   2) restore() only loads actor/critic weights; optimizer state is recreated,
#      so the first PPO update after reward swap is the dangerous step.
#   Strategy:
#      - keep the better-targeted reward change (gamma_align 0.2 -> 0.05)
#      - resume from v45 run3, not from regressed v46/v47 checkpoints
#      - sharply reduce update aggressiveness: lower LR, smaller clip, fewer PPO epochs
source ~/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

V45_MODEL_DIR="outputs/results/fov_penetration/mappo/v45_kill_heading_freebie/run3/models"
if [[ ! -d "$V45_MODEL_DIR" ]] || [[ ! -f "$V45_MODEL_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] V45 model dir incomplete: $V45_MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo " V48: slow resume on gamma fix"
echo " Resume from: $V45_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v48_gamma_freebie_slow_resume \
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
    --lr 1.0e-5 \
    --critic_lr 2.0e-5 \
    --entropy_coef 0.003 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.08 \
    --value_loss_coef 1.0 \
    --num_mini_batch 8 \
    --max_grad_norm 0.5 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 5 \
    --eval_episodes 20 \
    --log_interval 1 \
    --save_interval 10 \
    --user_name fov_team \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --data_chunk_length 25 \
    --use_recurrent_policy \
    --std_x_coef 1.0 \
    --std_y_coef 0.2 \
    --seed 4801 \
    --model_dir "$V45_MODEL_DIR" \
    2>&1 | tee outputs/v48_gamma_freebie_slow_resume.log