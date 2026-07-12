#!/bin/bash
# V47: revert dense sigma to 200 and cut gamma_align 0.2 -> 0.05
#   Hypothesis: V46 proved sigma 120 was too aggressive and removed too much
#   end-game attraction; the real remaining white-leech is gamma_align.
#   Remote reward breakdown on zero-action spawn step showed gamma_align alone
#   contributes about +0.198/step, larger than approach (+0.022), closing
#   (+0.056), and team progress (+0.030). Remove that free reward while keeping
#   the broader dense signal that let V45 reach the ~180m regime.
#   Resume from v45 run3, not v46, because v46 update0 regressed badly.
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
echo " V47: kill gamma_align freebie (0.2 -> 0.05)"
echo " Resume from: $V45_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v47_kill_gamma_freebie \
    --scenario scenario_1 \
    --ap_config v28 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 8000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 5.0e-5 \
    --critic_lr 5.0e-5 \
    --entropy_coef 0.003 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.15 \
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
    --seed 4701 \
    --model_dir "$V45_MODEL_DIR" \
    2>&1 | tee outputs/v47_kill_gamma_freebie.log