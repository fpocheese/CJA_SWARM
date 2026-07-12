#!/bin/bash
# V52: critic-only warmstart from the best surviving checkpoint under the
# current reward config.
#   Evidence:
#     - v45 run1 is the best compatible restore point found so far
#     - direct actor updates in v51 already regressed that regime on update 0
#   Hypothesis:
#     the first update should adapt the critic to the current reward while the
#     actor remains frozen, then actor training can resume from a safer base.
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
echo " V52: critic warmstart from v45 run1"
echo " Resume from: $V45_MODEL_DIR"
echo " Start: $(date)"
echo "=========================================="

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v52_run1_critic_warmstart \
    --scenario scenario_1 \
    --ap_config v28 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 640000 \
    --episode_length 8000 \
    --ppo_epoch 2 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 0.0 \
    --critic_lr 5.0e-5 \
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
    --seed 5201 \
    --model_dir "$V45_MODEL_DIR" \
    2>&1 | tee outputs/v52_run1_critic_warmstart.log