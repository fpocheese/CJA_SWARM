#!/bin/bash
# V59D: "protective" training from v59b u22 SOTA.
# Pattern across v58/v59/v59b/v59c: any non-trivial near-strike lambda + sustained
# PPO drifts policy away from u22 SOTA. v59c (lambda=2, lr=5e-6) regressed by
# u84 (closed=703m vs u22=1225m). Conclusion: u22 is currently the best
# achievable point with the current reward family.
# v59d uses the *original* v45 reward (lambda_near_strike=0, no near-strike
# bonus at all) and ultra-slow lr to avoid GPU idle while not regressing u22.
# Resume from u22 snapshot. Save every update so we can pick the best on diag.
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

# Intentionally NO FOV_REWARD_PROFILE -> default v45 reward (lambda_near_strike=0).
unset FOV_REWARD_PROFILE
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V59D] start=$(date) model_dir=$RESUME_DIR profile=<default v45>"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v59d_protect_u22 \
  --scenario scenario_1 \
  --ap_config v28 \
  --cuda \
  --n_rollout_threads 40 \
  --n_training_threads 4 \
  --n_eval_rollout_threads 10 \
  --num_env_steps 200000000 \
  --episode_length 8000 \
  --ppo_epoch 2 \
  --use_value_active_masks \
  --hidden_size 256 \
  --layer_N 3 \
  --lr 3.0e-6 \
  --critic_lr 1.0e-5 \
  --entropy_coef 0.005 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param 0.05 \
  --value_loss_coef 1.0 \
  --num_mini_batch 4 \
  --max_grad_norm 0.5 \
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
  --seed 5904 \
  --model_dir "$RESUME_DIR"
