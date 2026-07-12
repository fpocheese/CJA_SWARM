#!/bin/bash
# V59B: gentler version of V59.
# v59 results: u17 closed=1517m herr=36.7 (good), u41 closed=878m herr=44.2 (bad,
# 3/4 agents die ~step 3000). Conclusion: 10x near-strike + lr 1.5e-5 over many
# updates pulls policy toward terminal-aggressive maneuvers that kill the swarm
# before reaching HVT.
# v59b: same idea (terminal nudge only) but quartered bonus, requires non-trivial
# closing rate, lower lr, higher entropy, save_interval=2 to capture peak early.
# Resumes from V45 base (NOT v59 u41) to start from the surviving prior.
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

export FOV_REWARD_PROFILE=v59b
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V59B] start=$(date) model_dir=$V45_MODEL_DIR profile=$FOV_REWARD_PROFILE"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v59b_gentle_strike \
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
  --lr 1.0e-5 \
  --critic_lr 3.0e-5 \
  --entropy_coef 0.008 \
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
  --save_interval 2 \
  --user_name fov_team \
  --use_ReLU \
  --use_feature_normalization \
  --use_orthogonal \
  --gain 0.01 \
  --data_chunk_length 25 \
  --use_recurrent_policy \
  --std_x_coef 1.0 \
  --std_y_coef 0.2 \
  --seed 5902 \
  --model_dir "$V45_MODEL_DIR"
