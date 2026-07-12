#!/bin/bash
# V61: hard phase split for the user-directed penetration/terminal design.
# Penetration phase: reward crossing/cover/decoy behavior and threat handling.
# Terminal phase: suppress legacy penetration/HVT-shaping rewards and train only
# PN-like HVT LOS-rate + terminal progress shaping. Action space and env physics
# are unchanged. Resume from the best early v60 phase snapshot (u6).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p outputs logs_remote

RESUME_DIR="outputs/results/fov_penetration/mappo/v60_phase_decoy_pn/run1/models_u6_snapshot"
if [[ ! -d "$RESUME_DIR" ]] || [[ ! -f "$RESUME_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] resume snapshot incomplete: $RESUME_DIR"
    exit 1
fi

export FOV_REWARD_PROFILE=v61hardphase
export FOV_OBS_PHASE_MASK=v60_phase
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V61] start=$(date) model_dir=$RESUME_DIR reward=$FOV_REWARD_PROFILE obs_mask=$FOV_OBS_PHASE_MASK"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py   --algorithm_name mappo   --experiment_name v61_hard_phase_pn   --scenario scenario_1   --ap_config v28   --obs_phase_mask v60_phase   --cuda   --n_rollout_threads 40   --n_training_threads 4   --n_eval_rollout_threads 10   --num_env_steps 200000000   --episode_length 8000   --ppo_epoch 2   --use_value_active_masks   --hidden_size 256   --layer_N 3   --lr 5.0e-6   --critic_lr 2.0e-5   --entropy_coef 0.014   --gamma 0.99   --gae_lambda 0.95   --clip_param 0.06   --value_loss_coef 1.0   --num_mini_batch 4   --max_grad_norm 0.5   --use_max_grad_norm   --use_eval   --eval_interval 5   --eval_episodes 10   --log_interval 1   --save_interval 1   --user_name fov_team   --use_ReLU   --use_feature_normalization   --use_orthogonal   --gain 0.01   --data_chunk_length 25   --use_recurrent_policy   --std_x_coef 1.0   --std_y_coef 0.2   --seed 6101   --model_dir "$RESUME_DIR"
