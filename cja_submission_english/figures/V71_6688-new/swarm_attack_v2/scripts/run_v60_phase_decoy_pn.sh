#!/bin/bash
# V60: phase-gated observation + decoy-cover reward + PN-like terminal guidance.
# User hypothesis:
#   1) If offensive-HVT distance > nearest defender-HVT distance, the aircraft is
#      still penetrating the defense line. Policy should mostly observe threats
#      and learn to avoid/distract defenders.
#   2) If offensive-HVT distance <= nearest defender-HVT distance, the aircraft
#      has crossed the defense line. Policy should only observe its own HVT LOS
#      angular rates and learn PN-like terminal guidance to hit the HVT.
# Implementation:
#   - scripts.phase_obs_wrapper masks policy observations without changing env
#     dynamics, hit/kill geometry, action space, or observation dimensionality.
#   - FOV_REWARD_PROFILE=v60phase enables primary striker + decoy lock shaping.
# Resume from current SOTA v59b u22 snapshot.
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

export FOV_REWARD_PROFILE=v60phase
export FOV_OBS_PHASE_MASK=v60_phase
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V60] start=$(date) model_dir=$RESUME_DIR reward=$FOV_REWARD_PROFILE obs_mask=$FOV_OBS_PHASE_MASK"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
  --algorithm_name mappo \
  --experiment_name v60_phase_decoy_pn \
  --scenario scenario_1 \
  --ap_config v28 \
  --obs_phase_mask v60_phase \
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
  --lr 1.0e-5 \
  --critic_lr 2.5e-5 \
  --entropy_coef 0.012 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param 0.08 \
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
  --seed 6001 \
  --model_dir "$RESUME_DIR"
