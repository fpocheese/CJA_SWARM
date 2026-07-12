#!/bin/bash
# V62: hard phase split + terminal-only range attraction.
# Penetration phase stays threat/decoy/progress oriented. Terminal phase still
# masks observation to HVT LOS angular rates, but reward adds terminal-only
# dense/near-strike shaping so the attack stage has enough pull toward 5m.
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

export FOV_REWARD_PROFILE=v62terminaldense
export FOV_OBS_PHASE_MASK=v60_phase
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[V62] start=$(date) model_dir=$RESUME_DIR reward=$FOV_REWARD_PROFILE obs_mask=$FOV_OBS_PHASE_MASK"
conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py   --algorithm_name mappo   --experiment_name v62_terminal_dense_pn   --scenario scenario_1   --ap_config v28   --obs_phase_mask v60_phase   --cuda   --n_rollout_threads 40   --n_training_threads 4   --n_eval_rollout_threads 10   --num_env_steps 200000000   --episode_length 8000   --ppo_epoch 2   --use_value_active_masks   --hidden_size 256   --layer_N 3   --lr 3.0e-6   --critic_lr 1.5e-5   --entropy_coef 0.016   --gamma 0.99   --gae_lambda 0.95   --clip_param 0.05   --value_loss_coef 1.0   --num_mini_batch 4   --max_grad_norm 0.4   --use_max_grad_norm   --use_eval   --eval_interval 5   --eval_episodes 10   --log_interval 1   --save_interval 1   --user_name fov_team   --use_ReLU   --use_feature_normalization   --use_orthogonal   --gain 0.01   --data_chunk_length 25   --use_recurrent_policy   --std_x_coef 1.0   --std_y_coef 0.2   --seed 6201   --model_dir "$RESUME_DIR"
