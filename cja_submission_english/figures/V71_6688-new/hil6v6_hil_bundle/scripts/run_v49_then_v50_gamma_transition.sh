#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs_remote

echo "[CHAIN] Stage 1: v49 critic warmstart"
bash scripts/run_v49_gamma_critic_warmstart.sh > logs_remote/v49_gamma_critic_warmstart.out 2>&1

V49_MODEL_DIR="outputs/results/fov_penetration/mappo/v49_gamma_critic_warmstart/run1/models"
if [[ ! -f "$V49_MODEL_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] Missing v49 checkpoint: $V49_MODEL_DIR"
    exit 1
fi

echo "[CHAIN] Stage 2: launch v50 actor resume"
nohup bash scripts/run_v50_gamma_actor_resume.sh > logs_remote/v50_gamma_actor_resume.out 2>&1 < /dev/null &
echo "[CHAIN] v50 launcher pid=$!"