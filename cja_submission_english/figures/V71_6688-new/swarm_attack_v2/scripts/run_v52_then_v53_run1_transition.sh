#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs_remote

echo "[CHAIN] Stage 1: v52 critic warmstart"
bash scripts/run_v52_run1_critic_warmstart.sh > logs_remote/v52_run1_critic_warmstart.out 2>&1

V52_MODEL_DIR="outputs/results/fov_penetration/mappo/v52_run1_critic_warmstart/run1/models"
if [[ ! -f "$V52_MODEL_DIR/actor_agent3.pt" ]]; then
    echo "[FATAL] Missing v52 checkpoint: $V52_MODEL_DIR"
    exit 1
fi

echo "[CHAIN] Stage 2: launch v53 actor resume"
nohup bash scripts/run_v53_run1_actor_resume.sh > logs_remote/v53_run1_actor_resume.out 2>&1 < /dev/null &
echo "[CHAIN] v53 launcher pid=$!"