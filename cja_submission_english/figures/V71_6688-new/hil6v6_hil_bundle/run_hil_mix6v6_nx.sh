#!/usr/bin/env bash
set -euo pipefail

# Run one policy node on NX and connect to local HIL server.
# Usage:
#   ./run_hil_mix6v6_nx.sh <local_pc_ipv4> [port] [model_dir]
#
# Example:
#   ./run_hil_mix6v6_nx.sh 192.168.1.100 5500 outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models

LOCAL_IP="${1:?Usage: $0 <local_pc_ipv4> [port] [model_dir]}"
PORT="${2:-5500}"
MODEL_DIR="${3:-outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

${PYTHON_BIN} "${ROOT_DIR}/hil_v71_split/hil_policy_node.py" \
  --agent-id 0 \
  --source-agent 0 \
  --server-host "${LOCAL_IP}" \
  --server-port "${PORT}" \
  --model-dir "${MODEL_DIR}"
