#!/usr/bin/env bash
set -euo pipefail

# Start one attacker policy node on NX (agent 0) and connect to local HIL server.
# Example:
#   ssh nx_user@nx_ip "cd /path/to/repo && ./run_hil_mix6v6_nx.sh 192.168.1.20 5500 /absolute/path/to/models"

LOCAL_IP="${1:-}"
PORT="${2:-5500}"
MODEL_DIR="${3:-outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models}"

if [[ -z "${LOCAL_IP}" ]]; then
  echo "Usage: $0 <local_pc_ip> [port] [model_dir]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "${ROOT_DIR}/hil_v71_split/hil_policy_node.py" \
  --agent-id 0 \
  --source-agent 0 \
  --server-host "${LOCAL_IP}" \
  --server-port "${PORT}" \
  --model-dir "${MODEL_DIR}"
