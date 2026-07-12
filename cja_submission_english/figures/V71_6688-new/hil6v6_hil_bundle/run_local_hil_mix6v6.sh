#!/usr/bin/env bash
set -euo pipefail

# Launch 6v6 split-HIL where 5 policy nodes are local and 1 policy node runs on NX.
# Usage:
#   ./run_local_hil_mix6v6.sh <local_pc_ipv4> [port] [episodes] [out_json]
#
# Example:
#   ./run_local_hil_mix6v6.sh 192.168.1.100 5500 200 /tmp/v71_hil_mix6v6_summary.json

LOCAL_IP="${1:?Usage: $0 <local_pc_ipv4> [port] [episodes] [out_json]}"
PORT="${2:-5500}"
EPISODES="${3:-200}"
OUT_JSON="${4:-/tmp/v71_hil_mix6v6_hw_summary.json}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/.hil6v6_local_logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models}"

echo "[local] starting server on 0.0.0.0:${PORT} (episodes=${EPISODES})"
${PYTHON_BIN} "${ROOT_DIR}/hil_v71_split/hil_env_server.py" \
  --case 6v6 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --episodes "${EPISODES}" \
  --max-steps 8000 \
  --out "${OUT_JSON}" \
  >"${LOG_DIR}/hil_env_server.log" 2>&1 &
SERVER_PID=$!
echo "[local] server pid=${SERVER_PID}"

echo "[local] start 5 local policy nodes (agent 1..5)"
for AGENT_ID in 1 2 3 4 5; do
  SOURCE_AGENT=$((AGENT_ID % 4))
  ${PYTHON_BIN} "${ROOT_DIR}/hil_v71_split/hil_policy_node.py" \
    --agent-id "${AGENT_ID}" \
    --source-agent "${SOURCE_AGENT}" \
    --server-host "${LOCAL_IP}" \
    --server-port "${PORT}" \
    --model-dir "${MODEL_DIR}" \
    >"${LOG_DIR}/hil_node_${AGENT_ID}.log" 2>&1 &
  echo "[local] node ${AGENT_ID} pid=$!"
done

echo "[local] Waiting for server to finish all episodes..."
echo "[local] After this, launch NX node in separate shell:"
echo "     ./run_hil_mix6v6_nx.sh ${LOCAL_IP} ${PORT} ${MODEL_DIR}"
echo "[local] server log: ${LOG_DIR}/hil_env_server.log"

trap 'echo "[local] received interrupt; stopping server and nodes"; pkill -P ${SERVER_PID} 2>/dev/null || true; kill ${SERVER_PID} 2>/dev/null || true' INT TERM
wait "${SERVER_PID}"
echo "[local] finished. summary: ${OUT_JSON}"
