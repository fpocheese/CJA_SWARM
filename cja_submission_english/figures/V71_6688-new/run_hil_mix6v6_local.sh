#!/usr/bin/env bash
set -euo pipefail

# Launch 6v6 split-HIL where only one attacker policy runs on NX.
# Usage:
#   ./run_hil_mix6v6_local.sh 192.168.1.20 5500 /tmp/v71_hil_mix6v6_hw_summary.json 200

LOCAL_IP="${1:-}"
PORT="${2:-5500}"
OUT_JSON="${3:-/tmp/v71_hil_mix6v6_hw_summary.json}"
EPISODES="${4:-200}"

if [[ -z "${LOCAL_IP}" ]]; then
  echo "Usage: $0 <local_pc_ip> [port] [out_json] [episodes]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ROOT_DIR}/.hil6v6_logs"
mkdir -p "${LOG_DIR}"

echo "[local] start server: --case 6v6 --host 0.0.0.0 --port ${PORT} --episodes ${EPISODES}"
python "${ROOT_DIR}/hil_v71_split/hil_env_server.py" \
  --case 6v6 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --episodes "${EPISODES}" \
  --max-steps 8000 \
  --out "${OUT_JSON}" \
  > "${LOG_DIR}/hil_env_server.log" 2>&1 &
SERVER_PID=$!

echo "[local] server pid=${SERVER_PID}"
sleep 2

for AGENT_ID in 1 2 3 4 5; do
  SOURCE_AGENT=$(( AGENT_ID % 4 ))
  echo "[local] start policy node agent ${AGENT_ID} source_agent ${SOURCE_AGENT}"
  python "${ROOT_DIR}/hil_v71_split/hil_policy_node.py" \
    --agent-id "${AGENT_ID}" \
    --source-agent "${SOURCE_AGENT}" \
    --server-host "${LOCAL_IP}" \
    --server-port "${PORT}" \
    --model-dir "${ROOT_DIR}/outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models" \
    > "${LOG_DIR}/hil_node_${AGENT_ID}.log" 2>&1 &
done

echo "[local] local 5 nodes started. Now start NX node 0 separately:"
echo "python hil_v71_split/hil_policy_node.py --agent-id 0 --source-agent 0 --server-host ${LOCAL_IP} --server-port ${PORT} --model-dir ... "
echo "waiting for server/client completion..."

wait "${SERVER_PID}"
echo "[local] finished"
