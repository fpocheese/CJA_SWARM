# V71 MAPPO HIL (NX + 本地混合) 操作 Skill

用于“链接 NX / 部署代码 / 运行仿真 / 画图”的统一复用手册。  
本文件可直接作为后续复现该流程的标准说明。

## 0) 远端代码源与链接方式（原始代码源）

1. 远端服务器 SSH：
   - 用户名：`a2rl`
   - 地址：`192.168.1.91`
   - 密码：`123456`
   - 推荐连接命令（密码方式）：
     - `sshpass -p '123456' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_192_168_1_91 a2rl@192.168.1.91`
2. 远端代码目录（原始工程目录）：
   - `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2/`
3. 当前会话默认模型：
   - MAPPO V71 成功模型：`outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`
   - 关键权重文件：`actor_agent{0..3}.pt`

## 1) 本地环境假设

1. 本机代码根目录（当前会话）：  
   `/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new`
2. Python 环境：`/home/uav/anaconda3/envs/rlgpu/bin/python`
3. 本机 HIL 服务监听（示例）：  
   - IP：`192.168.47.92`
   - 端口：`5530`
4. 模型目录应在本机和 NX 上都可见（本 skill 采用下发 bundle 到 NX）。

## 2) NX 登录与网段分配（按你当前连接约定）

1. NX 用户/密码：
   - 用户名：`nvidia`
   - 密码：`nvidia`
2. 当前会话已测通的可达 IP（可变）：
   - `192.168.47.101`
   - `192.168.47.103`
   - `192.168.47.104`
   - `192.168.47.105`
   - `192.168.47.106`
3. 连接检查：
   - `ping -c 1 -W 1 <ip>`
   - `sshpass -p 'nvidia' ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o PasswordAuthentication=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 nvidia@<ip> "echo SSH_OK"`

## 3) 从远端同步最新代码到本地（建议先做）

```bash
cd /home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new

sshpass -p '123456' rsync -avz --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'outputs' \
  a2rl@192.168.1.91:/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2/ \
  ./swarm_attack_v2/
```

> 说明：如已本地已有代码可跳过此步；`--exclude` 仅示例，必要时可删掉避免遗漏文件。

## 4) 生成 NX 侧部署包（V71 MAPPO + HIL 侧依赖）

```bash
cd /home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new

cat > /tmp/v71_hil_nx_bundle.list <<'EOF'
swarm_attack_v2/hil_v71_split
swarm_attack_v2/outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models
swarm_attack_v2/third_party/MACPO/MACPO/macpo
swarm_attack_v2/scripts/phase_obs_wrapper.py
swarm_attack_v2/scripts/terminal_pn_action_wrapper.py
swarm_attack_v2/scripts/phase_obs.py
EOF

tar -czf /tmp/v71_hil_nx_bundle.tgz -T /tmp/v71_hil_nx_bundle.list
```

> 该包仅保留本次 HIL 推理必须文件，减少传输与部署耗时。若运行时报缺文件，需把缺失目录临时补充进 tar 包。

## 5) 一次性部署到全部 NX（可按需改为 1~5 台）

```bash
NX_IPS=(192.168.47.101 192.168.47.103 192.168.47.104 192.168.47.105 192.168.47.106)

for ip in "${NX_IPS[@]}"; do
  sshpass -p 'nvidia' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts_nx \
    nvidia@"$ip" "mkdir -p /home/nvidia/swarm_attack_v2"

  sshpass -p 'nvidia' scp -o StrictHostKeyChecking=no /tmp/v71_hil_nx_bundle.tgz \
    nvidia@"$ip":/tmp/v71_hil_nx_bundle.tgz

  sshpass -p 'nvidia' ssh -o StrictHostKeyChecking=no nvidia@"$ip" "\
    cd /home/nvidia/swarm_attack_v2 && \
    tar -xzf /tmp/v71_hil_nx_bundle.tgz -C /home/nvidia"

  # 避免 MACPO 导入依赖路径冲突，统一清空 __init__.py（你当前会话的兼容性处理）
  sshpass -p 'nvidia' ssh -o StrictHostKeyChecking=no nvidia@"$ip" \
    "mkdir -p /home/nvidia/swarm_attack_v2/third_party/MACPO/MACPO/macpo && : > /home/nvidia/swarm_attack_v2/third_party/MACPO/MACPO/macpo/__init__.py"
done
```

## 6) HIL 运行主流程（1/5/... NX 混合）

### 6.1 规则
- 本机作为环境服务端（`hil_env_server.py`）
- 前 `N` 个进攻飞行器放到 NX；其余在本地运行 policy node
- 目标 case 使用 V71 成功种子：  
  - `6v6` `seed=60015`  
  - `8v8` `seed=80047`  
  - `10v10` `seed=100015`

### 6.2 启动单工况示例（N=5）

```bash
PY=/home/uav/anaconda3/envs/rlgpu/bin/python
CDIR=/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/swarm_attack_v2
HOST=192.168.47.92
PORT=5530
NX_IPS=(192.168.47.101 192.168.47.103 192.168.47.104 192.168.47.105 192.168.47.106)
OUTDIR=/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/hil_outputs/manual_case
mkdir -p "$OUTDIR/6v6_seed60015"

nohup $PY hil_v71_split/hil_env_server.py \
  --case 6v6 --seed 60015 --episodes 1 --max-steps 8000 \
  --host "$HOST" --port "$PORT" \
  --full-output-dir "$OUTDIR/6v6_seed60015" \
  > "$OUTDIR/6v6_seed60015/server.log" 2>&1 &
SERVER_PID=$!

sleep 1

# 5个NX策略
for i in 0 1 2 3 4; do
  ip=${NX_IPS[$i]}
  sshpass -p 'nvidia' ssh -f -n nvidia@"$ip" \
    "cd /home/nvidia/swarm_attack_v2 && \
     PYTHONUNBUFFERED=1 $PY hil_v71_split/hil_policy_node.py \
     --agent-id ${i} --source-agent $((i % 4)) \
     --server-host $HOST --server-port $PORT > /tmp/hil_node_${i}.log 2>&1"
done

# 本地策略：剩余agent（例：n_agents=6 时，本地为5）
$PY hil_v71_split/hil_policy_node.py \
  --agent-id 5 --source-agent 1 --server-host "$HOST" --server-port "$PORT" \
  > "$OUTDIR/6v6_seed60015/local_agent5.log" 2>&1

wait $SERVER_PID
```

### 6.3 三工况自动化（6v6/8v8/10v10）

把下面内容作为最小脚本 `run_three_cases_5nx.sh` 后执行即可：

```bash
#!/usr/bin/env bash
set -euo pipefail

PY=/home/uav/anaconda3/envs/rlgpu/bin/python
cd /home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/swarm_attack_v2
HOST=192.168.47.92
PORT=5530
OUT_ROOT=/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/hil_outputs/original_success_5nx_restlocal
NX_IPS=(192.168.47.101 192.168.47.103 192.168.47.104 192.168.47.105 192.168.47.106)

run_case() {
  local CASE=$1 SEED=$2 N_AGENTS=$3
  local TAG="${CASE}_seed${SEED}"
  mkdir -p "$OUT_ROOT/$TAG"

  nohup $PY hil_v71_split/hil_env_server.py \
    --case "$CASE" --seed "$SEED" --episodes 1 --max-steps 8000 \
    --host "$HOST" --port "$PORT" \
    --full-output-dir "$OUT_ROOT/$TAG" \
    > "$OUT_ROOT/$TAG/server.log" 2>&1 &
  SERVER_PID=$!

  sleep 1

  local nx_cnt=$((N_AGENTS-1))
  for ((i=0;i<nx_cnt;i++)); do
    ip=${NX_IPS[$i]}
    sshpass -p 'nvidia' ssh -f -n nvidia@"$ip" \
      "cd /home/nvidia/swarm_attack_v2 && \
       PYTHONUNBUFFERED=1 $PY hil_v71_split/hil_policy_node.py \
       --agent-id $i --source-agent $((i%4)) \
       --server-host $HOST --server-port $PORT > /tmp/hil_node_${i}.log 2>&1"
  done

  for ((i=nx_cnt;i<N_AGENTS;i++)); do
    $PY hil_v71_split/hil_policy_node.py \
      --agent-id "$i" --source-agent "$((i%4))" \
      --server-host "$HOST" --server-port "$PORT" \
      > "$OUT_ROOT/$TAG/local_agent${i}.log" 2>&1 &
  done

  wait $SERVER_PID
  wait
}

run_case 6v6 60015 6
run_case 8v8 80047 8
run_case 10v10 100015 10
```

> 注意：上例用 `N_AGENTS-1` 做 “NX 数” 的示例，与你实际希望的「5 台 NX / n-5 本地」时仅将 `run_case` 最后一个参数改为 11（含 5NX + 6 本地）等。  

## 7) 停止所有 NX 侧策略节点（拔电前）

```bash
NX_IPS=(192.168.47.101 192.168.47.103 192.168.47.104 192.168.47.105 192.168.47.106)
for ip in "${NX_IPS[@]}"; do
  sshpass -p 'nvidia' ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o PasswordAuthentication=yes \
    -o StrictHostKeyChecking=no nvidia@"$ip" \
    "pkill -f 'hil_policy_node.py|hil_env_server.py|run_rta_mappo_ablation_mc.py|hil_protocol' || true; \
     pgrep -af 'hil_policy_node.py|hil_env_server.py|run_rta_mappo_ablation_mc.py|hil_protocol' || true"
done
```

## 8) 结果绘图（plot_all）

`plot_all.py` 会扫描给定目录下的每个子目录，自动读取：
- `summary.json`
- `trajectory_data.npz`
- `game_data.npz`（可选）

```bash
OUT=/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/hil_outputs/original_success_5nx_restlocal

cp plot_all.py "$OUT/"
cd "$OUT"

/home/uav/anaconda3/envs/rlgpu/bin/python plot_all.py
```

生成图像文件（通常位于各 case 目录）：
- `fig1a_speed.pdf`
- `fig1b_pitch_overload.pdf`
- `fig1c_yaw_overload.pdf`
- `fig1d_distance.pdf`
- `fig2a_gantt.pdf`
- `fig2b_assignment_cost.pdf`
- `fig2c_ratio.pdf`
- `fig2d_pitch_overload.pdf`
- `fig2e_yaw_overload.pdf`
- `fig3a_role_prob.pdf`
- `fig3b_lock_pressure.pdf`
- `fig3c_phi_neff.pdf`
- `fig3d_pen_prob.pdf`
- `fig3e_gamma_xi.pdf`
- `fig3f_hit_escape.pdf`
- `fig4_traj3d.pdf`

## 9) 结果路径（本次会话标准）

完整输出目录：
`hil_outputs/original_success_5nx_restlocal/`

子目录（每工况）：
- `6v6_seed60015/`
- `8v8_seed80047/`
- `10v10_seed100015/`

每个子目录应包含：
- `summary.json`
- `trajectory_data.npz`
- `game_data.npz`
- `server.log`
- `local_agent*.log`
- `nx_agent*.log`
- 各类 `fig*.pdf`

## 10) 关键故障排查（本次修复点）

1. `hil_policy_node.py` 的 policy 载入路径要优先指向 MACPO 的 `rMAPPOPolicy`，避免误走 on-policy 分支导致 HIL 策略行为漂移。
2. NX 与本地时间不同步/时钟落后会导致日志里 tar 时间异常，但不影响主要功能。
3. 端口与服务端/客户端 IP 不匹配是连接失败最常见原因：先 `ping` 再 SSH，再检查 `hil_env_server.py` 是否在本机上监听 `HOST:PORT`。
