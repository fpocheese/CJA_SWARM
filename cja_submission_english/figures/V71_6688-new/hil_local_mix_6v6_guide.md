# 6v6 HIL（1 架 NX，5 架数学仿真）启动指南

你当前这边是“NX 接在本地电脑上，不是远端服务器”，我这边当前环境无法直接访问你本地局域网。
下面给出可直接在你本地 PC 上执行的流程。  
该流程对应现有 `hil_v71_split` 架构：服务端环境跑在本地 PC，6 个 policy node 中 1 个在 NX 上，其余 5 个在本地仿真。

## 1）本地准备

1. 在本地解压你从远端拿到的部署包（若已在本地可跳过）：

   ```bash
   mkdir -p ~/hil_mix_6v6
   tar -xzf /path/to/v71_hil_6v6_bundle.tgz -C ~/hil_mix_6v6
   cd ~/hil_mix_6v6
   ```

2. 在本地创建并激活 Python 环境（与你训练代码一致，至少有 `torch/gym/numpy`）。

3. 从远端拷贝模型与脚本时请保持下面文件完整：  
   `hil_v71_split/*`, `envs/fov_penetration/*`, `third_party/MACPO/MACPO/macpo/*`,
   `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models/actor_agent{0..3}.pt`,
   `scripts/phase_obs_wrapper.py`, `scripts/terminal_pn_action_wrapper.py`。

## 2）先在本地起服务端

```bash
cd ~/hil_mix_6v6
python hil_v71_split/hil_env_server.py \
  --case 6v6 \
  --host 0.0.0.0 \
  --port 5500 \
  --episodes 200 \
  --max-steps 8000 \
  --out /tmp/v71_hil_mix6v6_hw_summary.json
```

- 该进程会阻塞等待 6 个策略节点连接；建议先开本地客户端，再开 NX 客户端。

## 3）本地起 5 个 policy node（非硬件）

另外开 5 个终端或用背景脚本：

```bash
cd ~/hil_mix_6v6
for i in 1 2 3 4 5; do
  python hil_v71_split/hil_policy_node.py \
    --agent-id "$i" \
    --source-agent $((i % 4)) \
    --server-host <LOCAL_IP> \
    --server-port 5500 \
    --model-dir outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models > /tmp/hil_node_${i}.log 2>&1 &
done
```

- `source-agent` 按你的模型是 4 份权重循环复用：0,1,2,3,0。
- `<LOCAL_IP>` 改成本地 PC 的内网 IPv4 地址，例如 `192.168.1.x`。

## 4）NX 上起 1 个 policy node（硬件侧）

```bash
cd /home/a2rl/000000GSY_mutiUAV/swarm_attack_v2   # NX 端路径
python hil_v71_split/hil_policy_node.py \
  --agent-id 0 \
  --source-agent 0 \
  --server-host <LOCAL_IP> \
  --server-port 5500 \
  --model-dir outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models \
  > /tmp/hil_node_nx0.log 2>&1
```

- 如果这台 NX 是你要“真实运行”的飞控侧：  
  - 让它的 `hil_policy_node.py` 通过本地输入总线/网络输出 `action`；
  - 端口与 `<LOCAL_IP>:5500` 能 reach 到本地服务端即可。

## 5）查看状态与成功率

服务端运行结束后解析结果：

```bash
python - <<'PY'
import json
res = json.load(open('/tmp/v71_hil_mix6v6_hw_summary.json'))
print('case=',res['case'], 'episodes=',res['episodes'])
print('success=',res['success_count'],'/',res['episodes'], 'rate=',res['success_rate'])
PY
```

如需导出每回合明细：

```bash
python - <<'PY'
import json, statistics
res = json.load(open('/tmp/v71_hil_mix6v6_hw_summary.json'))
for item in res['summaries']:
    print(item['episode'], item['seed'], item['success'], item['done_reason'], item['best_min_dist_m'], item['best_agent'])
PY
```

## 6）6v6 + 200 次“局部扰动 MC”替代脚本（不走 full-mc）

若你想使用与当前数学仿真一致的“成功种子周围局部扰动”思路（不是严格全 MC），用本地 Python 直接跑：

```bash
python run_rta_mappo_ablation_mc.py \
  --episodes 200 \
  --variants full no_threat_margin no_team_pen_hit \
  --mc-mode local_perturb \
  --cases 6v6 \
  --out-root /tmp/rta_v71_6v6_local_perturb_200 \
  --workers 6
```

生成的结果结构含 `success_count` 与 `success_rate`，可直接用于论文中的 6v6 两类实验汇总。

## 7）注意事项

- 该 split-HIL 架构默认 **不区分** “4v4/6v6/8v8”是否有硬件差异；你要求的“6v6 一档”在服务端直接用 `--case 6v6` 即可。  
- 本对话后续不再在实验配置中使用 4v4 结果。  
- 若 NX 侧进程未连接：重点看 NX 日志和服务端 `[server] agent x connected`；先确认端口、用户名、防火墙与 `PYTHONPATH`。

  
