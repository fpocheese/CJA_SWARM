# Swarm Attack 模型训练结果图绘制方法

## 数据来源

- 原始结果目录: `/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main/onpolicy/scripts/results/swarm_attack/round2_2000`
- 训练脚本: `onpolicy/scripts/train_simple_converge_v7.py`
- 绘图脚本: `onpolicy/scripts/plot_results_swarm_attack.py`
- 对比算法: Advanced-MAPPO, MAPPO, IPPO, IA2C, IQL

## 本轮训练设置

- 总训练回合数: `2000` episode。
- 随机种子: `11 12 13 14 15`。
- MAPPO 使用集中式 critic 并降低后期学习率，使曲线在 2000 episode 内收敛。
- Advanced-MAPPO/Ours 使用 hold-then-decay 学习率和熵系数调度，前期保持学习强度，约 1000 episode 后进入低学习率平台期。
- IPPO、IA2C、IQL 沿用原配置，仅跑满 2000 episode。

## 原图代码定位

- 原三联图绘图脚本: `onpolicy/scripts/plot_results_ieee.py` 和 `onpolicy/scripts/plot_results_ieee_v2.py`。
- 已有三图分离脚本: `onpolicy/scripts/plot_results_ieee_v3.py`，会输出 `fig_a_reward`, `fig_b_critic_loss`, `fig_c_entropy`。
- 本次实验使用 `onpolicy/scripts/plot_results_swarm_attack.py`，直接输出三张独立论文图，并对所有算法采用同一套统计处理。

## 统计与绘图方法

- 每条 seed 曲线先使用 `50` episode 移动平均进行平滑。
- 实线表示跨 seed 均值，阴影表示均值标准误差。
- Reward 为了紧凑展示除以 `65.0`；critic loss 使用 log 坐标；entropy 不做归一化。
- 所有算法使用完全一致的平滑、均值和阴影计算流程，不对单个算法做额外曲线重塑或目标曲线拼接。

## 输出文件

- `swarm_attack_reward.png` / `swarm_attack_reward.pdf`: 归一化团队奖励。
- `swarm_attack_critic_loss.png` / `swarm_attack_critic_loss.pdf`: critic loss, log 坐标。
- `swarm_attack_entropy.png` / `swarm_attack_entropy.pdf`: policy entropy。
- `plot_data/`: 每条曲线的均值和阴影上下界 CSV。

## 收敛性复核

| Algorithm | Last 200 reward | 1000-1200 | 1800-2000 | Delta | CV last500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ART-MAPPO (Ours) | 493.84 | 480.96 | 493.84 | 0.0268 | 0.0467 |
| MAPPO | 419.53 | 409.12 | 419.53 | 0.0254 | 0.0501 |
| IPPO | 350.35 | 288.04 | 350.35 | 0.2163 | 0.1024 |
| IA2C | 393.72 | 261.85 | 393.72 | 0.5036 | 0.0806 |
| IQL | 364.26 | 283.98 | 364.26 | 0.2827 | 0.0880 |

## 复现实验命令

```bash
conda run -n rlgpu python onpolicy/scripts/train_simple_converge_v7.py \
  --save_dir onpolicy/scripts/results/swarm_attack/round2_2000 \
  --num_episodes 2000 \
  --seeds 11 12 13 14 15 \
  --algorithms Advanced-MAPPO MAPPO IPPO IA2C IQL

conda run -n rlgpu python onpolicy/scripts/plot_results_swarm_attack.py \
  --data_dir onpolicy/scripts/results/swarm_attack/round2_2000 \
  --output_dir onpolicy/scripts/results/swarm_attack/round2_2000 \
  --train_episodes 2000 \
  --train_seeds 11 12 13 14 15
```

## 当前结果摘要

| Metric | Algorithm | Final mean | Seeds |
| --- | --- | ---: | ---: |
| Reward | ART-MAPPO (Ours) | 7.595069 | 5 |
| Reward | MAPPO | 6.435353 | 5 |
| Reward | IPPO | 5.378157 | 5 |
| Reward | IA2C | 6.028438 | 5 |
| Reward | IQL | 5.623275 | 5 |
| Critic loss | ART-MAPPO (Ours) | 0.017150 | 5 |
| Critic loss | MAPPO | 0.025765 | 5 |
| Critic loss | IPPO | 0.030148 | 5 |
| Critic loss | IA2C | 0.023057 | 5 |
| Critic loss | IQL | 0.085585 | 5 |
| Entropy | ART-MAPPO (Ours) | 1.811894 | 5 |
| Entropy | MAPPO | 1.812715 | 5 |
| Entropy | IPPO | 1.780130 | 5 |
| Entropy | IA2C | 1.894849 | 5 |
| Entropy | IQL | 1.850413 | 5 |
