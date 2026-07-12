# Swarm Attack Training Result Plotting Method

## Data Source

- Raw result directory: `/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main/onpolicy/scripts/results/swarm_attack/_smoke`
- Training script: `onpolicy/scripts/train_simple_converge_v7.py`
- Plotting script: `onpolicy/scripts/plot_results_swarm_attack.py`
- Compared algorithms: Advanced-MAPPO, MAPPO, IPPO, IA2C, IQL

## Statistical Processing

- Each seed curve is smoothed by a moving average window of `2` episodes.
- Curves show the cross-seed mean. Shaded regions show standard error of the mean.
- Reward is normalized by `65.0` for compact plotting. Loss and entropy are not normalized.
- All algorithms use the same processing pipeline; no algorithm-specific curve reshaping is used.

## Output Figures

- `swarm_attack_reward.png` / `swarm_attack_reward.pdf`: normalized team reward.
- `swarm_attack_critic_loss.png` / `swarm_attack_critic_loss.pdf`: critic loss in log scale.
- `swarm_attack_entropy.png` / `swarm_attack_entropy.pdf`: policy entropy.
- `plot_data/`: CSV exports for every plotted curve.

## Reproduction Commands

```bash
python onpolicy/scripts/train_simple_converge_v7.py \
  --save_dir onpolicy/scripts/results/swarm_attack \
  --num_episodes 6000 \
  --seeds 11 12 13 14 15 \
  --algorithms Advanced-MAPPO MAPPO IPPO IA2C IQL

python onpolicy/scripts/plot_results_swarm_attack.py \
  --data_dir onpolicy/scripts/results/swarm_attack \
  --output_dir onpolicy/scripts/results/swarm_attack
```

## Current Summary

| Metric | Algorithm | Final mean | Seeds |
| --- | --- | ---: | ---: |
| Reward | ART-MAPPO (Ours) | 1.897316 | 1 |
| Critic loss | ART-MAPPO (Ours) | 0.729990 | 1 |
| Entropy | ART-MAPPO (Ours) | 1.837666 | 1 |
