# Swarm Attack Training Result Plotting Method

This folder stores the new `swarm_attack` experiment data, figures, exported plot CSV files, and the plotting method note.

## Located Source Code

- The original three-panel figure style is in `onpolicy/scripts/plot_results_ieee.py` and `onpolicy/scripts/plot_results_ieee_v2.py`.
- The existing separated-figure version is in `onpolicy/scripts/plot_results_ieee_v3.py`, which writes `fig_a_reward`, `fig_b_critic_loss`, and `fig_c_entropy`. That script also contains Advanced-MAPPO-specific curve blending; it is useful for locating the old plotting path, but is not used for the new reproducible figures.
- The new reproducible plotting script for this run is `onpolicy/scripts/plot_results_swarm_attack.py`, which keeps the three figures separate and applies the same statistics to every algorithm.
- The training script is `onpolicy/scripts/train_simple_converge_v7.py`; it now accepts `--save_dir` so this experiment can be saved separately.

## Training Command

```bash
conda run -n rlgpu python onpolicy/scripts/train_simple_converge_v7.py \
  --save_dir onpolicy/scripts/results/swarm_attack \
  --num_episodes 1500 \
  --seeds 11 12 13 14 15 \
  --algorithms Advanced-MAPPO MAPPO IPPO IA2C IQL
```

## Plotting Command

```bash
conda run -n rlgpu python onpolicy/scripts/plot_results_swarm_attack.py \
  --data_dir onpolicy/scripts/results/swarm_attack \
  --output_dir onpolicy/scripts/results/swarm_attack
```

## Figure Outputs

- `swarm_attack_reward.png` and `swarm_attack_reward.pdf`
- `swarm_attack_critic_loss.png` and `swarm_attack_critic_loss.pdf`
- `swarm_attack_entropy.png` and `swarm_attack_entropy.pdf`
- `plot_data/`: CSV files containing the exact plotted mean and shaded bounds for each algorithm.

## Plotting Method

Each seed curve is smoothed with a moving average window of 50 episodes. The plotted line is the cross-seed mean, and the shaded band is the standard error of the mean. Reward is normalized by 65 for compact display; critic loss uses a logarithmic y-axis; entropy is plotted without normalization. All algorithms are processed with the same pipeline.