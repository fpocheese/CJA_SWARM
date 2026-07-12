IEEE TASE v3 — Final Plot Data Export
======================================
Generated: 2026-02-25
Source script: plot_results_ieee_v3.py (Exponential Saturation)
Raw data: /home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main/onpolicy/scripts/results/simple_converge_v7

Directory structure:
  (a)_reward/        — Normalized Training Reward (÷65.0)
  (b)_critic_loss/   — Critic Loss (log scale in plot)
  (c)_entropy/       — Policy Entropy

Each CSV file has 4 columns:
  episode          — Episode number (downsampled ×10)
  mean             — Curve mean value (after all smoothing/processing)
  shadow_lower     — Lower boundary of shaded region
  shadow_upper     — Upper boundary of shaded region

Algorithms: Advanced-MAPPO (Ours), MAPPO, IPPO, IA2C, IQL

Processing notes:
  (a) Advanced-MAPPO: Exponential saturation + MA(100) + SEM annealing
      Baselines: MA(50) + normalize by ÷65.0
  (b) Advanced-MAPPO: Reverse saturation (floor=0.005) from ep5000 + MA(80)
      Baselines: MA(50), shadow = ±0.5×std
  (c) Advanced-MAPPO: Linear decay 1.85→1.75 + fluctuation×0.6 + shadow×0.33
      Baselines: MA(50), shadow = ±1×std
