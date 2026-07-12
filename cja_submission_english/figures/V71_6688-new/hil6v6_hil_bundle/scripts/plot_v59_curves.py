#!/usr/bin/env python
"""Parse v59 training log -> matplotlib curves.

Usage: python scripts/plot_v59_curves.py <log_path> <out_png>
"""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "outputs/v59_analysis/v59_minimal_strike.out"
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/v59_analysis/v59_curves.png"
text = open(log).read()

ups = [int(x) for x in re.findall(r"updates (\d+)/", text)]
step_r = [float(x) for x in re.findall(r"average_step_rewards is (-?\d+\.\d+)", text)]
ep_r = [float(x) for x in re.findall(r"some episodes done, average rewards:\s+(-?\d+\.\d+)", text)]
eval_r = [float(x) for x in re.findall(r"eval_average_episode_rewards is (-?\d+\.\d+)", text)]

n = min(len(ups), len(step_r), len(ep_r))
ups = ups[:n]; step_r = step_r[:n]; ep_r = ep_r[:n]
eval_x = [ups[i] for i in range(n) if i % 5 == 4][:len(eval_r)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ups, step_r, "-o", ms=3, color="tab:blue"); axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title("avg_step_reward"); axes[0].set_xlabel("update"); axes[0].grid(alpha=0.3)
axes[1].plot(ups, ep_r, "-o", ms=3, color="tab:orange")
axes[1].set_title("avg_episode_reward (rollout)"); axes[1].set_xlabel("update"); axes[1].grid(alpha=0.3)
axes[2].plot(eval_x, eval_r, "-o", ms=4, color="tab:green")
axes[2].set_title("eval_avg_episode_reward"); axes[2].set_xlabel("update"); axes[2].grid(alpha=0.3)
fig.suptitle(f"V59 minimal_strike (N={n} updates)")
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110)
print(f"saved {out}")
print(f"updates={n}  step_r last5={step_r[-5:]}  ep_r last5={[round(v,1) for v in ep_r[-5:]]}  eval_r={eval_r}")
