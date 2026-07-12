#!/usr/bin/env python
"""V59 latest model: deterministic eval -> compact GIFs + per-step trajectory plot.

Renders 2 episodes (default seeds 1000,1001), max_steps cap configurable,
frame skip to keep gif <50MB.
Saves outputs to outputs/v59_analysis/.
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio
from envs.fov_penetration import FOVPenetrationEnv
from eval_v28_10episodes import load_policies

HIDDEN = 256
LAYER_N = 3


def render_frame(ax, env, step):
    ax.clear()
    hvt = env.hvt
    ax.scatter([hvt.x], [hvt.y], [hvt.z], c="red", s=140, marker="*", label="HVT")
    for off in env.offensives:
        c = "tab:blue" if off.alive else "lightgray"
        ax.scatter([off.x], [off.y], [off.z], c=c, s=40, marker="o")
        if off.alive:
            ax.plot([off.x, hvt.x], [off.y, hvt.y], [off.z, hvt.z], c="tab:blue", alpha=0.15, lw=0.5)
    for de in env.defensives:
        c = "tab:orange" if de.alive else "lightgray"
        ax.scatter([de.x], [de.y], [de.z], c=c, s=30, marker="^")
    ax.set_xlim(-2000, 2000); ax.set_ylim(-1500, 1500); ax.set_zlim(0, 600)
    ax.set_title(f"step={step} hits={env.hit_count} alive_off={sum(o.alive for o in env.offensives)}")


def run_one(env, policies, device, seed, max_steps, frame_skip, fig, ax):
    env.seed(seed)
    obs, _, _ = env.reset()
    rnn = [torch.zeros(1, 1, HIDDEN).to(device) for _ in range(env.n_agents)]
    masks = [torch.ones(1, 1).to(device) for _ in range(env.n_agents)]
    hvt = env.hvt
    n = env.n_agents
    init_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_d = list(init_d)
    traj = {i: [] for i in range(n)}  # (x,y,z) over time
    dist_over_t = {i: [] for i in range(n)}
    frames = []

    for step in range(max_steps):
        actions = []
        for i, po in enumerate(policies):
            o = np.asarray(obs[i] if not isinstance(obs, tuple) else obs[0][i]).flatten()
            ot = torch.FloatTensor(o).unsqueeze(0).to(device)
            with torch.no_grad():
                a, _, h = po.actor(ot, rnn[i], masks[i], deterministic=True)
            actions.append(a.cpu().numpy().flatten())
            rnn[i] = h
        obs, _, _, _, dones, _, _ = env.step(actions)
        for i, off in enumerate(env.offensives):
            if off.alive:
                d = off.distance_to(hvt.x, hvt.y, hvt.z)
                if d < min_d[i]:
                    min_d[i] = d
                traj[i].append((off.x, off.y, off.z))
                dist_over_t[i].append(d)
            else:
                dist_over_t[i].append(np.nan)
                traj[i].append((np.nan, np.nan, np.nan))
        if step % frame_skip == 0:
            render_frame(ax, env, step)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[..., :3].copy())
        if all(dones):
            break
    return {"init_d": init_d, "min_d": min_d, "traj": traj, "dist": dist_over_t,
            "hits": env.hit_count, "steps": step + 1, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/v59_analysis")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--frame_skip", type=int, default=20)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FOVPenetrationEnv(scenario="scenario_1")
    policies = load_policies(args.model_dir, env, device, hidden_size=HIDDEN, layer_N=LAYER_N)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    summaries = []
    fig_d, axes_d = plt.subplots(1, len(args.seeds), figsize=(5 * len(args.seeds), 4), squeeze=False)
    for k, seed in enumerate(args.seeds):
        res = run_one(env, policies, device, seed, args.max_steps, args.frame_skip, fig, ax)
        gif_path = os.path.join(args.out_dir, f"v59_seed{seed}.gif")
        imageio.mimsave(gif_path, res["frames"], fps=args.fps)
        ax_d = axes_d[0][k]
        for i in range(env.n_agents):
            ax_d.plot(res["dist"][i], lw=1, label=f"A{i}")
        ax_d.axhline(5.0, c="r", ls=":", lw=0.8, label="hit=5m")
        ax_d.set_title(f"seed={seed} steps={res['steps']} hits={res['hits']}")
        ax_d.set_xlabel("step"); ax_d.set_ylabel("dist to HVT (m)"); ax_d.grid(alpha=0.3)
        ax_d.legend(fontsize=7)
        closed = [res["init_d"][i] - res["min_d"][i] for i in range(env.n_agents)]
        line = (f"seed={seed} steps={res['steps']} hits={res['hits']} "
                f"min_d={[round(x,0) for x in res['min_d']]} "
                f"closed={[round(x,0) for x in closed]} gif={gif_path}")
        print(line); summaries.append(line)

    fig_d.tight_layout()
    fig_d.savefig(os.path.join(args.out_dir, "v59_distance_over_time.png"), dpi=110)
    with open(os.path.join(args.out_dir, "v59_eval_summary.txt"), "w") as f:
        f.write("\n".join(summaries))
    print("DONE")


if __name__ == "__main__":
    main()
