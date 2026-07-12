#!/usr/bin/env python
"""Generate extra diagnostics for the latest eval run.

Outputs:
- overload_offense.png: Offense pitch/yaw g over time
- overload_defense.png: Defense pitch/yaw g over time
- interceptor_assignment.png: Defender target index over time
- trajectory_time.png: Time-colored XY trajectories
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from envs.fov_penetration import FOVPenetrationEnv
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper
from eval_v28_10episodes import load_policies


G0 = 9.80665


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "outputs" / "v69_hourly_eval" / "latest_eval" / "summary.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "v69_hourly_eval" / "latest_eval_extra"),
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layer-n", type=int, default=3)
    return parser.parse_args()


def unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def resolve_target_idx(policy, offensives):
    if getattr(policy, "current_locked_target_idx", None) is not None:
        return int(policy.current_locked_target_idx)
    if getattr(policy, "lock_mode", None) == policy.STATE_INIT_GUIDE:
        if getattr(policy, "initial_assigned_target_idx", None) is not None:
            return int(policy.initial_assigned_target_idx)
    if getattr(policy, "current_locked_target_idx", None) is not None:
        return int(policy.current_locked_target_idx)
    if getattr(policy, "target", None) is not None:
        for j, off in enumerate(offensives):
            if off is policy.target:
                return int(j)
    return -1


def record_state(base_env, t, off_pitch, off_yaw, def_pitch, def_yaw, off_x, off_y, def_x, def_y, assignments):
    t.append(base_env.current_step * base_env.dt)
    for i, off in enumerate(base_env.offensives):
        if off.alive:
            off_pitch[i].append(float(off.an_pitch / G0))
            off_yaw[i].append(float(off.an_yaw / G0))
            off_x[i].append(float(off.x))
            off_y[i].append(float(off.y))
        else:
            off_pitch[i].append(np.nan)
            off_yaw[i].append(np.nan)
            off_x[i].append(np.nan)
            off_y[i].append(np.nan)
    for i, d in enumerate(base_env.defensives):
        if d.alive:
            def_pitch[i].append(float(d.an_pitch / G0))
            def_yaw[i].append(float(d.an_yaw / G0))
            def_x[i].append(float(d.x))
            def_y[i].append(float(d.y))
        else:
            def_pitch[i].append(np.nan)
            def_yaw[i].append(np.nan)
            def_x[i].append(np.nan)
            def_y[i].append(np.nan)
    for i, policy in enumerate(base_env.defensive_policies):
        assignments[i].append(resolve_target_idx(policy, base_env.offensives))


def plot_overload(time_s, pitch, yaw, title, out_path, labels):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for i, label in enumerate(labels):
        axes[0].plot(time_s, pitch[i], label=f"{label} pitch")
        axes[1].plot(time_s, yaw[i], label=f"{label} yaw")
    axes[0].set_ylabel("Pitch g")
    axes[1].set_ylabel("Yaw g")
    axes[1].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_assignments(time_s, assignments, out_path):
    n_def = len(assignments)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for i in range(n_def):
        ax = axes[i]
        ax.step(time_s, assignments[i], where="post")
        ax.set_title(f"Def{i} target")
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.set_yticklabels(["None", "Off0", "Off1", "Off2", "Off3"])
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.set_xlabel("Time (s)")
        if i % 2 == 0:
            ax.set_ylabel("Target idx")
    fig.suptitle("Interceptor Target Assignment (per defender)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory(time_s, off_x, off_y, def_x, def_y, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharex=True, sharey=True)
    ax_off, ax_def = axes

    valid_t = np.array(time_s)
    norm = plt.Normalize(vmin=float(np.nanmin(valid_t)), vmax=float(np.nanmax(valid_t)))
    cmap = plt.cm.viridis
    stride = max(1, int(len(valid_t) / 200))
    idxs = np.arange(0, len(valid_t), stride)

    for i in range(len(off_x)):
        ax_off.plot(off_x[i], off_y[i], alpha=0.6, linewidth=1.0)
        xs = np.array(off_x[i])[idxs]
        ys = np.array(off_y[i])[idxs]
        ts = valid_t[idxs]
        ax_off.scatter(xs, ys, c=ts, cmap=cmap, norm=norm, s=10, alpha=0.7)

    for i in range(len(def_x)):
        ax_def.plot(def_x[i], def_y[i], alpha=0.6, linewidth=1.0)
        xs = np.array(def_x[i])[idxs]
        ys = np.array(def_y[i])[idxs]
        ts = valid_t[idxs]
        ax_def.scatter(xs, ys, c=ts, cmap=cmap, norm=norm, s=10, alpha=0.7)

    for ax, title in [(ax_off, "Offense Trajectory (time-colored)"), (ax_def, "Defense Trajectory (time-colored)")]:
        ax.scatter([1200.0], [0.0], marker="*", s=90, color="black", zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    model_dir = Path(summary.get("model_dir", ""))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    obs_mask = summary.get("obs_mask", "")
    terminal_guidance = summary.get("terminal_guidance", "")
    pn_gain = float(summary.get("terminal_pn_gain", 3.0))
    pn_max_action = float(summary.get("terminal_pn_max_action", 0.8))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = FOVPenetrationEnv(scenario="scenario_1")
    if obs_mask and obs_mask != "none":
        env = PhaseMaskedFOVWrapper(env, mode=obs_mask)
    if terminal_guidance == "pn_los":
        env = TerminalPNActionWrapper(env, gain=pn_gain, max_action=pn_max_action)

    base_env = unwrap_env(env)

    device = torch.device("cpu")
    policies = load_policies(str(model_dir), env, device, hidden_size=args.hidden_size, layer_N=args.layer_n)

    if hasattr(env, "seed"):
        env.seed(args.seed)
    obs, share_obs, _ = env.reset()

    rnn_states = [torch.zeros(1, 1, args.hidden_size, device=device) for _ in range(env.n_agents)]
    masks = [torch.ones(1, 1, device=device) for _ in range(env.n_agents)]

    n_off = base_env.n_offensive
    n_def = base_env.n_defensive

    time_s = []
    off_pitch = [[] for _ in range(n_off)]
    off_yaw = [[] for _ in range(n_off)]
    def_pitch = [[] for _ in range(n_def)]
    def_yaw = [[] for _ in range(n_def)]
    off_x = [[] for _ in range(n_off)]
    off_y = [[] for _ in range(n_off)]
    def_x = [[] for _ in range(n_def)]
    def_y = [[] for _ in range(n_def)]
    assignments = [[] for _ in range(n_def)]

    record_state(base_env, time_s, off_pitch, off_yaw, def_pitch, def_yaw, off_x, off_y, def_x, def_y, assignments)

    for _ in range(args.max_steps):
        actions = []
        for i, policy in enumerate(policies):
            obs_t = torch.FloatTensor(np.asarray(obs[i]).flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, rnn_out = policy.actor(obs_t, rnn_states[i], masks[i], deterministic=True)
            actions.append(action.cpu().numpy().flatten())
            rnn_states[i] = rnn_out
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        record_state(base_env, time_s, off_pitch, off_yaw, def_pitch, def_yaw, off_x, off_y, def_x, def_y, assignments)
        if dones[0]:
            break

    plot_overload(time_s, off_pitch, off_yaw, "Offense Overload (Pitch/Yaw)", out_dir / "overload_offense.png", [f"Off{i}" for i in range(n_off)])
    plot_overload(time_s, def_pitch, def_yaw, "Defense Overload (Pitch/Yaw)", out_dir / "overload_defense.png", [f"Def{i}" for i in range(n_def)])
    plot_assignments(time_s, assignments, out_dir / "interceptor_assignment.png")
    plot_trajectory(time_s, off_x, off_y, def_x, def_y, out_dir / "trajectory_time.png")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
