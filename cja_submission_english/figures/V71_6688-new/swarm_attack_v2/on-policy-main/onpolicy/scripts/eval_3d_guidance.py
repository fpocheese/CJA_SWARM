#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path = [str(PROJECT_ROOT)] + [p for p in sys.path if str(Path(p).resolve()) != str(PROJECT_ROOT)]
stale_roots = ["/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main", "/home/uav/00gao_xueshu/togsy_2025"]
for stale in stale_roots:
    sys.path = [p for p in sys.path if not str(Path(p)).startswith(stale)]
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def select_model_dir(model_root: Path) -> Path:
    if (model_root / "actor.pt").exists() and (model_root / "critic.pt").exists():
        return model_root
    candidates = [p for p in model_root.glob("**/actor.pt")]
    if not candidates:
        raise FileNotFoundError(f"未在 {model_root} 找到 actor.pt")
    # 优先选最近修改的子目录里的模型
    best = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return best.parent


def build_args(case_name: str, raw_args):
    parser = get_config()

    parser.add_argument("--scenario_name", type=str, default="simple_world_comm_3d")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--algo", type=str, default="MAPPO", choices=["MAPPO", "Advanced-MAPPO", "IPPO", "IA2C", "IQL"])
    parser.add_argument("--case_3d", type=str, default="case1", choices=["case1", "case2"])
    parser.add_argument("--hit_radius_3d", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--sync_tol", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default="onpolicy/scripts/results/3d_eval")
    parser.add_argument("--model_dir_case1", type=str, default=None)
    parser.add_argument("--model_dir_case2", type=str, default=None)
    parser.add_argument("--require_all_hit", action="store_true", help="要求所有目标均被拦截才算成功")

    parser.set_defaults(
        model_dir=None,
        use_render=False,
        use_wandb=False,
        n_rollout_threads=1,
        episode_length=1500,
        hidden_size=256,
        layer_N=1,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        algorithm_name="rmappo",
        use_valuenorm=True,
        use_popart=False,
        ppo_epoch=10,
        clip_param=0.1,
        lr=5e-4,
        critic_lr=5e-4,
        entropy_coef=0.01,
        max_grad_norm=0.3,
        value_loss_coef=0.3,
        huber_delta=15.0,
        gamma=0.99,
        gae_lambda=0.95,
        log_interval=5,
        save_interval=100,
        use_linear_lr_decay=True,
        use_centralized_V=True,
        eval=False,
        eval_interval=5,
        eval_episodes=5,
        save_dir=None,
    )

    argv = [
        "--scenario_name", "simple_world_comm_3d",
        "--case_3d", case_name,
    ]
    for k, v in sorted(vars(raw_args).items()):
        if k in {"scenario_name", "case_3d", "model_dir", "model_dir_case1", "model_dir_case2", "outdir", "case1", "case2", "eval_episodes", "max_steps"}:
            continue
        if k in {"stochastic_eval", "eval_different_seed", "require_success_plot", "require_all_hit"}:
            continue
        if v is None:
            continue
        argv.extend([f"--{k}", str(v)])
    argv.extend(["--case_3d", case_name])
    argv.extend(["--seed", str(raw_args.seed)])
    if raw_args.eval_episodes is not None:
        argv.extend(["--eval_episodes", str(raw_args.eval_episodes)])
    if raw_args.max_steps is not None:
        argv.extend(["--max_steps", str(raw_args.max_steps)])
    argv.extend(["--sync_tol", str(raw_args.sync_tol)])

    args = parser.parse_args(argv)
    args.case_3d = case_name
    args.num_env_steps = args.episode_length
    args.stochastic_eval = raw_args.stochastic_eval
    args.eval_different_seed = raw_args.eval_different_seed
    args.require_success_plot = raw_args.require_success_plot
    args.require_all_hit = raw_args.require_all_hit
    return args


def collect_model(args, model_root: Path):
    model_dir = select_model_dir(model_root)
    actor_path = model_dir / "actor.pt"
    critic_path = model_dir / "critic.pt"
    if not actor_path.exists() or not critic_path.exists():
        raise FileNotFoundError(f"模型路径不完整: {model_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MPEEnv(args)
    env.seed(args.seed)
    env.world.seed = args.seed if hasattr(env.world, "seed") else None

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")

    policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )
    policy.actor.load_state_dict(torch.load(str(actor_path), map_location=device))
    policy.critic.load_state_dict(torch.load(str(critic_path), map_location=device))
    policy.actor.eval()
    policy.critic.eval()

    return env, policy, device, model_dir


def evaluate_case(args, env, policy, device):
    n_agents = len(env.world.policy_agents)
    target_num = None

    all_case_results = []
    all_success = []
    best_hit_count = -1
    best_min_dist = float("inf")
    rep_att = None
    rep_def = None
    rep_ctrl = None
    rep_tgo = None
    rep_hit_count = 0
    rep_min_dist = float("inf")
    

    for ep in range(args.eval_episodes):
        if args.eval_different_seed:
            # 每个回合使用不同随机种子，扩展初始态势采样覆盖范围
            eval_seed = args.seed + ep
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)

        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        rnn_states = np.zeros((n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)

        traj_att = []
        traj_def = []
        ctrl_hist = []
        tgo_hist = []
        hit_time = np.full(n_agents, np.nan, dtype=np.float32)
        dist_hist = []

        adv_agents = [ag for ag in env.world.agents if ag.adversary]
        tgt_ids = sorted({int(ag.target) for ag in adv_agents})
        tgt_size = len(tgt_ids)
        target_num = tgt_size
        tgt_index = {tid: k for k, tid in enumerate(tgt_ids)}

        for t in range(args.max_steps):
            with torch.no_grad():
                actions, rnn_states = policy.act(
                    obs,
                    rnn_states,
                    masks,
                    deterministic=not args.stochastic_eval,
                )
            actions_np = actions.detach().cpu().numpy()
            obs_next, rewards, dones, infos = env.step(actions_np, t)
            obs = np.asarray(obs_next, dtype=np.float32)

            def_pos = []
            att_pos = []
            tgo_this = []
            ctrl_this = []
            for ag in env.world.agents:
                if ag.adversary:
                    def_pos.append(ag.state.p_pos.copy())
                    ctrl_this.append(ag.state.load.copy())
                    tgo_this.append(ag.state.time_tgo[0])
                    target = env.world.agents[ag.target]
                    dist_hist.append(np.linalg.norm(target.state.p_pos - ag.state.p_pos))
                else:
                    att_pos.append(ag.state.p_pos.copy())

            traj_def.append(np.array(def_pos))
            traj_att.append(np.array(att_pos))
            ctrl_hist.append(np.array(ctrl_this))
            tgo_hist.append(np.array(tgo_this))

            dones_arr = np.asarray(dones, dtype=bool)
            for i, ag in enumerate(adv_agents):
                if getattr(ag.state, "actual_hit", False) and np.isnan(hit_time[i]):
                    hit_time[i] = (t + 1) * env.dt
            if dones_arr.all():
                break

            masks = np.ones_like(masks, dtype=np.float32)
            masks[dones_arr, 0] = 0.0

        # 按照目标统计是否成功
        hit_targets = np.zeros(tgt_size, dtype=bool)
        sync_targets = np.zeros(tgt_size, dtype=bool)
        per_target_time = np.full(tgt_size, np.nan, dtype=np.float32)
        for tid in tgt_ids:
            idx = tgt_index[int(tid)]
            group_times = [hit_time[i] for i, ag in enumerate(adv_agents) if int(ag.target) == int(tid) and not np.isnan(hit_time[i])]
            group_size = sum(1 for ag in adv_agents if int(ag.target) == int(tid))
            if group_times:
                hit_targets[idx] = True
                per_target_time[idx] = float(np.min(group_times))
            if len(group_times) == group_size and (max(group_times) - min(group_times)) <= args.sync_tol:
                sync_targets[idx] = True

        hit_count = int(np.nansum(~np.isnan(hit_time)))
        hit_target_count = int(np.count_nonzero(hit_targets))
        sync_target_count = int(np.count_nonzero(sync_targets))
        all_hit = hit_target_count >= tgt_size
        all_sync = sync_target_count >= tgt_size
        success_rate = hit_count / max(1, len(adv_agents))
        all_hit_rate = 1.0 if all_hit else 0.0
        all_sync_rate = 1.0 if all_sync else 0.0
        if np.isfinite(per_target_time).any():
            mean_target_time = float(np.nanmean(per_target_time))
        else:
            mean_target_time = float("nan")

        ep_traj_def = np.array(traj_def)
        ep_traj_att = np.array(traj_att)
        ep_ctrl = np.array(ctrl_hist)
        ep_tgo = np.array(tgo_hist)
        ep_min_dist = float(np.nanmin(np.array(dist_hist))) if len(dist_hist) > 0 else float("inf")

        if (not args.require_all_hit and (hit_count > best_hit_count or (hit_count == best_hit_count and ep_min_dist < best_min_dist))) or (
            args.require_all_hit and all_hit and ep_min_dist < best_min_dist
        ):
            best_hit_count = hit_count
            best_min_dist = ep_min_dist
            rep_att = ep_traj_att
            rep_def = ep_traj_def
            rep_ctrl = ep_ctrl
            rep_tgo = ep_tgo
            rep_hit_count = hit_count
            rep_min_dist = ep_min_dist

        all_case_results.append({
            "success_rate": success_rate,
            "hit_targets": hit_target_count,
            "sync_targets": sync_target_count,
            "all_hit": bool(all_hit),
            "all_sync": bool(all_sync),
            "mean_target_time": mean_target_time,
            "target_mask": hit_targets,
            "sync_mask": sync_targets,
            "hit_time": hit_time,
        })
        all_success.append(all_hit_rate if args.require_all_hit else success_rate)
        if (ep + 1) % 50 == 0:
            print(
                f"[INFO] case={args.case_3d} ep={ep + 1}/{args.eval_episodes} "
                f"best_hit={best_hit_count} best_min_dist={best_min_dist:.2f} "
                f"all_hit_mode={args.require_all_hit}"
            )

    return {
        "case_summary": {
            "episodes": args.eval_episodes,
        "success_rate": float(np.mean(all_success)),
            "target_success_rate": float(np.mean([r["hit_targets"] for r in all_case_results]) / max(1, target_num if target_num is not None else 1)),
            "sync_success_rate": float(np.mean([r["sync_targets"] for r in all_case_results]) / max(1, target_num if target_num is not None else 1)),
            "all_sync_rate": float(np.mean([1.0 if r["all_sync"] else 0.0 for r in all_case_results])) if all_case_results else 0.0,
        "mean_hit_time": float(np.nanmean([r["mean_target_time"] for r in all_case_results])) if all_case_results else float("nan"),
        },
        "rep_att": np.array(rep_att) if rep_att is not None else np.empty((0,)),
        "rep_def": np.array(rep_def) if rep_def is not None else np.empty((0,)),
        "rep_ctrl": np.array(rep_ctrl) if rep_ctrl is not None else np.empty((0,)),
        "rep_tgo": np.array(rep_tgo) if rep_tgo is not None else np.empty((0,)),
        "selected_hit_count": int(rep_hit_count),
        "selected_min_dist": float(rep_min_dist),
        "all_results": all_case_results,
    }


def plot_case(case_name: str, data: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    att = data["rep_att"]
    deff = data["rep_def"]
    ctrl = data["rep_ctrl"]
    tgo = data["rep_tgo"]

    if data["rep_att"].size == 0 or data["rep_def"].size == 0:
        print(f"[WARN] case={case_name} 未找到符合条件的拦截样本，跳过作图")
        return

    t_steps = np.arange(att.shape[0]) * 0.05

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for j in range(att.shape[1]):
        ax.plot(
            att[:, j, 0],
            att[:, j, 1],
            att[:, j, 2],
            linewidth=1.0,
            color="crimson",
            alpha=0.55,
            linestyle="--",
            label="attacker" if j == 0 else None,
        )
    for i in range(deff.shape[1]):
        ax.plot(
            deff[:, i, 0],
            deff[:, i, 1],
            deff[:, i, 2],
            linewidth=0.8,
            alpha=0.75,
            label="defender" if i == 0 else None,
        )
    ax.scatter([0.0], [0.0], [0.0], marker="*", s=80, color="black", label="target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    status_text = "success" if data.get("selected_hit_count", 0) > 0 else "no success"
    ax.set_title(f"Case {case_name} 3D Trajectory ({status_text})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"{case_name}_trajectory.{ext}", dpi=200)
    plt.close(fig)

    fig_top, ax_top = plt.subplots(figsize=(8, 6))
    for j in range(att.shape[1]):
        ax_top.plot(
            att[:, j, 0],
            att[:, j, 1],
            linestyle="--",
            color="crimson",
            alpha=0.55,
            label="attacker" if j == 0 else None,
        )
    for i in range(deff.shape[1]):
        ax_top.plot(
            deff[:, i, 0],
            deff[:, i, 1],
            linewidth=0.8,
            alpha=0.75,
            label="defender" if i == 0 else None,
        )
    ax_top.scatter([0.0], [0.0], marker="*", s=80, color="black", label="target")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_title(f"Case {case_name} Top-down Trajectory")
    ax_top.legend(loc="upper right", fontsize=8)
    fig_top.tight_layout()
    for ext in ["png", "pdf"]:
        fig_top.savefig(outdir / f"{case_name}_trajectory_xy.{ext}", dpi=200)
    plt.close(fig_top)

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax2[0].plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 0], alpha=0.2, color="tab:blue")
    ax2[0].set_ylabel("Yaw overload (g)")
    ax2[0].set_title(f"Case {case_name} Defender Commands")

    ax2[1].plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 1], alpha=0.2, color="tab:orange")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Pitch overload (g)")

    fig2.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(outdir / f"{case_name}_control.{ext}", dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    min_tgo = np.nanmin(tgo, axis=1) if tgo.size else np.array([])
    if min_tgo.ndim == 0:
        min_tgo = np.array([float(min_tgo)])
    ax3.plot(t_steps[:len(min_tgo)], min_tgo, color="tab:green")
    ax3.set_title(f"Case {case_name} Min TGO")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("time-to-go (s)")
    fig3.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(outdir / f"{case_name}_tgo.{ext}", dpi=200)
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--sync_tol", type=float, default=0.5, help="同目标组最大到达时间差阈值，单位 s")
    parser.add_argument("--stochastic_eval", action="store_true", help="使用随机采样动作进行评估")
    parser.add_argument("--eval_different_seed", action="store_true", help="每个回合用不同随机种子重置初始状态")
    parser.add_argument("--require_success_plot", action="store_true", help="优先选取有拦截成功样本进行作图")
    parser.add_argument("--require_all_hit", action="store_true", help="要求所有目标均被拦截才能视为成功")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dir_case1", type=str, default=None)
    parser.add_argument("--model_dir_case2", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="onpolicy/scripts/results/3d_eval")
    parser.add_argument("--case1", action="store_true")
    parser.add_argument("--case2", action="store_true")
    args = parser.parse_args()

    if not args.case1 and not args.case2:
        cases = ["case1", "case2"]
    else:
        cases = []
        if args.case1:
            cases.append("case1")
        if args.case2:
            cases.append("case2")

    case_summary_rows = []
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for case_name in cases:
        cfg = build_args(case_name, args)
        case_model_root = {
            "case1": args.model_dir_case1,
            "case2": args.model_dir_case2,
        }.get(case_name)
        model_root = Path(case_model_root) if case_model_root else Path(args.model_dir)
        env, policy, device, model_dir = collect_model(cfg, model_root)
        result = evaluate_case(cfg, env, policy, device)

        case_outdir = outdir / case_name
        case_outdir.mkdir(parents=True, exist_ok=True)
        plot_case(case_name, result, case_outdir)
        np.savez(
            case_outdir / f"{case_name}_selected_episode.npz",
            rep_att=result["rep_att"],
            rep_def=result["rep_def"],
            rep_ctrl=result["rep_ctrl"],
            rep_tgo=result["rep_tgo"],
            selected_hit_count=result["selected_hit_count"],
            selected_min_dist=result["selected_min_dist"],
            case=case_name,
        )

        row = {
            "case": case_name,
            "model": str(model_dir),
            "episodes": result["case_summary"]["episodes"],
            "attack_success_rate": result["case_summary"]["success_rate"],
            "target_success_rate": result["case_summary"]["target_success_rate"],
            "sync_success_rate": result["case_summary"]["sync_success_rate"],
            "all_sync_rate": result["case_summary"]["all_sync_rate"],
            "mean_target_time": result["case_summary"]["mean_hit_time"],
        }
        case_summary_rows.append(row)

    import csv
    with open(outdir / "eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "model", "episodes", "attack_success_rate", "target_success_rate", "sync_success_rate", "all_sync_rate", "mean_target_time"])
        writer.writeheader()
        for row in case_summary_rows:
            writer.writerow(row)

    print("[INFO] 评测完成")
    for r in case_summary_rows:
        print(f"{r['case']}: hit_rate={r['attack_success_rate']:.3f}, target_success={r['target_success_rate']:.3f}, sync_success={r['sync_success_rate']:.3f}, all_sync={r['all_sync_rate']:.3f}, mean_time={r['mean_target_time']}")


if __name__ == "__main__":
    main()
