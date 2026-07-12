#!/usr/bin/env python
"""
analyze_ep02_allocation.py
--------------------------
对 seed=2002 (ep02, 唯一成功回合) 做逐步取证分析，量化并可视化
"拦截集群分配失效" 的全过程。

输出到 outputs/ep02_analysis/:
  allocation_timeline.png  — 4×拦截器锁定状态时间线
  lock_pressure.png        — 每架进攻方被锁定数量随时间变化
  trajectory_xy.png        — XY 平面轨迹 + 分配失效时刻标注
  dist_to_hvt.png          — 距 HVT 距离 + 末端制导切换点
  ep02_events.json         — 关键事件表

运行:
  conda run -n rlgpu python scripts/analyze_ep02_allocation.py
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MACPO", "MACPO"))

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.policies_interceptor import InterceptorPolicy
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper
from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

MODEL_DIR = "outputs/results/fov_penetration/mappo/v70_team_survive/run1/models"
OUT_DIR   = "outputs/ep02_analysis"
SEED      = 2002
HIDDEN    = 256
LAYER_N   = 3

# CLI 覆盖 (供批量测试使用)
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument('--seed', type=int, default=None)
_cli_parser.add_argument('--out-dir', type=str, default=None)
_cli_parser.add_argument('--model-dir', type=str, default=None)
_cli_args, _ = _cli_parser.parse_known_args()
if _cli_args.seed is not None:
    SEED = _cli_args.seed
if _cli_args.out_dir is not None:
    OUT_DIR = _cli_args.out_dir
if _cli_args.model_dir is not None:
    MODEL_DIR = _cli_args.model_dir
OBS_MASK  = "v65_strict_los"
PN_GAIN   = 3.0
PN_MAX    = 0.8

# 拦截器状态名称和颜色
STATE_NAMES = {
    InterceptorPolicy.STATE_INIT_GUIDE: "INIT_GUIDE",
    InterceptorPolicy.STATE_LOCKED:     "LOCKED",
    InterceptorPolicy.STATE_MISSED:     "MISSED",
    InterceptorPolicy.STATE_ABANDONED:  "ABANDONED",
}
STATE_COLORS = {
    InterceptorPolicy.STATE_INIT_GUIDE: "#999999",
    InterceptorPolicy.STATE_LOCKED:     "#E74C3C",
    InterceptorPolicy.STATE_MISSED:     "#F39C12",
    InterceptorPolicy.STATE_ABANDONED:  "#2C3E50",
}
OFF_COLORS = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6"]  # A0-A3


def load_policies(raw_env, model_dir):
    parser = get_macpo_config()
    args = parser.parse_known_args([
        '--algorithm_name','mappo',
        '--hidden_size', str(HIDDEN),
        '--layer_N', str(LAYER_N),
        '--lr','5e-4','--critic_lr','5e-4',
        '--use_feature_normalization','--use_recurrent_policy',
    ])[0]
    obs_sp   = raw_env.observation_space[0]
    share_sp = raw_env.share_observation_space[0]
    act_sp   = raw_env.action_space[0]
    policies = []
    for i in range(raw_env.n_agents):
        p = R_MAPPOPolicy(args, obs_sp, share_sp, act_sp, device=torch.device('cpu'))
        sd = torch.load(os.path.join(model_dir, f'actor_agent{i}.pt'), map_location='cpu')
        p.actor.load_state_dict(sd, strict=False)
        p.actor.eval()
        policies.append(p)
    return policies


def run_and_collect(model_dir):
    raw_env = FOVPenetrationEnv(scenario='scenario_1')
    wrapped = PhaseMaskedFOVWrapper(raw_env, mode=OBS_MASK)
    wrapped = TerminalPNActionWrapper(wrapped, gain=PN_GAIN, max_action=PN_MAX)
    policies = load_policies(raw_env, model_dir)

    wrapped.seed(SEED)
    obs, _, _ = wrapped.reset()
    n_off = raw_env.n_offensive
    n_def = raw_env.n_defensive

    rnn = [np.zeros((1,1,HIDDEN), np.float32) for _ in range(n_off)]
    masks = [np.ones((1,1), np.float32) for _ in range(n_off)]

    # 记录表
    steps_log    = []   # step index
    time_log     = []   # time in seconds
    # 拦截器: lock_mode, locked_target, fov_loss_counter, alive, (x,y,z)
    def_state      = [[] for _ in range(n_def)]   # def_state[di][t] = lock_mode int
    def_target     = [[] for _ in range(n_def)]   # which off idx is locked (-1=none)
    def_init_target= [[] for _ in range(n_def)]   # initial_assigned_target_idx (匈牙利分配，全程固定)
    def_fov_loss   = [[] for _ in range(n_def)]
    def_alive      = [[] for _ in range(n_def)]
    def_x          = [[] for _ in range(n_def)]
    def_y          = [[] for _ in range(n_def)]
    # 进攻方: alive, dist_to_hvt, locked_by_count, terminal_phase, (x,y)
    off_alive    = [[] for _ in range(n_off)]
    off_dist     = [[] for _ in range(n_off)]
    off_locked   = [[] for _ in range(n_off)]   # 被多少个拦截器 LOCKED 状态追击
    off_terminal = [[] for _ in range(n_off)]
    off_x        = [[] for _ in range(n_off)]
    off_y        = [[] for _ in range(n_off)]

    events = []  # {"step", "time_s", "type", "desc"}

    prev_def_modes = [None]*n_def
    prev_def_tgt   = [None]*n_def

    for step in range(raw_env.max_steps):
        t = step * raw_env.dt

        # 当前时刻采样拦截器状态 (在 step 之前，即施加动作前)
        # 构建 locked_count per off (STATE_LOCKED 的拦截器数量)
        locked_count = [0]*n_off
        for di, pol in enumerate(raw_env.defensive_policies):
            m   = pol.lock_mode
            tgt = pol.current_locked_target_idx
            fv  = pol.fov_loss_counter
            alive_di = raw_env.defensives[di].alive

            init_tgt = pol.initial_assigned_target_idx
            def_state[di].append(m)
            def_target[di].append(tgt if tgt is not None else -1)
            def_init_target[di].append(init_tgt if init_tgt is not None else -1)
            def_fov_loss[di].append(fv)
            def_alive[di].append(int(alive_di))
            def_x[di].append(raw_env.defensives[di].x)
            def_y[di].append(raw_env.defensives[di].y)

            if alive_di and m == InterceptorPolicy.STATE_LOCKED and tgt is not None:
                locked_count[tgt] += 1

            # 事件检测：状态切换
            if prev_def_modes[di] is not None and m != prev_def_modes[di]:
                old_name = STATE_NAMES.get(prev_def_modes[di], str(prev_def_modes[di]))
                new_name = STATE_NAMES.get(m, str(m))
                events.append({
                    "step": step, "time_s": round(t, 2),
                    "type": f"D{di}_state_change",
                    "desc": f"拦截器D{di}: {old_name} → {new_name}, target={tgt}",
                })
            if prev_def_tgt[di] is not None and tgt != prev_def_tgt[di]:
                events.append({
                    "step": step, "time_s": round(t, 2),
                    "type": f"D{di}_target_change",
                    "desc": f"拦截器D{di}: 目标切换 {prev_def_tgt[di]} → {tgt}",
                })
            prev_def_modes[di] = m
            prev_def_tgt[di]   = tgt

        hvt = raw_env.hvt
        for oi, off in enumerate(raw_env.offensives):
            off_alive[oi].append(int(off.alive))
            off_dist[oi].append(off.distance_to(hvt.x, hvt.y, hvt.z))
            off_locked[oi].append(locked_count[oi])
            off_x[oi].append(off.x)
            off_y[oi].append(off.y)

        steps_log.append(step)
        time_log.append(t)

        # 执行动作
        actions = []
        for i in range(n_off):
            with torch.no_grad():
                a, _, h = policies[i].actor(
                    np.array(obs[i]).reshape(1,-1), rnn[i], masks[i], deterministic=True)
                actions.append(a.squeeze().numpy())
                rnn[i] = h.numpy()

        obs, _, rewards, costs, dones, infos, _ = wrapped.step(actions)

        # 终端制导状态采样 (step 后)
        for oi in range(n_off):
            is_term = False
            if hasattr(wrapped, '_phase_flags'):
                is_term = bool(wrapped._phase_flags[oi])
            elif hasattr(wrapped, 'env') and hasattr(wrapped.env, '_phase_flags'):
                is_term = bool(wrapped.env._phase_flags[oi])
            off_terminal[oi].append(int(is_term))

        if dones[0]:
            break

    T = len(steps_log)
    time_arr = np.array(time_log[:T])

    # 确定谁是 striker (命中 HVT 的进攻方)
    striker_idx = -1
    for oi, off in enumerate(raw_env.offensives):
        if off.hit_hvt:
            striker_idx = oi
            break
    hit_step = raw_env.current_step
    hit_time = hit_step * raw_env.dt

    return {
        "time_arr": time_arr,
        "steps": T,
        "hit_time": hit_time,
        "hit_step": hit_step,
        "striker_idx": striker_idx,
        "n_off": n_off, "n_def": n_def,
        "def_state": def_state,
        "def_target": def_target,
        "def_init_target": def_init_target,
        "def_fov_loss": def_fov_loss,
        "def_alive": def_alive,
        "def_x": def_x, "def_y": def_y,
        "off_alive": off_alive,
        "off_dist": off_dist,
        "off_locked": off_locked,
        "off_terminal": off_terminal,
        "off_x": off_x, "off_y": off_y,
        "events": events,
        "hvt": (raw_env.hvt.x, raw_env.hvt.y),
    }


def clip_to_T(lst, T):
    return lst[:T] if len(lst) >= T else lst + [lst[-1]]*(T-len(lst))


def fig1_allocation_timeline(data, out_dir):
    """4 个子图, 每行 = 1 个拦截器, 颜色 = lock_mode, 矩形填色"""
    T = data["steps"]
    time_arr = data["time_arr"]
    n_def    = data["n_def"]
    hit_time = data["hit_time"]

    fig, axes = plt.subplots(n_def, 1, figsize=(12, 6), sharex=True)
    if n_def == 1:
        axes = [axes]

    for di in range(n_def):
        ax = axes[di]
        states = clip_to_T(data["def_state"][di], T)
        alive  = clip_to_T(data["def_alive"][di], T)
        targets= clip_to_T(data["def_target"][di], T)

        prev_s, seg_start = states[0], 0
        for t_idx in range(1, T):
            if states[t_idx] != prev_s or t_idx == T-1:
                end_idx = t_idx if states[t_idx] != prev_s else T
                color = STATE_COLORS.get(prev_s, "#CCCCCC")
                ax.axvspan(time_arr[seg_start], time_arr[min(end_idx, T-1)],
                           alpha=0.7, color=color, lw=0)
                # 在中间写目标编号
                mid_t = (time_arr[seg_start] + time_arr[min(end_idx, T-1)]) / 2
                tgt_mid = targets[min((seg_start+end_idx)//2, T-1)]
                label = f"→A{tgt_mid}" if tgt_mid >= 0 else "─"
                sname = STATE_NAMES.get(prev_s, "?")[:3]
                ax.text(mid_t, 0.5, f"{sname}\n{label}", ha='center', va='center',
                        fontsize=7, transform=ax.get_xaxis_transform())
                seg_start = t_idx
                prev_s = states[t_idx]

        # 拦截器死亡标记
        if 0 in alive:
            death_t = time_arr[alive.index(0)] if 0 in alive else None
            if death_t:
                ax.axvline(death_t, color='black', ls=':', lw=1.2, label='死亡')

        ax.axvline(hit_time, color='gold', ls='--', lw=1.5, label=f'HVT命中 t={hit_time:.1f}s')
        ax.set_ylabel(f'D{di}', fontsize=9, rotation=0, labelpad=30)
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    # 图例
    patches = [mpatches.Patch(color=c, label=STATE_NAMES[s])
               for s, c in STATE_COLORS.items()]
    patches.append(Line2D([0],[0], color='gold', ls='--', lw=1.5, label=f'HVT命中'))
    axes[-1].set_xlabel("时间 (s)", fontsize=10)
    axes[0].set_title("拦截器锁定状态时间线（颜色=状态，标注=目标编号）", fontsize=11)
    fig.legend(handles=patches, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "allocation_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")


def fig2_lock_pressure(data, out_dir):
    """每架进攻方被 LOCKED 状态拦截器追击的数量"""
    T        = data["steps"]
    time_arr = data["time_arr"]
    n_off    = data["n_off"]
    hit_time = data["hit_time"]
    striker  = data["striker_idx"]

    fig, ax = plt.subplots(figsize=(12, 4))
    for oi in range(n_off):
        lk = np.array(clip_to_T(data["off_locked"][oi], T), dtype=float)
        al = np.array(clip_to_T(data["off_alive"][oi], T), dtype=float)
        lk[al == 0] = np.nan
        lw = 2.5 if oi == striker else 1.2
        ls = '-' if oi == striker else '--'
        label = f"A{oi} (striker)" if oi == striker else f"A{oi}"
        ax.plot(time_arr, lk, color=OFF_COLORS[oi], lw=lw, ls=ls, label=label)

    ax.axhline(0, color='green', ls=':', lw=1, alpha=0.6, label='锁定压力=0（空窗）')
    ax.axvline(hit_time, color='gold', ls='--', lw=1.5, label=f'HVT命中 t={hit_time:.1f}s')
    ax.set_xlabel("时间 (s)", fontsize=10)
    ax.set_ylabel("被 LOCKED 状态追击的拦截器数", fontsize=10)
    ax.set_title("进攻方锁定压力时间线（=0 表示分配失效/空窗）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "lock_pressure.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")


def fig3_trajectory_xy(data, out_dir):
    """XY 平面轨迹, 标注分配失效时刻 (striker 锁定压力=0 的窗口)"""
    T        = data["steps"]
    n_off    = data["n_off"]
    n_def    = data["n_def"]
    striker  = data["striker_idx"]
    hvt_xy   = data["hvt"]
    hit_time = data["hit_time"]
    time_arr = data["time_arr"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制轨迹
    for oi in range(n_off):
        xs = np.array(clip_to_T(data["off_x"][oi], T))
        ys = np.array(clip_to_T(data["off_y"][oi], T))
        al = np.array(clip_to_T(data["off_alive"][oi], T))
        lw = 2.5 if oi == striker else 1.2
        label = f"A{oi} (striker)" if oi == striker else f"A{oi}"
        ax.plot(xs[al==1], ys[al==1], color=OFF_COLORS[oi], lw=lw, label=label, alpha=0.85)
        ax.plot(xs[0], ys[0], 'o', color=OFF_COLORS[oi], ms=7)
        # 死亡标记
        dead_idx = np.where(np.diff(al) < 0)[0]
        if len(dead_idx):
            ax.plot(xs[dead_idx[0]], ys[dead_idx[0]], 'x', color=OFF_COLORS[oi], ms=10, mew=2)

    for di in range(n_def):
        xs = np.array(data["def_x"][di][:T])
        ys = np.array(data["def_y"][di][:T])
        al = np.array(data["def_alive"][di][:T])
        ax.plot(xs[al==1], ys[al==1], color='#E74C3C', lw=1.0, alpha=0.5,
                label=f"D{di}" if di == 0 else "_")
        ax.plot(xs[0], ys[0], 's', color='#E74C3C', ms=6)

    # HVT
    ax.scatter([hvt_xy[0]], [hvt_xy[1]], s=150, marker='*', color='gold',
               edgecolor='k', zorder=10, label='HVT')

    # 在 striker 轨迹上标注"锁定压力=0"的窗口（分配失效区段）
    if striker >= 0:
        lk = np.array(clip_to_T(data["off_locked"][striker], T))
        al = np.array(clip_to_T(data["off_alive"][striker], T))
        xs = np.array(clip_to_T(data["off_x"][striker], T))
        ys = np.array(clip_to_T(data["off_y"][striker], T))
        gap_mask = (lk == 0) & (al == 1)
        if np.any(gap_mask):
            ax.scatter(xs[gap_mask], ys[gap_mask], s=25, color='lime',
                       zorder=8, label='striker 空窗（无拦截器LOCKED）', alpha=0.6)
        # 标注首次进入末端制导
        term = np.array(clip_to_T(data["off_terminal"][striker], T))
        if np.any(term):
            first_term = np.where(term)[0][0]
            ax.scatter([xs[first_term]], [ys[first_term]], s=120, marker='^',
                       color='cyan', edgecolor='k', zorder=11, label=f'PN制导切入 t={time_arr[first_term]:.1f}s')

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title("XY 平面轨迹（绿点=striker空窗段，△=PN切入点）", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_xy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")


def fig4_dist_to_hvt(data, out_dir):
    """所有进攻方距 HVT 距离 + 末端制导切换点"""
    T        = data["steps"]
    time_arr = data["time_arr"]
    n_off    = data["n_off"]
    striker  = data["striker_idx"]
    hit_time = data["hit_time"]

    fig, ax = plt.subplots(figsize=(12, 4))
    for oi in range(n_off):
        dist = np.array(clip_to_T(data["off_dist"][oi], T))
        al   = np.array(clip_to_T(data["off_alive"][oi], T))
        term = np.array(clip_to_T(data["off_terminal"][oi], T))
        dist_plot = dist.copy().astype(float)
        dist_plot[al == 0] = np.nan
        lw = 2.5 if oi == striker else 1.2
        label = f"A{oi} (striker)" if oi == striker else f"A{oi}"
        ax.plot(time_arr, dist_plot, color=OFF_COLORS[oi], lw=lw, label=label)
        if np.any(term):
            ft = np.where(term)[0][0]
            ax.axvline(time_arr[ft], color=OFF_COLORS[oi], ls=':', lw=1.2, alpha=0.7)
            ax.text(time_arr[ft]+0.3, dist_plot[ft], f"PN↑A{oi}", fontsize=7,
                    color=OFF_COLORS[oi])

    ax.axhline(5.0, color='gold', ls='--', lw=1.2, label='命中阈值 5m')
    ax.axvline(hit_time, color='gold', ls='-', lw=1.5, alpha=0.8)
    ax.set_xlabel("时间 (s)", fontsize=10)
    ax.set_ylabel("距 HVT 距离 (m)", fontsize=10)
    ax.set_title("进攻方 vs HVT 距离 + PN制导切入（虚线）", fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dist_to_hvt.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")


def summarize_events(data, out_dir):
    """打印 + 保存事件列表 + 关键统计"""
    events   = data["events"]
    striker  = data["striker_idx"]
    T        = data["steps"]
    time_arr = data["time_arr"]
    hit_time = data["hit_time"]

    # 计算 striker 的空窗统计
    if striker >= 0:
        lk = np.array(clip_to_T(data["off_locked"][striker], T))
        al = np.array(clip_to_T(data["off_alive"][striker], T))
        gap_steps = int(np.sum((lk == 0) & (al == 1)))
        gap_secs  = round(gap_steps * 0.01, 2)
        total_alive_secs = round(int(np.sum(al == 1)) * 0.01, 2)
        first_gap_t = None
        for i in range(T):
            if lk[i] == 0 and al[i] == 1:
                first_gap_t = round(time_arr[i], 2)
                break
    else:
        gap_secs = 0.0; total_alive_secs = 0.0; first_gap_t = None

    # 每架进攻方死亡时间
    deaths = {}
    for oi in range(data["n_off"]):
        al = np.array(clip_to_T(data["off_alive"][oi], T))
        dead = np.where(np.diff(al) < 0)[0]
        if len(dead):
            deaths[oi] = round(time_arr[dead[0]], 2)
        elif oi == striker:
            deaths[oi] = None  # survived

    summary = {
        "seed": SEED,
        "result": "SUCCESS",
        "striker_agent": striker,
        "hit_time_s": hit_time,
        "total_steps": T,
        "striker_gap_secs_no_lock": gap_secs,
        "striker_alive_secs": total_alive_secs,
        "striker_gap_fraction": round(gap_secs / max(total_alive_secs, 0.01), 3),
        "first_gap_time_s": first_gap_t,
        "attacker_death_times": deaths,
        "state_change_events": [e for e in events if "state_change" in e["type"]],
        "target_change_events": [e for e in events if "target_change" in e["type"]],
    }

    path = os.path.join(out_dir, "ep_events.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {path}")

    print("\n========== ep02 突防机制分析 ==========")
    print(f"  突防者 (striker):    A{striker}")
    print(f"  命中时刻:            t = {hit_time:.1f}s  (step {data['hit_step']})")
    print(f"  striker 存活时长:    {total_alive_secs}s")
    print(f"  striker 空窗总时长:  {gap_secs}s  ({summary['striker_gap_fraction']*100:.1f}% 时间无 LOCKED 拦截器)")
    print(f"  首次空窗出现:        t = {first_gap_t}s")
    print(f"\n  进攻方死亡时刻: {deaths}")
    print(f"\n  拦截器状态切换事件 ({len(summary['state_change_events'])} 次):")
    for e in summary["state_change_events"]:
        print(f"    t={e['time_s']:.1f}s  {e['desc']}")
    print(f"\n  拦截器目标切换事件 ({len(summary['target_change_events'])} 次):")
    for e in summary["target_change_events"]:
        print(f"    t={e['time_s']:.1f}s  {e['desc']}")

    return summary


def fig5_target_assignment_per_interceptor(data, out_dir):
    """
    每个拦截器"实际追击目标"的时间线：
      - initial_assigned_target_idx（匈牙利初始分配）在 INIT_GUIDE 阶段
      - current_locked_target_idx 在 LOCKED/MISSED/ABANDONED 阶段
    颜色 = 目标编号，-1 = 无目标（MISSED/ABANDONED 且无候选）
    同时用灰色背景条标出拦截器的状态（已死/MISSED/LOCKED）
    """
    T        = data["steps"]
    time_arr = data["time_arr"]
    n_def    = data["n_def"]
    n_off    = data["n_off"]
    hit_time = data["hit_time"]

    # 颜色：A0-A3 的目标编号 → 颜色，-1 → 白色
    tgt_colors = {-1: "#FFFFFF", 0: OFF_COLORS[0], 1: OFF_COLORS[1],
                  2: OFF_COLORS[2], 3: OFF_COLORS[3]}

    fig, axes = plt.subplots(n_def, 1, figsize=(13, 7), sharex=True)
    if n_def == 1:
        axes = [axes]

    for di in range(n_def):
        ax = axes[di]
        states       = clip_to_T(data["def_state"][di], T)
        targets      = clip_to_T(data["def_target"][di], T)       # current_locked (-1=无锁)
        init_targets = clip_to_T(data["def_init_target"][di], T)  # 匈牙利初始分配（全程固定）
        alive        = clip_to_T(data["def_alive"][di], T)

        # 构建"展示目标"序列:
        #   INIT_GUIDE → 用 initial_assigned_target_idx（浅色，飞向但未触发FOV锁定）
        #   其他状态   → 用 current_locked_target_idx  （深色，已FOV锁定后追击）
        display_tgt   = []
        is_init_phase = []
        for t_i in range(T):
            if states[t_i] == InterceptorPolicy.STATE_INIT_GUIDE:
                display_tgt.append(init_targets[t_i])
                is_init_phase.append(True)
            else:
                display_tgt.append(targets[t_i])
                is_init_phase.append(False)

        # 按(展示目标, 是否初始阶段)分段填色
        def _flush(s, e, tgt, init_ph):
            if s >= e:
                return
            color = tgt_colors.get(tgt, "#CCCCCC")
            alpha = 0.30 if init_ph else 0.85
            ax.axvspan(time_arr[s], time_arr[min(e, T-1)], alpha=alpha, color=color, lw=0)
            mid_t = (time_arr[s] + time_arr[min(e, T-1)]) / 2
            lbl = f"A{tgt}" if tgt >= 0 else "—"
            sublbl = "(飞向)" if init_ph else "(锁定)"
            ax.text(mid_t, 0.55, lbl, ha='center', va='center',
                    fontsize=9, fontweight='normal' if init_ph else 'bold',
                    color='#555555' if init_ph else 'black',
                    transform=ax.get_xaxis_transform())
            ax.text(mid_t, 0.25, sublbl, ha='center', va='center',
                    fontsize=6.5, color='#777777' if init_ph else '#222222',
                    transform=ax.get_xaxis_transform())

        prev_tgt, prev_init = display_tgt[0], is_init_phase[0]
        seg_start = 0
        for t_idx in range(1, T):
            if display_tgt[t_idx] != prev_tgt or is_init_phase[t_idx] != prev_init:
                _flush(seg_start, t_idx - 1, prev_tgt, prev_init)
                seg_start = t_idx
                prev_tgt  = display_tgt[t_idx]
                prev_init = is_init_phase[t_idx]
        _flush(seg_start, T - 1, prev_tgt, prev_init)

        # 拦截器死亡标记
        if 0 in alive:
            death_idx = alive.index(0)
            ax.axvline(time_arr[death_idx], color='black', ls=':', lw=1.5)
            ax.text(time_arr[death_idx], 0.95, '†死',
                    ha='left', va='top', fontsize=7,
                    transform=ax.get_xaxis_transform())

        ax.axvline(hit_time, color='gold', ls='--', lw=1.8)
        ax.set_ylabel(f'D{di}', fontsize=9, rotation=0, labelpad=30)
        ax.set_yticks([])
        ax.set_ylim(0, 1)

        # 右侧注释匈牙利初始分配目标
        hungarian_tgt = init_targets[0]
        ax.text(1.01, 0.5,
                f"匈牙利:\nA{hungarian_tgt}" if hungarian_tgt >= 0 else "匈牙利:\n—",
                ha='left', va='center', fontsize=7,
                transform=ax.transAxes, color='#333333')

    # 图例
    patches_locked = [mpatches.Patch(facecolor=tgt_colors[i], alpha=0.85,
                                     label=f'A{i} 锁定追击', edgecolor='#666')
                      for i in range(n_off)]
    patches_init   = [mpatches.Patch(facecolor=tgt_colors[i], alpha=0.30,
                                     label=f'A{i} 飞向(未锁)', edgecolor='#666')
                      for i in range(n_off)]
    patches_extra  = [
        mpatches.Patch(facecolor='#FFFFFF', label='无目标', edgecolor='#999'),
        Line2D([0],[0], color='gold', ls='--', lw=1.8, label=f'HVT命中 t={hit_time:.1f}s'),
    ]
    axes[-1].set_xlabel("时间 (s)", fontsize=10)
    axes[0].set_title(
        "拦截器目标时间线（浅色=飞向/INIT_GUIDE未锁定，深色=FOV触发锁定后追击）",
        fontsize=11)
    fig.legend(handles=patches_locked + patches_init + patches_extra,
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=7.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "target_assignment_per_interceptor.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[1/6] 复现 seed={SEED}, 逐步采集拦截器分配状态...")
    data = run_and_collect(MODEL_DIR)
    print(f"      完成: {data['steps']} 步, striker=A{data['striker_idx']}, hit_t={data['hit_time']:.1f}s")

    # 早退: 不成功的回合只写一个简短 marker, 不画图（节省时间）
    if data['striker_idx'] < 0:
        marker = os.path.join(OUT_DIR, "ep_events.json")
        with open(marker, 'w') as f:
            json.dump({"seed": SEED, "result": "MISS",
                       "total_steps": data['steps']}, f, indent=2)
        print(f"  [MISS] 无突防, 仅写入 marker, 跳过绘图")
        sys.exit(2)  # 用退出码区分

    print("[2/6] 绘制拦截器锁定状态时间线...")
    fig1_allocation_timeline(data, OUT_DIR)

    print("[3/6] 绘制锁定压力图...")
    fig2_lock_pressure(data, OUT_DIR)

    print("[4/6] 绘制 XY 轨迹 + 空窗标注...")
    fig3_trajectory_xy(data, OUT_DIR)

    print("[5/6] 绘制距 HVT 距离 + PN 切入...")
    fig4_dist_to_hvt(data, OUT_DIR)

    print("[6/6] 绘制每拦截器追击目标时间线（含初始分配）...")
    fig5_target_assignment_per_interceptor(data, OUT_DIR)

    summarize_events(data, OUT_DIR)
    print(f"\n所有文件已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
