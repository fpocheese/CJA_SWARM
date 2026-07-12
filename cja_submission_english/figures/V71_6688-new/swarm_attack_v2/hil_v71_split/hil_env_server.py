#!/usr/bin/env python3
"""Server-side V71 environment for split HIL closed-loop simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.fov_penetration import FOVPenetrationEnv
from hil_v71_split.fixed_initial_state import reset_with_fixed_initial
from hil_v71_split.hil_protocol import JsonLineSocket, listen
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


CASES = {
    "4v4": (4, 4, 90000),
    "6v6": (6, 6, 60000),
    "8v8": (8, 8, 80000),
    "10v10": (10, 10, 100000),
}


def make_env(case: str):
    n_off, n_def, _ = CASES[case]
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": n_off, "n_defensive": n_def},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    return raw_env, env


def clean_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def vector_from(info: dict, key: str, n: int) -> np.ndarray:
    value = info.get(key, None)
    if value is None:
        return np.zeros(n, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(n, dtype=np.float32)
    count = min(n, arr.size)
    if count:
        out[:count] = arr[:count]
    return out


def matrix_from(info: dict, key: str, rows: int, cols: int) -> np.ndarray:
    value = info.get(key, None)
    if value is None:
        return np.zeros((rows, cols), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    out = np.zeros((rows, cols), dtype=np.float32)
    if arr.ndim == 2:
        r = min(rows, arr.shape[0])
        c = min(cols, arr.shape[1])
        out[:r, :c] = arr[:r, :c]
    else:
        flat = arr.reshape(-1)
        count = min(rows * cols, flat.size)
        out.reshape(-1)[:count] = flat[:count]
    return out


def init_full_record(n_off: int, n_def: int):
    rec = {
        "steps": [], "time": [], "actor_actions": [],
        "off_x": [[] for _ in range(n_off)], "off_y": [[] for _ in range(n_off)],
        "off_z": [[] for _ in range(n_off)], "off_v": [[] for _ in range(n_off)],
        "off_heading": [[] for _ in range(n_off)], "off_gamma": [[] for _ in range(n_off)],
        "off_an_pitch": [[] for _ in range(n_off)], "off_an_yaw": [[] for _ in range(n_off)],
        "off_lbc": [[] for _ in range(n_off)], "off_alive": [[] for _ in range(n_off)],
        "off_hit": [[] for _ in range(n_off)], "off_d_hvt": [[] for _ in range(n_off)],
        "def_x": [[] for _ in range(n_def)], "def_y": [[] for _ in range(n_def)],
        "def_z": [[] for _ in range(n_def)], "def_v": [[] for _ in range(n_def)],
        "def_an": [[] for _ in range(n_def)],
        "def_an_pitch": [[] for _ in range(n_def)], "def_an_yaw": [[] for _ in range(n_def)],
        "def_initial_target": [[] for _ in range(n_def)],
        "def_assigned_target": [[] for _ in range(n_def)],
        "def_lmode": [[] for _ in range(n_def)], "def_ltgt": [[] for _ in range(n_def)],
        "def_alive": [[] for _ in range(n_def)],
        "assign_cost": [], "fov_sat": [],
    }
    game = {
        "decoy_Phi": [], "pen_N_eff": [],
        "esc_Gamma_mean": [], "esc_Xi_mean": [],
        "esc_Gamma_matrix": [], "esc_Xi_matrix": [],
        "decoy_role_decoy": [[] for _ in range(n_off)],
        "decoy_role_pen": [[] for _ in range(n_off)],
        "decoy_role_stealth": [[] for _ in range(n_off)],
        "decoy_lock_pressure": [[] for _ in range(n_off)],
        "pen_P_pen": [[] for _ in range(n_off)],
        "esc_E_esc": [[] for _ in range(n_off)],
        "hvt_P_hit": [[] for _ in range(n_off)],
        "hvt_rho": [[] for _ in range(n_off)],
        "hvt_closing": [[] for _ in range(n_off)],
        "def_lmode": [[] for _ in range(n_def)],
        "def_ltgt": [[] for _ in range(n_def)],
    }
    return rec, game


def append_full_trajectory(rec: dict, raw_env, actions, step: int):
    hvt = raw_env.hvt
    n_off = len(raw_env.offensives)
    n_def = len(raw_env.defensives)
    rec["steps"].append(int(step))
    rec["time"].append(float(step * raw_env.dt))
    rec["actor_actions"].append(np.asarray(actions, dtype=np.float32))

    for i, off in enumerate(raw_env.offensives):
        rec["off_x"][i].append(float(off.x))
        rec["off_y"][i].append(float(off.y))
        rec["off_z"][i].append(float(off.z))
        rec["off_v"][i].append(clean_float(getattr(off, "v", 0.0)))
        rec["off_heading"][i].append(clean_float(getattr(off, "heading", 0.0)))
        rec["off_gamma"][i].append(clean_float(getattr(off, "gamma", 0.0)))
        rec["off_an_pitch"][i].append(clean_float(getattr(off, "an_pitch", 0.0)))
        rec["off_an_yaw"][i].append(clean_float(getattr(off, "an_yaw", 0.0)))
        rec["off_lbc"][i].append(int(getattr(off, "locked_by_count", 0)))
        rec["off_alive"][i].append(int(getattr(off, "alive", True)))
        rec["off_hit"][i].append(int(getattr(off, "hit_hvt", False)))
        rec["off_d_hvt"][i].append(float(off.distance_to(hvt.x, hvt.y, hvt.z)))

    policies = getattr(raw_env, "defensive_policies", [])
    for j, defender in enumerate(raw_env.defensives):
        an_pitch = clean_float(getattr(defender, "an_pitch", 0.0))
        an_yaw = clean_float(getattr(defender, "an_yaw", 0.0))
        policy = policies[j] if j < len(policies) else None
        initial = getattr(policy, "initial_assigned_target_idx", None) if policy is not None else None
        assigned = getattr(policy, "assigned_target_idx", None) if policy is not None else None
        lock_mode = getattr(policy, "lock_mode", 0) if policy is not None else 0
        rec["def_x"][j].append(float(defender.x))
        rec["def_y"][j].append(float(defender.y))
        rec["def_z"][j].append(float(defender.z))
        rec["def_v"][j].append(clean_float(getattr(defender, "v", 0.0)))
        rec["def_an"][j].append(float(np.sqrt(an_pitch ** 2 + an_yaw ** 2) / 9.80665))
        rec["def_an_pitch"][j].append(float(an_pitch / 9.80665))
        rec["def_an_yaw"][j].append(float(an_yaw / 9.80665))
        rec["def_initial_target"][j].append(int(initial) if initial is not None else -1)
        rec["def_assigned_target"][j].append(int(assigned) if assigned is not None else -1)
        rec["def_lmode"][j].append(int(lock_mode))
        rec["def_ltgt"][j].append(int(assigned) if assigned is not None else -1)
        rec["def_alive"][j].append(int(getattr(defender, "alive", True)))

    cmat = np.zeros((n_def, n_off), dtype=np.float32)
    for j, defender in enumerate(raw_env.defensives):
        for i, off in enumerate(raw_env.offensives):
            cmat[j, i] = defender.distance_to(off.x, off.y, off.z)
    rec["assign_cost"].append(cmat)
    locked = sum(1 for policy in policies if getattr(policy, "lock_mode", 0) == 2)
    rec["fov_sat"].append(float(locked / max(n_def, 1)))


def append_full_game(game: dict, raw_env):
    n_off = len(raw_env.offensives)
    n_def = len(raw_env.defensives)
    decoy = getattr(raw_env, "_ap_decoy_info", {}) or {}
    pen = getattr(raw_env, "_ap_pen_info", {}) or {}
    esc = getattr(raw_env, "_ap_esc_info", {}) or {}
    hvt = getattr(raw_env, "_ap_hvt_info", {}) or {}

    game["decoy_Phi"].append(clean_float(decoy.get("Phi_decoy", 0.0)))
    game["pen_N_eff"].append(clean_float(pen.get("N_eff", 0.0)))
    gamma = matrix_from(esc, "_Gamma_matrix", n_off, n_def)
    xi = matrix_from(esc, "_Xi_matrix", n_off, n_def)
    game["esc_Gamma_matrix"].append(gamma)
    game["esc_Xi_matrix"].append(xi)
    game["esc_Gamma_mean"].append(clean_float(np.mean(gamma) if gamma.size else 0.0))
    game["esc_Xi_mean"].append(clean_float(np.mean(xi) if xi.size else 0.0))

    for i in range(n_off):
        game["decoy_role_decoy"][i].append(vector_from(decoy, "role_decoy_per_agent", n_off)[i])
        game["decoy_role_pen"][i].append(vector_from(decoy, "role_penetrate_per_agent", n_off)[i])
        game["decoy_role_stealth"][i].append(vector_from(decoy, "role_stealth_per_agent", n_off)[i])
        game["decoy_lock_pressure"][i].append(vector_from(decoy, "lock_pressure_per_agent", n_off)[i])
        game["pen_P_pen"][i].append(vector_from(pen, "P_pen_per_agent", n_off)[i])
        game["esc_E_esc"][i].append(vector_from(esc, "E_i_esc", n_off)[i])
        game["hvt_P_hit"][i].append(vector_from(hvt, "P_hit_per_agent", n_off)[i])
        game["hvt_rho"][i].append(vector_from(hvt, "rho_per_agent", n_off)[i])
        game["hvt_closing"][i].append(vector_from(hvt, "closing_per_agent", n_off)[i])

    policies = getattr(raw_env, "defensive_policies", [])
    for j in range(n_def):
        policy = policies[j] if j < len(policies) else None
        assigned = getattr(policy, "assigned_target_idx", None) if policy is not None else None
        game["def_lmode"][j].append(int(getattr(policy, "lock_mode", 0)) if policy is not None else 0)
        game["def_ltgt"][j].append(int(assigned) if assigned is not None else -1)


def finalize_npz_dict(data: dict):
    out = {}
    for key, value in data.items():
        if key in ("death_step", "hit_step"):
            continue
        if isinstance(value, list):
            out[key] = np.asarray(value)
        else:
            out[key] = value
    return out


def accept_clients(host: str, port: int, n_agents: int) -> dict[int, JsonLineSocket]:
    server = listen(host, port)
    clients: dict[int, JsonLineSocket] = {}
    print(f"[server] listening on {host}:{port}, waiting for {n_agents} policy nodes", flush=True)
    try:
        while len(clients) < n_agents:
            sock, addr = server.accept()
            sock.setsockopt(6, 1, 1)  # IPPROTO_TCP, TCP_NODELAY
            peer = JsonLineSocket(sock)
            hello = peer.recv()
            if hello.get("type") != "hello":
                raise ValueError(f"client {addr} did not send hello: {hello}")
            agent_id = int(hello["agent_id"])
            if agent_id < 0 or agent_id >= n_agents:
                raise ValueError(f"invalid agent_id {agent_id}, expected [0,{n_agents})")
            if agent_id in clients:
                raise ValueError(f"duplicate agent_id {agent_id}")
            clients[agent_id] = peer
            print(f"[server] agent {agent_id} connected from {addr}, hello={hello}", flush=True)
    finally:
        server.close()
    return clients


def run_episode(raw_env, env, clients: dict[int, JsonLineSocket], seed: int,
                max_steps: int, episode: int, fixed_initial_npz: str | None = None,
                trajectory_out: str | None = None,
                full_output_dir: str | None = None,
                case: str | None = None,
                model_dir_label: str | None = None):
    if fixed_initial_npz:
        obs, _, _ = reset_with_fixed_initial(raw_env, env, fixed_initial_npz, seed=seed)
    else:
        env.seed(seed)
        obs, _, _ = env.reset()
    n_agents = env.n_agents
    masks = [1.0 for _ in range(n_agents)]
    hvt = raw_env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in raw_env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0
    traj = None
    full_rec = None
    full_game = None
    if full_output_dir:
        full_rec, full_game = init_full_record(len(raw_env.offensives), len(raw_env.defensives))
    if trajectory_out:
        traj = {
            "steps": [],
            "time": [],
            "actor_actions": [],
            "off_x": [[] for _ in range(n_agents)],
            "off_y": [[] for _ in range(n_agents)],
            "off_z": [[] for _ in range(n_agents)],
            "off_v": [[] for _ in range(n_agents)],
            "off_heading": [[] for _ in range(n_agents)],
            "off_gamma": [[] for _ in range(n_agents)],
            "off_d_hvt": [[] for _ in range(n_agents)],
            "def_x": [[] for _ in range(len(raw_env.defensives))],
            "def_y": [[] for _ in range(len(raw_env.defensives))],
            "def_z": [[] for _ in range(len(raw_env.defensives))],
            "def_v": [[] for _ in range(len(raw_env.defensives))],
        }

    def record(step_idx: int, action_list):
        if traj is None:
            return
        traj["steps"].append(int(step_idx))
        traj["time"].append(float(step_idx * raw_env.dt))
        traj["actor_actions"].append(np.asarray(action_list, dtype=np.float32))
        for i, off in enumerate(raw_env.offensives):
            traj["off_x"][i].append(float(off.x))
            traj["off_y"][i].append(float(off.y))
            traj["off_z"][i].append(float(off.z))
            traj["off_v"][i].append(float(getattr(off, "v", np.nan)))
            traj["off_heading"][i].append(float(getattr(off, "heading", np.nan)))
            traj["off_gamma"][i].append(float(getattr(off, "gamma", np.nan)))
            traj["off_d_hvt"][i].append(float(off.distance_to(hvt.x, hvt.y, hvt.z)))
        for j, defender in enumerate(raw_env.defensives):
            traj["def_x"][j].append(float(defender.x))
            traj["def_y"][j].append(float(defender.y))
            traj["def_z"][j].append(float(defender.z))
            traj["def_v"][j].append(float(getattr(defender, "v", np.nan)))

    for step in range(max_steps):
        for agent_id in range(n_agents):
            clients[agent_id].send({
                "type": "obs",
                "episode": int(episode),
                "step": int(step),
                "seed": int(seed),
                "mask": float(masks[agent_id]),
                "obs": np.asarray(obs[agent_id], dtype=np.float32).reshape(-1).astype(float).tolist(),
            })

        actions = [None for _ in range(n_agents)]
        for agent_id in range(n_agents):
            msg = clients[agent_id].recv()
            if msg.get("type") != "action":
                raise ValueError(f"agent {agent_id} sent unexpected message: {msg}")
            if int(msg.get("agent_id", -1)) != agent_id:
                raise ValueError(f"agent id mismatch: expected {agent_id}, got {msg}")
            actions[agent_id] = np.asarray(msg["action"], dtype=np.float32)

        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}
        record(final_step, actions)
        if full_rec is not None and full_game is not None:
            append_full_trajectory(full_rec, raw_env, actions, final_step)
            append_full_game(full_game, raw_env)
        for i, off in enumerate(raw_env.offensives):
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
                min_step[i] = final_step
        masks = [0.0 if bool(dones[i]) else 1.0 for i in range(n_agents)]
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    summary = {
        "case": case or "",
        "seed": int(seed),
        "episode": int(episode),
        "model_dir": model_dir_label or "",
        "clone_map": {str(i): int(i % 4) for i in range(n_agents)},
        "n_offensive": int(len(raw_env.offensives)),
        "n_defensive": int(len(raw_env.defensives)),
        "success": bool(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": [int(i) for i in getattr(raw_env, "hit_indices", [])],
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * raw_env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_min_dist_m": float(min_d[best_agent]),
        "best_hvt_distance_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
        "min_dist_per_agent_m": [float(x) for x in min_d],
    }
    if traj is not None:
        out_path = Path(trajectory_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            steps=np.asarray(traj["steps"], dtype=np.int64),
            time=np.asarray(traj["time"], dtype=np.float64),
            actor_actions=np.asarray(traj["actor_actions"], dtype=np.float32),
            off_x=np.asarray(traj["off_x"], dtype=np.float64),
            off_y=np.asarray(traj["off_y"], dtype=np.float64),
            off_z=np.asarray(traj["off_z"], dtype=np.float64),
            off_v=np.asarray(traj["off_v"], dtype=np.float64),
            off_heading=np.asarray(traj["off_heading"], dtype=np.float64),
            off_gamma=np.asarray(traj["off_gamma"], dtype=np.float64),
            off_d_hvt=np.asarray(traj["off_d_hvt"], dtype=np.float64),
            def_x=np.asarray(traj["def_x"], dtype=np.float64),
            def_y=np.asarray(traj["def_y"], dtype=np.float64),
            def_z=np.asarray(traj["def_z"], dtype=np.float64),
            def_v=np.asarray(traj["def_v"], dtype=np.float64),
            hvt_x=float(hvt.x),
            hvt_y=float(hvt.y),
            hvt_z=float(hvt.z),
        )
    if full_rec is not None and full_game is not None and full_output_dir:
        full_rec["hvt_x"] = float(hvt.x)
        full_rec["hvt_y"] = float(hvt.y)
        full_rec["hvt_z"] = float(hvt.z)
        full_rec["hit_count"] = int(raw_env.hit_count)
        hit_step = {}
        death_step = {}
        for i, alive_series in enumerate(full_rec["off_alive"]):
            for idx, (alive, hit) in enumerate(zip(alive_series, full_rec["off_hit"][i])):
                step_val = int(full_rec["steps"][idx])
                if hit and i not in hit_step:
                    hit_step[i] = step_val
                if not alive and not hit and i not in death_step:
                    death_step[i] = step_val
        summary["hit_step"] = {str(k): int(v) for k, v in hit_step.items()}
        summary["death_step"] = {str(k): int(v) for k, v in death_step.items()}
        summary["files"] = ["game_data.npz", "summary.json", "trajectory_data.npz"]

        full_dir = Path(full_output_dir)
        full_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(full_dir / "trajectory_data.npz", **finalize_npz_dict(full_rec))
        np.savez_compressed(full_dir / "game_data.npz", **finalize_npz_dict(full_game))
        with (full_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        summary["min_dist_per_agent_m"] = [round(float(x), 3) for x in min_d]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), default="4v4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--fixed-initial-npz", default=None,
                        help="trajectory_data.npz whose first frame is used as the fixed initial scenario")
    parser.add_argument("--out", default="/tmp/v71_hil_split_summary.json")
    parser.add_argument("--trajectory-out", default=None,
                        help="optional npz path for HIL trajectory recording")
    parser.add_argument("--full-output-dir", default=None,
                        help="optional directory for plot_all.py compatible full data")
    parser.add_argument("--model-dir-label", default="",
                        help="model directory label written to full summary.json")
    args = parser.parse_args()

    n_agents, _, seed_base = CASES[args.case]
    seed0 = seed_base if args.seed is None else int(args.seed)
    raw_env, env = make_env(args.case)
    clients = accept_clients(args.host, args.port, n_agents)
    started = time.time()
    summaries = []
    try:
        for ep in range(int(args.episodes)):
            summary = run_episode(raw_env, env, clients, seed0 + ep, int(args.max_steps), ep,
                                  fixed_initial_npz=args.fixed_initial_npz,
                                  trajectory_out=args.trajectory_out if ep == 0 else None,
                                  full_output_dir=args.full_output_dir if ep == 0 else None,
                                  case=args.case,
                                  model_dir_label=args.model_dir_label)
            summaries.append(summary)
            print(f"[server] episode={ep} seed={seed0 + ep} success={int(summary['success'])} "
                  f"reason={summary['done_reason']} best={summary['best_min_dist_m']:.2f}m", flush=True)
    finally:
        for peer in clients.values():
            try:
                peer.send({"type": "close"})
            except Exception:
                pass
            peer.close()

    output = {
        "case": args.case,
        "episodes": len(summaries),
        "success_count": sum(1 for item in summaries if item["success"]),
        "success_rate": sum(1 for item in summaries if item["success"]) / max(len(summaries), 1),
        "elapsed_s": time.time() - started,
        "summaries": summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[server] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
