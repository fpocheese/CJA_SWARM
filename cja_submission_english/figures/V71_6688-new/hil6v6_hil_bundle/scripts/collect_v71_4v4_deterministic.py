#!/usr/bin/env python
"""Collect successful V71 clone-weight samples for fixed 4v4/6v6/8v8 cases.

This is a dedicated experiment script. It does not change the generic scenario
registry or training/evaluation entry points.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from macpo.config import get_config as get_macpo_config
from envs.fov_penetration import FOVPenetrationEnv
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


MODEL_DIR = Path(os.environ.get(
    "V71_MODEL_DIR",
    "outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models",
))
HIDDEN = int(os.environ.get("V71_HIDDEN", "256"))
LAYER_N = int(os.environ.get("V71_LAYER_N", "3"))
MAX_STEPS = int(os.environ.get("V71_MAX_STEPS", "8000"))

_ENV = None
_RAW_ENV = None
_POLICIES = None
_CASE = None


def parse_case(case: str) -> tuple[int, int]:
    token = case.strip().lower()
    if token == "4v4":
        return 4, 4
    if token == "6v6":
        return 6, 6
    if token == "8v8":
        return 8, 8
    raise ValueError(f"Unsupported case {case!r}; expected 4v4, 6v6 or 8v8")


def make_raw_env(n_off: int, n_def: int, seed: int | None = None):
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": n_off, "n_defensive": n_def},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    if seed is not None:
        raw_env.seed(seed)
    return raw_env, env


def make_policy_args():
    parser = get_macpo_config()
    return parser.parse_known_args([
        "--algorithm_name", "mappo",
        "--hidden_size", str(HIDDEN),
        "--layer_N", str(LAYER_N),
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--use_feature_normalization",
        "--use_recurrent_policy",
    ])[0]


def load_cloned_policies(raw_env):
    args = make_policy_args()
    device = torch.device("cpu")
    policies = []
    for agent_id in range(raw_env.n_agents):
        policy = R_MAPPOPolicy(
            args,
            raw_env.observation_space[agent_id],
            raw_env.share_observation_space[agent_id],
            raw_env.action_space[agent_id],
            device=device,
        )
        src_agent = agent_id % 4
        actor_path = MODEL_DIR / f"actor_agent{src_agent}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        policy.actor.load_state_dict(torch.load(actor_path, map_location=device), strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def init_worker(case: str):
    global _ENV, _RAW_ENV, _POLICIES, _CASE
    torch.set_num_threads(1)
    _CASE = case
    n_off, n_def = parse_case(case)
    _RAW_ENV, _ENV = make_raw_env(n_off, n_def)
    _POLICIES = load_cloned_policies(_RAW_ENV)


def actor_actions(policies, obs, rnn_states, masks):
    actions = []
    new_rnn = []
    for agent_id, policy in enumerate(policies):
        obs_tensor = torch.FloatTensor(np.asarray(obs[agent_id]).flatten()).unsqueeze(0)
        with torch.no_grad():
            action, _, hidden = policy.actor(
                obs_tensor,
                rnn_states[agent_id],
                masks[agent_id],
                deterministic=True,
            )
        actions.append(action.cpu().numpy().flatten())
        new_rnn.append(hidden)
    return actions, new_rnn


def run_scan_one(seed: int):
    env = _ENV
    policies = _POLICIES
    if env is None or policies is None:
        raise RuntimeError("Worker not initialized")

    env.seed(seed)
    obs, _, _ = env.reset()
    n_agents = env.n_agents
    hvt = env.hvt
    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    for step in range(MAX_STEPS):
        actions, rnn_states = actor_actions(policies, obs, rnn_states, masks)
        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}

        for i, off in enumerate(env.offensives):
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
                min_step[i] = final_step
        masks = [
            torch.tensor([[0.0 if dones[i] else 1.0]], dtype=torch.float32)
            for i in range(n_agents)
        ]
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    return {
        "case": _CASE,
        "seed": int(seed),
        "success": bool(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ",".join(str(i) for i in env.hit_indices),
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_min_dist_m": float(min(min_d)),
        "best_agent": best_agent,
        "best_min_step": int(min_step[best_agent]),
        "min_dist_per_agent_m": json.dumps([round(float(x), 3) for x in min_d]),
    }


def clean_float(value, default=0.0):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def vector_from(mapping, key: str, n: int, default=0.0):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if value is None:
        return [default] * n
    arr = np.asarray(value).reshape(-1)
    out = [clean_float(v, default) for v in arr[:n]]
    if len(out) < n:
        out.extend([default] * (n - len(out)))
    return out


def matrix_from(mapping, key: str, rows: int, cols: int, default=0.0):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    out = np.full((rows, cols), default, dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        out.fill(float(arr))
        return out
    arr = np.atleast_2d(arr)
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return out


def append_trajectory(rec: dict, raw_env, actions, step: int):
    hvt = raw_env.hvt
    n_off = raw_env.n_offensive
    n_def = raw_env.n_defensive
    rec["steps"].append(step)
    rec["time"].append(step * raw_env.dt)
    rec["actor_actions"].append(np.asarray(actions, dtype=np.float32))

    for i, off in enumerate(raw_env.offensives):
        rec["off_x"][i].append(off.x)
        rec["off_y"][i].append(off.y)
        rec["off_z"][i].append(off.z)
        rec["off_v"][i].append(off.v)
        rec["off_heading"][i].append(off.heading)
        rec["off_gamma"][i].append(off.gamma)
        rec["off_an_pitch"][i].append(off.an_pitch)
        rec["off_an_yaw"][i].append(off.an_yaw)
        rec["off_lbc"][i].append(off.locked_by_count)
        rec["off_alive"][i].append(int(off.alive))
        rec["off_hit"][i].append(int(off.hit_hvt))
        rec["off_d_hvt"][i].append(off.distance_to(hvt.x, hvt.y, hvt.z))

    for j, defender in enumerate(raw_env.defensives):
        rec["def_x"][j].append(defender.x)
        rec["def_y"][j].append(defender.y)
        rec["def_z"][j].append(defender.z)
        rec["def_v"][j].append(defender.v)
        rec["def_an"][j].append(np.sqrt(defender.an_pitch**2 + defender.an_yaw**2) / 9.80665)
        rec["def_an_pitch"][j].append(defender.an_pitch / 9.80665)
        rec["def_an_yaw"][j].append(defender.an_yaw / 9.80665)
        policy = raw_env.defensive_policies[j]
        rec["def_initial_target"][j].append(
            policy.initial_assigned_target_idx if policy.initial_assigned_target_idx is not None else -1
        )
        rec["def_assigned_target"][j].append(
            policy.assigned_target_idx if policy.assigned_target_idx is not None else -1
        )
        attack_target = getattr(policy, "current_attack_target_idx", None)
        rec["def_current_attack_target"][j].append(
            attack_target if attack_target is not None else -1
        )
        rec["def_lmode"][j].append(policy.lock_mode)
        rec["def_ltgt"][j].append(
            attack_target if attack_target is not None else -1
        )
        rec["def_alive"][j].append(int(defender.alive))

    cmat = np.zeros((n_def, n_off), dtype=np.float32)
    for j, defender in enumerate(raw_env.defensives):
        for i, off in enumerate(raw_env.offensives):
            cmat[j, i] = defender.distance_to(off.x, off.y, off.z)
    rec["assign_cost"].append(cmat)
    n_locked = sum(1 for p in raw_env.defensive_policies if p.lock_mode == 2)
    rec["fov_sat"].append(n_locked / max(n_def, 1))


def append_game(game: dict, raw_env):
    n_off = raw_env.n_offensive
    n_def = raw_env.n_defensive
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

    for j, policy in enumerate(raw_env.defensive_policies):
        attack_target = getattr(policy, "current_attack_target_idx", None)
        game["def_lmode"][j].append(policy.lock_mode)
        game["def_current_attack_target"][j].append(
            attack_target if attack_target is not None else -1
        )
        game["def_ltgt"][j].append(
            attack_target if attack_target is not None else -1
        )


def init_record(n_off: int, n_def: int):
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
        "def_current_attack_target": [[] for _ in range(n_def)],
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
        "def_current_attack_target": [[] for _ in range(n_def)],
        "def_ltgt": [[] for _ in range(n_def)],
    }
    return rec, game


def finalize_npz_dict(data: dict):
    out = {}
    for key, value in data.items():
        if key in ("death_step", "hit_step"):
            continue
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                out[key] = np.asarray(value)
            else:
                out[key] = np.asarray(value)
        else:
            out[key] = value
    return out


def record_success(case: str, seed: int, out_dir: Path):
    n_off, n_def = parse_case(case)
    raw_env, env = make_raw_env(n_off, n_def, seed=seed)
    policies = load_cloned_policies(raw_env)
    obs, _, _ = env.reset()
    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_off)]
    masks = [torch.ones(1, 1) for _ in range(n_off)]
    rec, game = init_record(n_off, n_def)
    final_info = {}

    for step in range(MAX_STEPS):
        actions, rnn_states = actor_actions(policies, obs, rnn_states, masks)
        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_info = infos[0] if infos else {}
        cur_step = step + 1
        append_trajectory(rec, raw_env, actions, cur_step)
        append_game(game, raw_env)
        masks = [
            torch.tensor([[0.0 if dones[i] else 1.0]], dtype=torch.float32)
            for i in range(n_off)
        ]
        if all(dones):
            break

    rec["hvt_x"] = raw_env.hvt.x
    rec["hvt_y"] = raw_env.hvt.y
    rec["hvt_z"] = raw_env.hvt.z
    rec["hit_count"] = raw_env.hit_count
    rec["death_step"] = {}
    rec["hit_step"] = {}
    for i, alive_series in enumerate(rec["off_alive"]):
        for idx, (alive, hit) in enumerate(zip(alive_series, rec["off_hit"][i])):
            step_val = int(rec["steps"][idx])
            if hit and i not in rec["hit_step"]:
                rec["hit_step"][i] = step_val
            if not alive and not hit and i not in rec["death_step"]:
                rec["death_step"][i] = step_val

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_out = finalize_npz_dict(rec)
    game_out = finalize_npz_dict(game)
    np.savez_compressed(out_dir / "trajectory_data.npz", **traj_out)
    np.savez_compressed(out_dir / "game_data.npz", **game_out)

    min_d = [float(np.min(np.asarray(x, dtype=float))) for x in rec["off_d_hvt"]]
    best_agent = int(np.argmin(min_d))
    summary = {
        "case": case,
        "seed": int(seed),
        "model_dir": str(MODEL_DIR),
        "clone_map": {str(i): int(i % 4) for i in range(n_off)},
        "n_offensive": n_off,
        "n_defensive": n_def,
        "success": bool(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": [int(i) for i in raw_env.hit_indices],
        "hit_step": {str(k): int(v) for k, v in rec["hit_step"].items()},
        "death_step": {str(k): int(v) for k, v in rec["death_step"].items()},
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(rec["steps"][-1]) if len(rec["steps"]) else 0,
        "final_time_s": float(rec["time"][-1]) if len(rec["time"]) else 0.0,
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_hvt_distance_m": min_d[best_agent],
        "min_dist_per_agent_m": min_d,
        "files": ["game_data.npz", "summary.json", "trajectory_data.npz"],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def write_scan_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: row["seed"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)


def choose_best_success(rows: list[dict]):
    successes = [row for row in rows if row["success"]]
    if not successes:
        return None
    return sorted(
        successes,
        key=lambda row: (
            -int(row["hit_count"]),
            float(row["best_min_dist_m"]),
            int(row["final_step"]),
            -int(row["offensive_alive"]),
        ),
    )[0]


def run_case(case: str, seed_base: int, scan_eps: int, max_eps: int, workers: int, out_root: Path):
    print(f"=== scanning {case}: seed_base={seed_base}, scan_eps={scan_eps}, max_eps={max_eps} ===", flush=True)
    all_rows = []
    batch_start = 0
    while batch_start < max_eps:
        batch_end = min(batch_start + scan_eps, max_eps)
        seeds = [seed_base + i for i in range(batch_start, batch_end)]
        with mp.Pool(processes=max(1, min(workers, len(seeds))), initializer=init_worker, initargs=(case,)) as pool:
            for row in pool.imap_unordered(run_scan_one, seeds):
                all_rows.append(row)
                print(
                    f"{case} ep={len(all_rows):03d} seed={row['seed']} success={int(row['success'])} "
                    f"hits={row['hit_count']} reason={row['done_reason']} best={row['best_min_dist_m']:.2f}m",
                    flush=True,
                )
        best = choose_best_success(all_rows)
        if best is not None:
            break
        batch_start = batch_end
        if batch_start < max_eps:
            print(f"{case}: no success yet, extending scan to {min(batch_start + scan_eps, max_eps)} episodes", flush=True)

    case_dir = out_root / case
    write_scan_csv(case_dir / "scan_episodes.csv", all_rows)
    best = choose_best_success(all_rows)
    if best is None:
        closest = min(all_rows, key=lambda row: float(row["best_min_dist_m"])) if all_rows else None
        with (case_dir / "scan_summary.json").open("w") as f:
            json.dump({
                "case": case,
                "success_count": 0,
                "n_scanned": len(all_rows),
                "closest_episode": closest,
            }, f, indent=2, ensure_ascii=False)
        print(f"{case}: no successful episode found after {len(all_rows)} episodes", flush=True)
        return None

    print(f"{case}: recording best success seed={best['seed']} best={best['best_min_dist_m']:.2f}m", flush=True)
    summary = record_success(case, int(best["seed"]), case_dir)
    summary["scan_success_count"] = sum(1 for row in all_rows if row["success"])
    summary["scan_n_episodes"] = len(all_rows)
    with (case_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["4v4"])
    parser.add_argument("--scan-eps", type=int, default=int(os.environ.get("V71_6688_SCAN_EPS", "50")))
    parser.add_argument("--max-eps", type=int, default=int(os.environ.get("V71_6688_MAX_EPS", "200")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("V71_6688_WORKERS", "10")))
    parser.add_argument("--out-root", default=os.environ.get("V71_6688_OUT_ROOT", "outputs/v71_6688_success"))
    parser.add_argument("--seed-base-4v4", type=int, default=int(os.environ.get("V71_4V4_SEED_BASE", "90000")))
    parser.add_argument("--seed-base-6v6", type=int, default=int(os.environ.get("V71_6V6_SEED_BASE", "60000")))
    parser.add_argument("--seed-base-8v8", type=int, default=int(os.environ.get("V71_8V8_SEED_BASE", "80000")))
    args = parser.parse_args()

    torch.set_num_threads(1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"{stamp}_v71_4v4_clone_success"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries = {}
    for case in args.cases:
        token = case.strip().lower()
        if token == "4v4":
            seed_base = args.seed_base_4v4
        elif token == "6v6":
            seed_base = args.seed_base_6v6
        else:
            seed_base = args.seed_base_8v8
        summaries[case] = run_case(case, seed_base, args.scan_eps, args.max_eps, args.workers, out_root)
    top = {
        "created_at": stamp,
        "elapsed_s": time.time() - started,
        "out_root": str(out_root),
        "cases": summaries,
    }
    with (out_root / "summary_all.json").open("w") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    print("=== all done ===", flush=True)
    print(json.dumps(top, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
