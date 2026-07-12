#!/usr/bin/env python
"""
Evaluate 2v2 / 3v3 penetration by reusing the trained 1v1 attacker policy.

Design:
  - The environment runs as a true multi-aircraft simulation.
  - Each attacker identifies its own interceptor and builds a 1v1-compatible
    local observation (same 33-D layout as the trained single-attacker policy).
  - The same actor_agent0.pt is invoked independently for every attacker.
  - Outputs:
      * per-scenario episode summary CSV / JSON
      * selected best-episode detailed trajectory JSON
      * detailed trajectory/time-series plots
      * evaluation summary plots
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.config import G
from envs.fov_penetration.policies_interceptor import InterceptorPolicy


C_ATT = ["#1f5faa", "#2a9d8f", "#6a4c93", "#ff8c42"]
C_DEF = ["#b03030", "#e76f51", "#8d0801", "#9c6644"]
C_HVT = "#d4a017"


def load_single_policy(model_dir: str, hidden_size: int, layer_N: int, device: torch.device):
    from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
    from macpo.config import get_config

    policy_env = FOVPenetrationEnv()
    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.algorithm_name = "macpo"
    all_args.hidden_size = hidden_size
    all_args.layer_N = layer_N

    policy = MACPPOPolicy(
        all_args,
        policy_env.observation_space[0],
        policy_env.share_observation_space[0],
        policy_env.action_space[0],
        device=device,
    )
    actor_path = os.path.join(model_dir, "actor_agent0.pt")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.eval()
    return policy, policy_env


def actor_action(policy, obs: np.ndarray, hidden_size: int, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    rnn = torch.zeros(1, 1, hidden_size, device=device)
    masks = torch.ones(1, 1, device=device)
    with torch.no_grad():
        action, _, _ = policy.actor(obs_t, rnn, masks, deterministic=True)
    return action.cpu().numpy().flatten()


def make_scenario_config(n_side: int) -> Dict:
    off_spread = {2: 450.0, 3: 650.0}[n_side]
    def_spread = {2: 500.0, 3: 700.0}[n_side]
    return {
        "n_offensive": n_side,
        "n_defensive": n_side,
        "terminate_on_first_hvt_hit": False,
        "max_steps": 1000,
        "assignment": {
            "method": "hungarian",
            "reassign": False,
        },
        "offensive_init": {
            "spread_xy": off_spread,
            "pos_noise_xy": 140.0,
            "pos_noise_z": 40.0,
            "heading_noise": 0.18,
            "gamma_noise": 0.02,
        },
        "defensive_init": {
            "spread_xy": def_spread,
            "pos_noise_xy": 140.0,
            "pos_noise_z": 40.0,
            "heading_noise": 0.18,
            "gamma_noise": 0.08,
        },
    }


class SinglePolicyAdapter:
    """Build 1v1-compatible local observations for every attacker."""

    def __init__(self, ref_env: FOVPenetrationEnv):
        self.ref_cfg = ref_env.config
        self.obs_dim = ref_env.obs_dim
        self.pair_exposure_steps: Dict[int, int] = {}
        self.last_obs_step: Dict[int, int] = {}
        self.initial_reverse_assignment: Dict[int, int] = {}

    def reset(self, env: FOVPenetrationEnv):
        self.pair_exposure_steps = {i: 0 for i in range(env.n_offensive)}
        self.last_obs_step = {i: -1 for i in range(env.n_offensive)}
        self.initial_reverse_assignment = {}
        for def_idx, off_idx in env.assignments.items():
            if off_idx not in self.initial_reverse_assignment:
                self.initial_reverse_assignment[off_idx] = def_idx

    def select_interceptor(self, env: FOVPenetrationEnv, off_idx: int) -> Optional[int]:
        off = env.offensives[off_idx]
        candidates: List[int] = []
        for def_idx, policy in enumerate(env.defensive_policies):
            if not env.defensives[def_idx].alive:
                continue
            if policy.assigned_target_idx == off_idx:
                candidates.append(def_idx)
        if candidates:
            return min(candidates, key=lambda di: env.defensives[di].distance_3d(off))

        init_idx = self.initial_reverse_assignment.get(off_idx)
        if init_idx is not None and env.defensives[init_idx].alive:
            return init_idx

        alive_defs = [di for di, d in enumerate(env.defensives) if d.alive]
        if not alive_defs:
            return None
        return min(alive_defs, key=lambda di: env.defensives[di].distance_3d(off))

    def build_obs(self, env: FOVPenetrationEnv, off_idx: int, def_idx: Optional[int]) -> np.ndarray:
        cfg = self.ref_cfg
        obs_range = cfg["obs_range"]
        vel_range = cfg["vel_range"]
        z_range = cfg["z_range"]
        off = env.offensives[off_idx]

        if not off.alive or off.hit_hvt:
            return np.zeros(self.obs_dim, dtype=np.float32)

        obs: List[float] = []
        obs.extend([
            off.x / obs_range,
            off.y / obs_range,
            off.z / z_range,
            off.v / vel_range,
            off.heading / np.pi,
            off.gamma / (np.pi / 4),
            off.nx / 4.0,
            off.ny / 5.0,
            off.nz / 3.0,
        ])

        hvt = env.hvt
        obs.extend([
            (hvt.x - off.x) / obs_range,
            (hvt.y - off.y) / obs_range,
            (hvt.z - off.z) / z_range,
        ])

        pair_detected = False
        pair_def_alive = 0.0
        n_threats = 0.0
        if def_idx is not None:
            d = env.defensives[def_idx]
            if d.alive:
                pair_def_alive = 1.0
                dist = d.distance_3d(off)
                pair_detected = d.is_in_fov(off.x, off.y, off.z,
                                            env.config["fov_half_angle"],
                                            env.config["detection_range"])
                obs.extend([
                    (d.x - off.x) / obs_range,
                    (d.y - off.y) / obs_range,
                    (d.z - off.z) / z_range,
                    (d.v - off.v) / vel_range,
                    (d.heading - off.heading) / np.pi,
                    (d.gamma - off.gamma) / (np.pi / 4),
                    1.0,
                ])

                dx_me = off.x - d.x
                dy_me = off.y - d.y
                angle_to_me = np.arctan2(dy_me, dx_me)
                heading_diff = angle_to_me - d.heading
                heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                threat_heading = np.cos(heading_diff)
                obs.append(threat_heading)

                cos_gd = np.cos(d.gamma)
                vx_d = d.v * cos_gd * np.cos(d.heading)
                vy_d = d.v * cos_gd * np.sin(d.heading)
                vz_d = d.v * np.sin(d.gamma)
                cos_ga = np.cos(off.gamma)
                vx_a = off.v * cos_ga * np.cos(off.heading)
                vy_a = off.v * cos_ga * np.sin(off.heading)
                vz_a = off.v * np.sin(off.gamma)
                closing_v = -(
                    (dx_me * (vx_a - vx_d) + dy_me * (vy_a - vy_d) + (off.z - d.z) * (vz_a - vz_d))
                    / max(dist, 1.0)
                )
                obs.append(np.clip(closing_v / vel_range, -1.0, 1.0))

                sat_ratio = env.defensive_policies[def_idx].is_overload_saturated()[2]
                obs.append(np.clip(sat_ratio, 0.0, 3.0) / 3.0)
                is_engaged = 1.0 if (
                    env.defensive_policies[def_idx].engagement_state
                    == InterceptorPolicy.STATE_ENGAGED
                ) else 0.0
                obs.append(is_engaged)

                if abs(heading_diff) < np.deg2rad(30.0) and dist < 3000.0:
                    n_threats = 1.0
            else:
                obs.extend([0.0] * 11)
        else:
            obs.extend([0.0] * 11)

        if env.current_step != self.last_obs_step[off_idx]:
            if pair_detected:
                self.pair_exposure_steps[off_idx] += 1
            else:
                self.pair_exposure_steps[off_idx] = 0
            self.last_obs_step[off_idx] = env.current_step

        obs.append(1.0 if pair_detected else 0.0)
        obs.append(1.0 if pair_detected else 0.0)
        obs.append(min(self.pair_exposure_steps[off_idx] / 50.0, 1.0))

        dist_hvt = off.distance_to(hvt.x, hvt.y, hvt.z)
        obs.append(dist_hvt / obs_range)
        obs.append(env.current_step / max(env.max_steps, 1))

        obs.append(0.0)  # front_rank in 1v1
        obs.append(n_threats)
        obs.append(dist_hvt / obs_range)  # team_min_dist -> own dist in 1v1 view
        obs.append(pair_def_alive)
        obs.append(0.0)  # n_escaped_agents in 1v1 view

        return np.asarray(obs, dtype=np.float32)


def episode_record_template(env: FOVPenetrationEnv, scenario_name: str, seed: int):
    return {
        "scenario": scenario_name,
        "seed": seed,
        "dt": env.dt,
        "max_steps": env.max_steps,
        "hvt": {"x": env.hvt.x, "y": env.hvt.y, "z": env.hvt.z},
        "initial_assignments": {int(k): int(v) for k, v in env.assignments.items()},
        "attackers": {
            str(i): {
                "x": [], "y": [], "z": [], "v": [], "heading": [], "gamma": [],
                "nx": [], "ny": [], "nz": [], "alive": [], "hit_hvt": [],
                "dist_hvt": [], "assigned_defender": [], "pair_detected": [],
                "guidance_mode": [],
            }
            for i in range(env.n_offensive)
        },
        "defenders": {
            str(i): {
                "x": [], "y": [], "z": [], "v": [], "heading": [], "gamma": [],
                "alive": [], "assigned_target": [], "demanded_ny": [], "demanded_nz": [],
            }
            for i in range(env.n_defensive)
        },
        "team": {
            "hit_count": [],
            "offensive_alive": [],
            "defensive_alive": [],
        },
        "final_info": {},
    }


def append_snapshot(env: FOVPenetrationEnv, record: Dict, adapter: SinglePolicyAdapter):
    for off_idx in range(env.n_offensive):
        off = env.offensives[off_idx]
        def_idx = adapter.select_interceptor(env, off_idx)
        pair_detected = bool(
            off.alive
            and not off.hit_hvt
            and def_idx is not None
            and env.defensives[def_idx].alive
            and env.defensives[def_idx].is_in_fov(
                off.x, off.y, off.z,
                env.config["fov_half_angle"],
                env.config["detection_range"],
            )
        )

        atk = record["attackers"][str(off_idx)]
        atk["x"].append(float(off.x))
        atk["y"].append(float(off.y))
        atk["z"].append(float(off.z))
        atk["v"].append(float(off.v))
        atk["heading"].append(float(off.heading))
        atk["gamma"].append(float(off.gamma))
        atk["nx"].append(float(off.nx))
        atk["ny"].append(float(off.ny))
        atk["nz"].append(float(off.nz))
        atk["alive"].append(bool(off.alive))
        atk["hit_hvt"].append(bool(off.hit_hvt))
        atk["dist_hvt"].append(float(off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)))
        atk["assigned_defender"].append(int(def_idx) if def_idx is not None else -1)
        atk["pair_detected"].append(pair_detected)
        atk["guidance_mode"].append(int(env._off_guidance_mode[off_idx]))

    for def_idx in range(env.n_defensive):
        d = env.defensives[def_idx]
        dp = env.defensive_policies[def_idx]
        drec = record["defenders"][str(def_idx)]
        drec["x"].append(float(d.x))
        drec["y"].append(float(d.y))
        drec["z"].append(float(d.z))
        drec["v"].append(float(d.v))
        drec["heading"].append(float(d.heading))
        drec["gamma"].append(float(d.gamma))
        drec["alive"].append(bool(d.alive))
        drec["assigned_target"].append(
            int(dp.assigned_target_idx) if dp.assigned_target_idx is not None else -1
        )
        drec["demanded_ny"].append(float(dp.demanded_ny))
        drec["demanded_nz"].append(float(dp.demanded_nz))


def run_episode_multi(
    env: FOVPenetrationEnv,
    policy,
    adapter: SinglePolicyAdapter,
    hidden_size: int,
    device: torch.device,
    scenario_name: str,
    seed: int,
):
    env.seed(seed)
    obs, _, _ = env.reset()
    adapter.reset(env)
    record = episode_record_template(env, scenario_name, seed)
    append_snapshot(env, record, adapter)

    for step in range(env.max_steps):
        actions = []
        for off_idx in range(env.n_offensive):
            off = env.offensives[off_idx]
            def_idx = adapter.select_interceptor(env, off_idx)
            if off.alive and not off.hit_hvt:
                single_obs = adapter.build_obs(env, off_idx, def_idx)
                actions.append(actor_action(policy, single_obs, hidden_size, device))
            else:
                actions.append(np.zeros(3, dtype=np.float32))

        obs, _, rewards, costs, dones, infos, _ = env.step(actions)
        info = infos[0]
        record["team"]["hit_count"].append(int(info["hit_count"]))
        record["team"]["offensive_alive"].append(int(info["offensive_alive"]))
        record["team"]["defensive_alive"].append(int(info["defensive_alive"]))
        append_snapshot(env, record, adapter)

        if any(dones):
            record["final_info"] = {
                "done_reason": info.get("done_reason", "unknown"),
                "success": bool(info.get("success", False)),
                "all_attackers_hit": bool(info.get("all_attackers_hit", False)),
                "hit_count": int(info.get("hit_count", 0)),
                "hit_indices": [int(x) for x in info.get("hit_indices", [])],
                "hvt_hit_min_distance": info.get("hvt_hit_min_distance", None),
                "offensive_alive": int(info.get("offensive_alive", 0)),
                "defensive_alive": int(info.get("defensive_alive", 0)),
                "resolved_attackers": int(info.get("resolved_attackers", 0)),
                "unresolved_attackers": int(info.get("unresolved_attackers", 0)),
                "kill_events": info.get("kill_events", []),
            }
            record["n_steps"] = len(record["attackers"]["0"]["x"])
            break
    else:
        record["final_info"] = {
            "done_reason": "timeout",
            "success": False,
            "all_attackers_hit": False,
            "hit_count": int(env.hit_count),
            "hit_indices": [int(x) for x in env.hit_indices],
            "hvt_hit_min_distance": (float(min(env.hit_hvt_distances))
                                      if env.hit_hvt_distances else None),
            "offensive_alive": int(sum(1 for o in env.offensives if o.alive)),
            "defensive_alive": int(sum(1 for d in env.defensives if d.alive)),
            "resolved_attackers": int(sum(1 for o in env.offensives if (not o.alive) or o.hit_hvt)),
            "unresolved_attackers": int(sum(1 for o in env.offensives if o.alive and not o.hit_hvt)),
            "kill_events": list(env.kill_events),
        }
        record["n_steps"] = len(record["attackers"]["0"]["x"])

    return record


def summarize_episode(record: Dict) -> Dict:
    hit_count = int(record["final_info"]["hit_count"])
    n_off = len(record["attackers"])
    attacker_hit_flags = []
    attacker_min_d = []
    attacker_final_status = []
    for idx in range(n_off):
        atk = record["attackers"][str(idx)]
        hit_flag = bool(atk["hit_hvt"][-1]) if atk["hit_hvt"] else False
        alive_flag = bool(atk["alive"][-1]) if atk["alive"] else False
        attacker_hit_flags.append(int(hit_flag))
        attacker_min_d.append(float(min(atk["dist_hvt"])) if atk["dist_hvt"] else float("inf"))
        if hit_flag:
            attacker_final_status.append("hit_hvt")
        elif alive_flag:
            attacker_final_status.append("alive_timeout")
        else:
            attacker_final_status.append("killed")

    return {
        "scenario": record["scenario"],
        "seed": record["seed"],
        "done_reason": record["final_info"]["done_reason"],
        "success_any": int(record["final_info"]["success"]),
        "success_all": int(record["final_info"]["all_attackers_hit"]),
        "hit_count": hit_count,
        "mean_min_dist_hvt": float(np.mean(attacker_min_d)),
        "best_min_dist_hvt": float(np.min(attacker_min_d)),
        "offensive_alive_final": int(record["final_info"]["offensive_alive"]),
        "defensive_alive_final": int(record["final_info"]["defensive_alive"]),
        "attacker_hit_flags": attacker_hit_flags,
        "attacker_final_status": attacker_final_status,
    }


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_plot_multi_format(fig, base_path_no_ext: str, dpi: int = 180):
    fig.savefig(base_path_no_ext + ".png", dpi=dpi)
    fig.savefig(base_path_no_ext + ".pdf")


def write_summary_csv(path: str, rows: List[Dict], n_offensive: int):
    fieldnames = [
        "scenario", "seed", "done_reason", "success_any", "success_all", "hit_count",
        "mean_min_dist_hvt", "best_min_dist_hvt", "offensive_alive_final", "defensive_alive_final",
    ]
    for i in range(n_offensive):
        fieldnames.append(f"attacker{i}_hit")
        fieldnames.append(f"attacker{i}_status")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            for i in range(n_offensive):
                flat[f"attacker{i}_hit"] = row["attacker_hit_flags"][i]
                flat[f"attacker{i}_status"] = row["attacker_final_status"][i]
            writer.writerow(flat)


def plot_detailed_episode(record: Dict, out_path: str):
    n_steps = record["n_steps"]
    t = np.arange(n_steps) * record["dt"]
    n_off = len(record["attackers"])
    n_def = len(record["defenders"])

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    ax_xy, ax_hvt, ax_pair, ax_mode, ax_alt, ax_alive = axes.flatten()

    # XY trajectory
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        ax_xy.plot(atk["x"][:n_steps], atk["y"][:n_steps], color=C_ATT[i], lw=2.0, label=f"Att{i}")
        ax_xy.scatter(atk["x"][0], atk["y"][0], color=C_ATT[i], marker="o", s=35)
        ax_xy.scatter(atk["x"][n_steps - 1], atk["y"][n_steps - 1], color=C_ATT[i], marker="x", s=55)
    for j in range(n_def):
        drec = record["defenders"][str(j)]
        ax_xy.plot(drec["x"][:n_steps], drec["y"][:n_steps], color=C_DEF[j], lw=1.6, ls="--", label=f"Def{j}")
    ax_xy.scatter(record["hvt"]["x"], record["hvt"]["y"], color=C_HVT, marker="*", s=260, edgecolors="k", label="HVT")
    ax_xy.set_title("Top-Down Trajectories")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(ncol=2, fontsize=8)

    # Distance to HVT
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        ax_hvt.plot(t, atk["dist_hvt"][:n_steps], color=C_ATT[i], lw=1.8, label=f"Att{i}")
    ax_hvt.axhline(5.0, color="orange", ls=":", lw=1.0, label="HVT hit 5m")
    ax_hvt.set_title("Distance to HVT")
    ax_hvt.set_xlabel("Time (s)")
    ax_hvt.set_ylabel("Distance (m)")
    ax_hvt.grid(True, alpha=0.25)
    ax_hvt.legend(fontsize=8)

    # Pair distance
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        pair_dist = []
        for k in range(n_steps):
            def_idx = atk["assigned_defender"][k]
            if def_idx < 0:
                pair_dist.append(np.nan)
                continue
            drec = record["defenders"][str(def_idx)]
            dx = atk["x"][k] - drec["x"][k]
            dy = atk["y"][k] - drec["y"][k]
            dz = atk["z"][k] - drec["z"][k]
            pair_dist.append(float(np.sqrt(dx * dx + dy * dy + dz * dz)))
        ax_pair.plot(t, pair_dist, color=C_ATT[i], lw=1.8, label=f"Att{i}-Def")
    ax_pair.axhline(5.0, color="red", ls=":", lw=1.0, label="Intercept 5m")
    ax_pair.set_title("Assigned Pair Distance")
    ax_pair.set_xlabel("Time (s)")
    ax_pair.set_ylabel("Distance (m)")
    ax_pair.grid(True, alpha=0.25)
    ax_pair.legend(fontsize=8)

    # Guidance mode
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        ax_mode.step(t, atk["guidance_mode"][:n_steps], where="post", color=C_ATT[i], lw=1.8, label=f"Att{i}")
    ax_mode.set_title("Attacker Guidance Mode (0=RL, 1=Terminal PN)")
    ax_mode.set_xlabel("Time (s)")
    ax_mode.set_ylabel("Mode")
    ax_mode.set_yticks([0, 1])
    ax_mode.grid(True, alpha=0.25)
    ax_mode.legend(fontsize=8)

    # Altitude
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        ax_alt.plot(t, atk["z"][:n_steps], color=C_ATT[i], lw=1.8, label=f"Att{i}")
    for j in range(n_def):
        drec = record["defenders"][str(j)]
        ax_alt.plot(t, drec["z"][:n_steps], color=C_DEF[j], lw=1.3, ls="--", label=f"Def{j}")
    ax_alt.set_title("Altitude Evolution")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, alpha=0.25)
    ax_alt.legend(ncol=2, fontsize=8)

    # Alive / hit state
    for i in range(n_off):
        atk = record["attackers"][str(i)]
        state = []
        for k in range(n_steps):
            if atk["hit_hvt"][k]:
                state.append(2.0)
            elif atk["alive"][k]:
                state.append(1.0)
            else:
                state.append(0.0)
        ax_alive.step(t, state, where="post", color=C_ATT[i], lw=1.8, label=f"Att{i}")
    ax_alive.set_title("Attacker State (0=dead, 1=alive, 2=hit HVT)")
    ax_alive.set_xlabel("Time (s)")
    ax_alive.set_ylabel("State")
    ax_alive.set_yticks([0, 1, 2])
    ax_alive.grid(True, alpha=0.25)
    ax_alive.legend(fontsize=8)

    fig.suptitle(
        f"{record['scenario']} Best Episode | seed={record['seed']} | "
        f"done={record['final_info']['done_reason']} | hits={record['final_info']['hit_count']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot_multi_format(fig, os.path.splitext(out_path)[0], dpi=180)
    plt.close(fig)


def plot_summary(rows: List[Dict], scenario_name: str, n_offensive: int, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_hits, ax_reason, ax_agent, ax_min = axes.flatten()

    ep_ids = np.arange(len(rows))
    hit_counts = [row["hit_count"] for row in rows]
    colors = ["#2a9d8f" if row["success_all"] else "#f4a261" if row["success_any"] else "#d62828" for row in rows]
    ax_hits.bar(ep_ids, hit_counts, color=colors, edgecolor="black", alpha=0.85)
    ax_hits.set_title("Hit Count per Episode")
    ax_hits.set_xlabel("Episode")
    ax_hits.set_ylabel("HVT Hits")
    ax_hits.set_ylim(0, n_offensive + 0.5)

    reason_counts: Dict[str, int] = {}
    for row in rows:
        reason_counts[row["done_reason"]] = reason_counts.get(row["done_reason"], 0) + 1
    ax_reason.bar(list(reason_counts.keys()), list(reason_counts.values()),
                  color="#6c8ebf", edgecolor="black", alpha=0.85)
    ax_reason.set_title("Done Reason Distribution")
    ax_reason.set_ylabel("Count")
    ax_reason.tick_params(axis="x", rotation=20)

    per_att_hits = np.zeros(n_offensive, dtype=np.float64)
    for row in rows:
        per_att_hits += np.asarray(row["attacker_hit_flags"], dtype=np.float64)
    per_att_rates = per_att_hits / max(len(rows), 1)
    ax_agent.bar(np.arange(n_offensive), per_att_rates, color=C_ATT[:n_offensive], edgecolor="black", alpha=0.85)
    ax_agent.set_title("Per-Attacker Hit Rate")
    ax_agent.set_xlabel("Attacker Index")
    ax_agent.set_ylabel("Hit Rate")
    ax_agent.set_ylim(0, 1.05)

    best_min = [row["best_min_dist_hvt"] for row in rows]
    mean_min = [row["mean_min_dist_hvt"] for row in rows]
    ax_min.plot(ep_ids, best_min, "o-", color="#6a4c93", label="Best attacker min dist")
    ax_min.plot(ep_ids, mean_min, "s--", color="#ff8c42", label="Mean attacker min dist")
    ax_min.axhline(5.0, color="orange", ls=":", lw=1.0, label="HVT hit 5m")
    ax_min.set_title("HVT Distance Statistics")
    ax_min.set_xlabel("Episode")
    ax_min.set_ylabel("Distance (m)")
    ax_min.grid(True, alpha=0.25)
    ax_min.legend(fontsize=8)

    team_all_hit_rate = np.mean([row["success_all"] for row in rows]) if rows else 0.0
    team_any_hit_rate = np.mean([row["success_any"] for row in rows]) if rows else 0.0
    fig.suptitle(
        f"{scenario_name} Summary | any-hit={team_any_hit_rate:.1%}, all-hit={team_all_hit_rate:.1%}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot_multi_format(fig, os.path.splitext(out_path)[0], dpi=180)
    plt.close(fig)


def choose_best_episode(rows: List[Dict]) -> int:
    best_idx = 0
    best_key = None
    for i, row in enumerate(rows):
        key = (
            int(row["success_all"]),
            int(row["hit_count"]),
            -float(row["mean_min_dist_hvt"]),
        )
        if best_key is None or key > best_key:
            best_idx = i
            best_key = key
    return best_idx


def _obj_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _first_hit_step(best_record: Dict) -> Tuple[int, int]:
    for idx, atk in best_record["attackers"].items():
        hits = atk.get("hit_hvt", [])
        for s, flag in enumerate(hits):
            if flag:
                return int(idx), int(s)
    return -1, -1


def save_best_episode_npz(best_record: Dict, summary_json: Dict, scenario_dir: str):
    n_off = len(best_record["attackers"])
    n_def = len(best_record["defenders"])
    n_steps = int(best_record.get("n_steps", 0))
    dt = float(best_record.get("dt", 0.1))

    t = np.arange(n_steps, dtype=np.float32) * dt

    def pad_agent_series(series_list, fill=0.0, dtype=np.float32):
        out = np.full((4, n_steps), fill, dtype=dtype)
        for i, seq in enumerate(series_list[:4]):
            seq = np.asarray(seq)
            L = min(len(seq), n_steps)
            if L:
                out[i, :L] = seq[:L].astype(dtype, copy=False)
        return out

    # attacker per-step series
    off_x = pad_agent_series([best_record["attackers"][str(i)]["x"] for i in range(n_off)], fill=np.nan)
    off_y = pad_agent_series([best_record["attackers"][str(i)]["y"] for i in range(n_off)], fill=np.nan)
    off_z = pad_agent_series([best_record["attackers"][str(i)]["z"] for i in range(n_off)], fill=np.nan)
    off_v = pad_agent_series([best_record["attackers"][str(i)]["v"] for i in range(n_off)], fill=np.nan)
    off_heading = pad_agent_series([best_record["attackers"][str(i)]["heading"] for i in range(n_off)], fill=np.nan)
    off_gamma = pad_agent_series([best_record["attackers"][str(i)]["gamma"] for i in range(n_off)], fill=np.nan)
    off_nx = pad_agent_series([best_record["attackers"][str(i)]["nx"] for i in range(n_off)], fill=np.nan)
    off_ny = pad_agent_series([best_record["attackers"][str(i)]["ny"] for i in range(n_off)], fill=np.nan)
    off_nz = pad_agent_series([best_record["attackers"][str(i)]["nz"] for i in range(n_off)], fill=np.nan)
    off_an_pitch = off_ny * float(G)
    off_an_yaw = off_nz * float(G)
    off_alive = pad_agent_series([best_record["attackers"][str(i)]["alive"] for i in range(n_off)], fill=0, dtype=np.int8)
    off_hit = pad_agent_series([best_record["attackers"][str(i)]["hit_hvt"] for i in range(n_off)], fill=0, dtype=np.int8)
    off_d_hvt = pad_agent_series([best_record["attackers"][str(i)]["dist_hvt"] for i in range(n_off)], fill=np.nan)

    # count defender locks per attacker to build off_lbc
    off_lbc = np.zeros((4, n_steps), dtype=np.float32)
    for s in range(n_steps):
        counts = {i: 0 for i in range(4)}
        for j in range(n_def):
            drec = best_record["defenders"][str(j)]
            tgt_seq = drec.get("assigned_target", [])
            tgt = int(tgt_seq[s]) if s < len(tgt_seq) else -1
            if 0 <= tgt < 4:
                counts[tgt] += 1
        for i in range(4):
            off_lbc[i, s] = counts[i]

    # defender per-step series
    def_x = pad_agent_series([best_record["defenders"][str(j)]["x"] for j in range(n_def)], fill=np.nan)
    def_y = pad_agent_series([best_record["defenders"][str(j)]["y"] for j in range(n_def)], fill=np.nan)
    def_z = pad_agent_series([best_record["defenders"][str(j)]["z"] for j in range(n_def)], fill=np.nan)
    def_v = pad_agent_series([best_record["defenders"][str(j)]["v"] for j in range(n_def)], fill=np.nan)
    def_heading = pad_agent_series([best_record["defenders"][str(j)].get("heading", []) for j in range(n_def)], fill=np.nan)
    def_gamma = pad_agent_series([best_record["defenders"][str(j)].get("gamma", []) for j in range(n_def)], fill=np.nan)
    def_alive = pad_agent_series([best_record["defenders"][str(j)]["alive"] for j in range(n_def)], fill=0, dtype=np.int8)
    def_assigned_target = pad_agent_series([best_record["defenders"][str(j)]["assigned_target"] for j in range(n_def)], fill=-1, dtype=np.int32)
    def_lmode = (def_assigned_target >= 0).astype(np.int8)
    def_initial_target = np.full((4,), -1, dtype=np.int32)
    for j in range(n_def):
        seq = best_record["defenders"][str(j)].get("assigned_target", [])
        for v in seq:
            if int(v) >= 0:
                def_initial_target[j] = int(v)
                break
    def_ltgt = def_assigned_target.copy()
    def_an_pitch = np.full((4, n_steps), np.nan, dtype=np.float32)
    def_an_yaw = np.full((4, n_steps), np.nan, dtype=np.float32)
    for j in range(n_def):
        dn_y = np.asarray(best_record["defenders"][str(j)].get("demanded_ny", []), dtype=np.float32)
        dn_z = np.asarray(best_record["defenders"][str(j)].get("demanded_nz", []), dtype=np.float32)
        if dn_y.size:
            def_an_pitch[j, :min(dn_y.size, n_steps)] = dn_y[:n_steps] * float(G)
        if dn_z.size:
            def_an_yaw[j, :min(dn_z.size, n_steps)] = dn_z[:n_steps] * float(G)

    # assignment cost and saturation
    assign_cost = np.zeros((n_steps, 4, 4), dtype=np.float32)
    fov_sat = np.zeros((n_steps,), dtype=np.float32)
    for s in range(n_steps):
        for j in range(n_def):
            dx = def_x[j, s]
            dy = def_y[j, s]
            dz = def_z[j, s]
            if not np.isfinite(dx):
                continue
            for i in range(n_off):
                ox = off_x[i, s]
                oy = off_y[i, s]
                oz = off_z[i, s]
                if np.isfinite(ox):
                    assign_cost[s, j, i] = float(np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2 + (dz - oz) ** 2))
        fov_sat[s] = float(np.sum(def_lmode[:n_def, s] > 0) / max(n_def, 1))

    # game metrics (best-effort, but saved at simulation time)
    # Use actual per-step values if available; otherwise fill with zeros.
    decoy_Phi = np.zeros((n_steps,), dtype=np.float32)
    decoy_role_decoy = np.zeros((4, n_steps), dtype=np.float32)
    decoy_role_pen = np.zeros((4, n_steps), dtype=np.float32)
    decoy_role_stealth = np.zeros((4, n_steps), dtype=np.float32)
    decoy_lock_pressure = off_lbc.copy()
    pen_N_eff = np.zeros((n_steps,), dtype=np.float32)
    pen_P_pen = np.zeros((4, n_steps), dtype=np.float32)
    esc_Gamma_mean = np.zeros((n_steps,), dtype=np.float32)
    esc_Xi_mean = np.zeros((n_steps,), dtype=np.float32)
    esc_E_esc = np.zeros((4, n_steps), dtype=np.float32)
    hvt_P_hit = off_hit.astype(np.float32)
    hvt_rho = off_d_hvt.astype(np.float32)
    hvt_closing = np.zeros((4, n_steps), dtype=np.float32)
    if n_steps > 1:
        hvt_closing[:, 1:] = np.maximum(-(off_d_hvt[:, 1:] - off_d_hvt[:, :-1]) / dt, 0.0)

    hit_count = int(best_record.get("final_info", {}).get("hit_count", 0))
    hvt_x = float(best_record.get("hvt", {}).get("x", 0.0))
    hvt_y = float(best_record.get("hvt", {}).get("y", 0.0))
    hvt_z = float(best_record.get("hvt", {}).get("z", 0.0))

    hitter, hit_step = _first_hit_step(best_record)
    summary_json.update({
        "hitter": int(hitter),
        "hit_step": int(hit_step),
        "hit_time_s": float(hit_step * dt) if hit_step >= 0 else None,
        "best_hvt_distance_m": float(np.nanmin(off_d_hvt[hitter])) if hitter >= 0 else None,
    })

    traj = {
        "steps": np.arange(n_steps, dtype=np.int32),
        "time": t,
        "off_x": off_x, "off_y": off_y, "off_z": off_z, "off_v": off_v,
        "off_heading": off_heading, "off_gamma": off_gamma,
        "off_an_pitch": off_an_pitch, "off_an_yaw": off_an_yaw,
        "off_lbc": off_lbc, "off_alive": off_alive, "off_hit": off_hit,
        "off_d_hvt": off_d_hvt,
        "def_x": def_x, "def_y": def_y, "def_z": def_z, "def_v": def_v,
        "def_heading": def_heading, "def_gamma": def_gamma,
        "def_an_pitch": def_an_pitch, "def_an_yaw": def_an_yaw,
        "def_lmode": def_lmode, "def_initial_target": def_initial_target,
        "def_assigned_target": def_assigned_target, "def_ltgt": def_ltgt,
        "def_alive": def_alive,
        "assign_cost": assign_cost,
        "fov_sat": fov_sat,
        "hvt_x": hvt_x, "hvt_y": hvt_y, "hvt_z": hvt_z,
        "hit_count": hit_count,
    }
    np.savez_compressed(os.path.join(scenario_dir, "trajectory_data.npz"), **traj)

    game = {
        "decoy_Phi": decoy_Phi,
        "decoy_role_decoy": decoy_role_decoy,
        "decoy_role_pen": decoy_role_pen,
        "decoy_role_stealth": decoy_role_stealth,
        "decoy_lock_pressure": decoy_lock_pressure,
        "pen_N_eff": pen_N_eff,
        "pen_P_pen": pen_P_pen,
        "esc_Gamma_mean": esc_Gamma_mean,
        "esc_Xi_mean": esc_Xi_mean,
        "esc_E_esc": esc_E_esc,
        "hvt_P_hit": hvt_P_hit,
        "hvt_rho": hvt_rho,
        "hvt_closing": hvt_closing,
        "def_lmode": def_lmode,
        "def_ltgt": def_ltgt,
    }
    np.savez_compressed(os.path.join(scenario_dir, "game_data.npz"), **game)


def evaluate_scenario(
    scenario_name: str,
    n_side: int,
    policy,
    ref_env: FOVPenetrationEnv,
    hidden_size: int,
    device: torch.device,
    n_episodes: int,
    seed_start: int,
    output_root: str,
):
    scenario_dir = os.path.join(output_root, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    details_dir = os.path.join(scenario_dir, "episode_details")
    os.makedirs(details_dir, exist_ok=True)

    env = FOVPenetrationEnv(config=make_scenario_config(n_side))
    adapter = SinglePolicyAdapter(ref_env)

    episode_records = []
    summary_rows = []
    print(f"\n[eval] {scenario_name}: {n_side}v{n_side}, episodes={n_episodes}")
    for ep in range(n_episodes):
        seed = seed_start + ep
        record = run_episode_multi(
            env, policy, adapter, hidden_size, device, scenario_name, seed
        )
        summary = summarize_episode(record)
        episode_records.append(record)
        summary_rows.append(summary)
        detail_name = f"ep{ep:02d}_seed{seed}_detail.json"
        save_json(os.path.join(details_dir, detail_name), record)
        print(
            f"  ep={ep:02d} seed={seed} done={summary['done_reason']:<16} "
            f"hits={summary['hit_count']}/{n_side} "
            f"all_hit={bool(summary['success_all'])} "
            f"best_min={summary['best_min_dist_hvt']:.1f}m"
        )

    best_idx = choose_best_episode(summary_rows)
    best_record = episode_records[best_idx]

    summary_json = {
        "scenario": scenario_name,
        "n_offensive": n_side,
        "n_defensive": n_side,
        "episodes": summary_rows,
        "team_any_hit_rate": float(np.mean([row["success_any"] for row in summary_rows])) if summary_rows else 0.0,
        "team_all_hit_rate": float(np.mean([row["success_all"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_hit_count": float(np.mean([row["hit_count"] for row in summary_rows])) if summary_rows else 0.0,
        "best_episode_seed": best_record["seed"],
        "best_episode_done_reason": best_record["final_info"]["done_reason"],
        "best_episode_hit_count": best_record["final_info"]["hit_count"],
    }

    hitter, hit_step = _first_hit_step(best_record)
    summary_json["hitter"] = int(hitter)
    summary_json["hit_step"] = int(hit_step)
    summary_json["hit_time_s"] = float(hit_step * best_record["dt"]) if hit_step >= 0 else None
    summary_json["best_hvt_distance_m"] = float(np.nanmin([np.min(best_record["attackers"][str(i)]["dist_hvt"]) for i in range(n_side)])) if n_side > 0 else None
    if hitter >= 0:
        max_lbc = 0
        hit_len = len(best_record["attackers"][str(hitter)]["assigned_defender"])
        for s in range(hit_len):
            max_lbc = max(
                max_lbc,
                sum(
                    1 for j in range(len(best_record["defenders"]))
                    if s < len(best_record["defenders"][str(j)]["assigned_target"])
                    and int(best_record["defenders"][str(j)]["assigned_target"][s]) == hitter
                )
            )
        summary_json["hitter_locked_by_count_max"] = int(max_lbc)
    else:
        summary_json["hitter_locked_by_count_max"] = None
    write_summary_csv(os.path.join(scenario_dir, "episode_summary.csv"), summary_rows, n_side)
    save_json(os.path.join(scenario_dir, "summary.json"), summary_json)
    save_json(os.path.join(scenario_dir, "best_episode_detail.json"), best_record)
    save_best_episode_npz(best_record, summary_json, scenario_dir)
    plot_detailed_episode(best_record, os.path.join(scenario_dir, "best_episode_detail.png"))
    plot_summary(summary_rows, scenario_name, n_side, os.path.join(scenario_dir, "summary_plot.png"))
    print(f"  [saved] {scenario_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/results/fov_penetration/macpo/v22_1v1_penetration/run1/models",
    )
    parser.add_argument("--output", type=str, default="outputs/multi_single_policy_eval")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=8)
    parser.add_argument("--seed_start", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cpu")
    policy, ref_env = load_single_policy(args.model_dir, args.hidden_size, args.layer_N, device)
    print(f"[eval] Loaded single-attacker actor from {args.model_dir}")

    evaluate_scenario(
        scenario_name="scenario_2v2_single_policy",
        n_side=2,
        policy=policy,
        ref_env=ref_env,
        hidden_size=args.hidden_size,
        device=device,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        output_root=args.output,
    )
    evaluate_scenario(
        scenario_name="scenario_3v3_single_policy",
        n_side=3,
        policy=policy,
        ref_env=ref_env,
        hidden_size=args.hidden_size,
        device=device,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start + 100,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()
