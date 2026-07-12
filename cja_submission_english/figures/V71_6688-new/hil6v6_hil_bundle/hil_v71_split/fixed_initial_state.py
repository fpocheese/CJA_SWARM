#!/usr/bin/env python3
"""Load and apply fixed V71 initial states from saved trajectory_data.npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.fov_penetration.analytic_priors.assignment_mismatch import compute_initial_assignment
from envs.fov_penetration.analytic_priors.hvt_guidance import compute_hvt_guidance_features


def _scalar(data, key: str, default: float = 0.0) -> float:
    if key not in data.files:
        return float(default)
    return float(np.asarray(data[key]).reshape(-1)[0])


def load_initial_state(npz_path: str | Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "time" in data.files and len(np.asarray(data["time"]).reshape(-1)) > 1:
        dt = float(np.asarray(data["time"]).reshape(-1)[1] - np.asarray(data["time"]).reshape(-1)[0])
    else:
        dt = 0.01
    dt = max(dt, 1e-6)
    state = {
        "hvt": (
            _scalar(data, "hvt_x", 1200.0),
            _scalar(data, "hvt_y", 0.0),
            _scalar(data, "hvt_z", 0.0),
        ),
        "off": [],
        "def": [],
    }
    n_off = int(np.asarray(data["off_x"]).shape[0])
    n_def = int(np.asarray(data["def_x"]).shape[0])
    def_has_two_samples = np.asarray(data["def_x"]).shape[1] > 1
    for i in range(n_off):
        state["off"].append({
            "x": float(data["off_x"][i, 0]),
            "y": float(data["off_y"][i, 0]),
            "z": float(data["off_z"][i, 0]),
            "v": float(data["off_v"][i, 0]),
            "heading": float(data["off_heading"][i, 0]),
            "gamma": float(data["off_gamma"][i, 0]),
        })
    for i in range(n_def):
        heading = float(data["def_heading"][i, 0]) if "def_heading" in data.files else None
        gamma = float(data["def_gamma"][i, 0]) if "def_gamma" in data.files else None
        if (heading is None or gamma is None) and def_has_two_samples:
            dx = float(data["def_x"][i, 1] - data["def_x"][i, 0])
            dy = float(data["def_y"][i, 1] - data["def_y"][i, 0])
            dz = float(data["def_z"][i, 1] - data["def_z"][i, 0])
            heading = float(np.arctan2(dy, dx))
            gamma = float(np.arctan2(dz, max(np.hypot(dx, dy), dt)))
        state["def"].append({
            "x": float(data["def_x"][i, 0]),
            "y": float(data["def_y"][i, 0]),
            "z": float(data["def_z"][i, 0]),
            "v": float(data["def_v"][i, 0]),
            "heading": heading,
            "gamma": gamma,
            "initial_target": int(data["def_initial_target"][i, 0]) if "def_initial_target" in data.files else None,
            "assigned_target": int(data["def_assigned_target"][i, 0]) if "def_assigned_target" in data.files else None,
            "lock_mode": int(data["def_lmode"][i, 0]) if "def_lmode" in data.files else None,
            "locked_target": int(data["def_ltgt"][i, 0]) if "def_ltgt" in data.files else None,
        })
    return state


def _estimate_defender_attitude(raw_env, def_state: dict, target_idx: int | None) -> tuple[float, float]:
    if target_idx is not None and 0 <= target_idx < len(raw_env.offensives):
        target = raw_env.offensives[target_idx]
        dx = float(target.x - def_state["x"])
        dy = float(target.y - def_state["y"])
        dz = float(target.z - def_state["z"])
    else:
        dx = float(raw_env.hvt.x - def_state["x"])
        dy = float(raw_env.hvt.y - def_state["y"])
        dz = float(raw_env.hvt.z - def_state["z"])
    heading = float(np.arctan2(dy, dx))
    gamma = float(np.arctan2(dz, max(np.hypot(dx, dy), 1.0)))
    return heading, gamma


def _refresh_episode_caches(raw_env) -> None:
    raw_env._run_target_assignment()
    raw_env._update_lock_on_map()
    raw_env.prev_dists_to_hvt = [
        off.distance_to(raw_env.hvt.x, raw_env.hvt.y, raw_env.hvt.z)
        for off in raw_env.offensives
    ]
    raw_env.initial_dists_to_hvt = [max(d, 1.0) for d in raw_env.prev_dists_to_hvt]
    raw_env.prev_team_min_dist = min(raw_env.prev_dists_to_hvt)
    raw_env.hit_count = 0
    raw_env.hit_indices = []
    raw_env.kill_events = []
    raw_env.escape_events_total = []
    raw_env.engagement_tracking = {}
    raw_env.miss_cooldowns = {}
    raw_env.lock_events_log = []

    if getattr(raw_env, "_ap_enabled", False):
        raw_env._ap_prev_q_matrix = None
        if raw_env.ap_config.get("enable_assignment_mismatch_reward", False):
            raw_env._ap_fixed_assignment = compute_initial_assignment(
                raw_env.offensives,
                raw_env.defensives,
                raw_env.config["fov_half_angle"],
                raw_env.ap_config,
            )
        else:
            raw_env._ap_fixed_assignment = {}
        raw_env._ap_prev_M_tilde = None
        raw_env._ap_prev_hvt_omega = np.zeros(raw_env.n_offensive, dtype=np.float32)
        raw_env._ap_prev_Phi_decoy = None
        raw_env._ap_prev_N_eff = None

    raw_env._ap_decoy_info = {}
    raw_env._ap_pen_info = {}
    raw_env._ap_esc_info = {}
    raw_env._ap_hvt_info = {}
    raw_env._ap_Z_matrix = np.zeros((raw_env.n_offensive, raw_env.n_defensive), dtype=np.float32)
    raw_env._ap_Z_tilde = np.zeros(raw_env.n_offensive, dtype=np.float32)
    raw_env._ap_psi_agg = np.zeros(raw_env.n_offensive, dtype=np.float32)
    raw_env._ap_Gamma_matrix = [[0.0] * raw_env.n_defensive for _ in range(raw_env.n_offensive)]
    raw_env._ap_Xi_matrix = [[0.0] * raw_env.n_defensive for _ in range(raw_env.n_offensive)]
    raw_env._ap_prev_E_esc = None

    if getattr(raw_env, "_ap_enabled", False):
        ap = raw_env.config.get("analytic_priors", {})
        pn_nav_gain = ap.get("pn_nav_gain", 3.0)
        rho_list, closing_list, omega_list = [], [], []
        omega_dot_list, pn_hint_list, p_hit_list = [], [], []
        for i, off in enumerate(raw_env.offensives):
            if not off.alive or off.hit_hvt:
                rho_list.append(0.0)
                closing_list.append(0.0)
                omega_list.append(0.0)
                omega_dot_list.append(0.0)
                pn_hint_list.append(0.0)
                p_hit_list.append(0.0)
                continue
            feats = compute_hvt_guidance_features(
                off,
                raw_env.hvt,
                raw_env.dt,
                prev_omega_los=raw_env._ap_prev_hvt_omega[i],
                pn_nav_gain=pn_nav_gain,
            )
            raw_env._ap_prev_hvt_omega[i] = feats["omega_los"]
            rho_list.append(feats["rho"])
            closing_list.append(feats["closing_speed"])
            omega_list.append(feats["omega_los"])
            omega_dot_list.append(feats["omega_los_dot"])
            pn_hint_list.append(feats["pn_hint"])
            p_hit_list.append(0.0)
        raw_env._ap_hvt_info = {
            "rho_per_agent": rho_list,
            "closing_per_agent": closing_list,
            "omega_per_agent": omega_list,
            "omega_dot_per_agent": omega_dot_list,
            "pn_hint_per_agent": pn_hint_list,
            "P_hit_per_agent": p_hit_list,
        }


def apply_initial_state(raw_env, state: dict) -> None:
    hx, hy, hz = state["hvt"]
    raw_env.hvt.x = float(hx)
    raw_env.hvt.y = float(hy)
    raw_env.hvt.z = float(hz)

    if len(state["off"]) != raw_env.n_offensive:
        raise ValueError(f"offensive count mismatch: state={len(state['off'])}, env={raw_env.n_offensive}")
    if len(state["def"]) != raw_env.n_defensive:
        raise ValueError(f"defensive count mismatch: state={len(state['def'])}, env={raw_env.n_defensive}")

    for aircraft, item in zip(raw_env.offensives, state["off"]):
        aircraft.reset(item["x"], item["y"], item["z"], item["v"], item["heading"], item["gamma"])

    for idx, (aircraft, item) in enumerate(zip(raw_env.defensives, state["def"])):
        target_idx = item.get("initial_target")
        heading = item.get("heading")
        gamma = item.get("gamma")
        if heading is None or gamma is None:
            heading, gamma = _estimate_defender_attitude(raw_env, item, target_idx)
        aircraft.reset(item["x"], item["y"], item["z"], item["v"], heading, gamma)
        if idx < len(raw_env.defensive_policies) and target_idx is not None:
            policy = raw_env.defensive_policies[idx]
            if hasattr(policy, "set_initial_target"):
                target = raw_env.offensives[int(target_idx)] if 0 <= int(target_idx) < len(raw_env.offensives) else None
                try:
                    policy.set_initial_target(int(target_idx), target)
                except TypeError:
                    policy.set_initial_target(int(target_idx), raw_env.offensives, raw_env.defensives, raw_env.hvt)
            locked_target = item.get("locked_target")
            lock_mode = item.get("lock_mode")
            if lock_mode is not None and hasattr(policy, "lock_mode"):
                policy.lock_mode = int(lock_mode)
                if hasattr(policy, "engagement_state"):
                    policy.engagement_state = int(lock_mode)
            if locked_target is not None and 0 <= int(locked_target) < len(raw_env.offensives):
                target = raw_env.offensives[int(locked_target)]
                if getattr(policy, "lock_mode", None) == getattr(policy, "STATE_LOCKED", 2):
                    policy.current_locked_target_idx = int(locked_target)
                    policy.target = target
                    policy.has_ever_locked = True
                    if hasattr(policy, "_update_known_position"):
                        policy._update_known_position(target)

    raw_env.current_step = 0
    _refresh_episode_caches(raw_env)


def reset_with_fixed_initial(raw_env, wrapped_env, npz_path: str | Path, seed: int = 0):
    wrapped_env.seed(seed)
    wrapped_env.reset()
    state = load_initial_state(npz_path)
    apply_initial_state(raw_env, state)
    raw_obs = raw_env._get_obs()
    share_obs = raw_env._get_share_obs()
    avail = raw_env._get_avail_actions()

    phase_env = getattr(wrapped_env, "env", None)
    if hasattr(phase_env, "_mask_obs"):
        obs = phase_env._mask_obs(raw_obs)
    elif hasattr(wrapped_env, "_mask_obs"):
        obs = wrapped_env._mask_obs(raw_obs)
    else:
        obs = raw_obs
    if hasattr(wrapped_env, "_last_obs"):
        wrapped_env._last_obs = wrapped_env._copy_obs(obs)
    return obs, share_obs, avail
