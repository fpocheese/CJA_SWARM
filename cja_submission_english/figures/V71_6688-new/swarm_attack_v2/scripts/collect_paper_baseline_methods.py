#!/usr/bin/env python
"""Paper-baseline penetration experiments for V71_6688 comparison.

This script is intentionally standalone. It imports the existing simulator and
recording helpers, but it does not modify the original environment, policies,
or training/evaluation entry points.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.fov_penetration import FOVPenetrationEnv
from scripts.collect_v71_6688_success import (
    append_game,
    append_trajectory,
    finalize_npz_dict,
    init_record,
)


G = 9.80665
CASES = {
    "caseB_seed50042_torch1": {"n": 4, "seed": 50042},
    "6v6": {"n": 6, "seed": 60015},
    "8v8": {"n": 8, "seed": 80047},
}


def wrap_pi(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def norm2(v: np.ndarray, eps: float = 1e-9) -> float:
    return max(float(np.linalg.norm(v)), eps)


def action_from_point(off, point: np.ndarray, speed_bias: float = 0.0,
                      yaw_gain: float = 2.3, pitch_gain: float = 2.0,
                      gamma_limit_deg: float = 16.0) -> np.ndarray:
    """Convert a desired 3-D aimpoint to the simulator's normalized action."""
    dx = float(point[0] - off.x)
    dy = float(point[1] - off.y)
    dz = float(point[2] - off.z)
    horiz = max(math.hypot(dx, dy), 1.0)
    psi_des = math.atan2(dy, dx)
    gamma_des = math.atan2(dz, horiz)
    lim = math.radians(gamma_limit_deg)
    gamma_des = float(np.clip(gamma_des, -lim, lim))

    yaw_err = wrap_pi(psi_des - float(off.heading))
    pitch_err = wrap_pi(gamma_des - float(off.gamma))

    params = off.params
    an_yaw_max = float(params.get("an_yaw_max", 5.0 * G))
    an_pitch_max = float(params.get("an_pitch_max", 2.5 * G))
    ax_min = float(params.get("ax_min", -8.0))
    ax_max = float(params.get("ax_max", 8.0))

    an_yaw_cmd = yaw_gain * float(off.v) * yaw_err
    an_pitch_cmd = G * math.cos(float(off.gamma)) + pitch_gain * float(off.v) * pitch_err

    trim = G
    if an_pitch_cmd >= trim:
        a_pitch = (an_pitch_cmd - trim) / max(an_pitch_max - trim, 1e-6)
    else:
        a_pitch = (an_pitch_cmd - trim) / max(an_pitch_max + trim, 1e-6)

    ax_cmd = float(np.clip(speed_bias, ax_min, ax_max))
    a_ax = ax_cmd / ax_max if ax_cmd >= 0.0 else ax_cmd / max(-ax_min, 1e-6)
    return np.asarray([
        np.clip(a_ax, -1.0, 1.0),
        np.clip(a_pitch, -0.95, 0.95),
        np.clip(an_yaw_cmd / max(an_yaw_max, 1e-6), -0.95, 0.95),
    ], dtype=np.float32)


def alive_indices(items) -> list[int]:
    return [i for i, item in enumerate(items) if item.alive]


def greedy_nearest_defender(env, off_idx: int, used: set[int]) -> int | None:
    off = env.offensives[off_idx]
    best = None
    best_d = float("inf")
    for j, defender in enumerate(env.defensives):
        if (not defender.alive) or j in used:
            continue
        d = off.distance_to(defender.x, defender.y, defender.z)
        if d < best_d:
            best = j
            best_d = d
    if best is None:
        for j, defender in enumerate(env.defensives):
            if defender.alive:
                d = off.distance_to(defender.x, defender.y, defender.z)
                if d < best_d:
                    best = j
                    best_d = d
    return best


def garcia_apollonius_point(env, off, defender) -> np.ndarray:
    """BDDG-inspired evader aimpoint in a local HVT-border frame.

    The Garcia-Casbeer-Von Moll-Pachter BDDG solution is planar and border
    based. For this point-target simulator, the HVT-centered radial coordinate
    is used as the border-normal coordinate and the closed-form Apollonius
    aimpoint is recomputed as state feedback.
    """
    h = env.hvt
    off_xy = np.asarray([off.x - h.x, off.y - h.y], dtype=np.float64)
    s = norm2(off_xy)
    e_s = off_xy / s
    e_l = np.asarray([-e_s[1], e_s[0]], dtype=np.float64)
    def_xy = np.asarray([defender.x - h.x, defender.y - h.y], dtype=np.float64)

    x_e, y_e = 0.0, s
    x_p = float(np.dot(def_xy, e_l))
    y_p = float(np.dot(def_xy, e_s))
    alpha = float(np.clip(off.v / max(defender.v, 1.0), 0.15, 0.97))
    a2 = alpha * alpha
    dij = math.hypot(x_e - x_p, y_e - y_p)
    denom = max(1.0 - a2, 1e-4)
    x_star = (x_e - a2 * x_p) / denom
    y_star = (y_e - a2 * y_p - alpha * dij) / denom

    y_star = float(np.clip(y_star, 80.0, 0.88 * s))
    x_star = float(np.clip(x_star, -550.0, 550.0))
    point_xy = np.asarray([h.x, h.y], dtype=np.float64) + e_s * y_star + e_l * x_star

    # Blend to the actual point target in the terminal segment.
    hvt_dist = off.distance_to(h.x, h.y, h.z)
    terminal = np.clip((900.0 - hvt_dist) / 650.0, 0.0, 1.0)
    point_xy = (1.0 - terminal) * point_xy + terminal * np.asarray([h.x, h.y])
    z_terminal = np.clip((700.0 - hvt_dist) / 500.0, 0.0, 1.0)
    z_ref = (1.0 - z_terminal) * max(80.0, 0.55 * off.z) + z_terminal * h.z
    return np.asarray([point_xy[0], point_xy[1], z_ref], dtype=np.float64)


def actions_garcia_bddg(env) -> list[np.ndarray]:
    used: set[int] = set()
    actions = []
    order = sorted(alive_indices(env.offensives),
                   key=lambda i: env.offensives[i].distance_to(env.hvt.x, env.hvt.y, env.hvt.z))
    assignment: dict[int, int | None] = {}
    for i in order:
        j = greedy_nearest_defender(env, i, used)
        assignment[i] = j
        if j is not None:
            used.add(j)
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        j = assignment.get(i)
        if j is None:
            point = np.asarray([env.hvt.x, env.hvt.y, env.hvt.z], dtype=np.float64)
        else:
            point = garcia_apollonius_point(env, off, env.defensives[j])
        actions.append(action_from_point(off, point, speed_bias=1.5,
                                         yaw_gain=2.0, pitch_gain=2.0,
                                         gamma_limit_deg=15.0))
    return actions


def weiyang_encirclement_point(env, off_idx: int) -> np.ndarray:
    """Wei-Yang multi-attacker/one-target inspired encirclement guidance."""
    h = env.hvt
    off = env.offensives[off_idx]
    n = max(env.n_offensive, 1)
    rel = np.asarray([off.x - h.x, off.y - h.y], dtype=np.float64)
    dist = norm2(rel)
    base_angle = math.atan2(rel[1], rel[0])

    ring_radius = float(np.clip(0.12 * dist, 70.0, 280.0))
    desired_angle = base_angle + (2.0 * math.pi / n) * ((off_idx % 3) - 1) * 0.38
    ring_xy = np.asarray([
        h.x + ring_radius * math.cos(desired_angle),
        h.y + ring_radius * math.sin(desired_angle),
    ], dtype=np.float64)

    # Trackers intentionally keep defenders busy; interceptors preserve a
    # lateral separation until the target is close, then all collapse to HVT.
    tracker = (off_idx % 3 == 0) or off.locked_by_count > 0
    if tracker and off.locked_by_count > 0:
        side = 1.0 if (off_idx % 2 == 0) else -1.0
        perp = np.asarray([-rel[1], rel[0]], dtype=np.float64) / dist
        ring_xy = 0.70 * ring_xy + 0.30 * (np.asarray([h.x, h.y]) + side * 260.0 * perp)

    collapse = np.clip((1650.0 - off.distance_to(h.x, h.y, h.z)) / 1150.0, 0.0, 1.0)
    ring_weight = 0.38 * (1.0 - collapse) + 0.08
    point_xy = ring_weight * ring_xy + (1.0 - ring_weight) * np.asarray([h.x, h.y])
    z_ref = (1.0 - collapse) * max(90.0, 0.62 * off.z) + collapse * h.z
    return np.asarray([point_xy[0], point_xy[1], z_ref], dtype=np.float64)


def actions_weiyang_ta(env) -> list[np.ndarray]:
    actions = []
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        point = weiyang_encirclement_point(env, i)
        actions.append(action_from_point(off, point, speed_bias=1.0,
                                         yaw_gain=2.45, pitch_gain=2.1,
                                         gamma_limit_deg=17.0))
    return actions


def actions_dualcl_field(env) -> list[np.ndarray]:
    """Potential-field evasion baseline inspired by recent DualCL PE setup.

    The cited work uses potential fields for the evader and curriculum-trained
    UAV pursuers. Here we use the directly reproducible part as the attacking
    swarm baseline: HVT attraction plus defender/team repulsion.
    """
    h = env.hvt
    actions = []
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        pos = np.asarray([off.x, off.y, off.z], dtype=np.float64)
        to_h = np.asarray([h.x - off.x, h.y - off.y, h.z - off.z], dtype=np.float64)
        force = 1.35 * to_h / norm2(to_h)
        for defender in env.defensives:
            if not defender.alive:
                continue
            away = pos - np.asarray([defender.x, defender.y, defender.z], dtype=np.float64)
            d = norm2(away)
            if d < 950.0:
                force += 1.15 * away / d * ((950.0 - d) / 950.0) ** 2
        for k, mate in enumerate(env.offensives):
            if k == i or not mate.alive:
                continue
            away = pos - np.asarray([mate.x, mate.y, mate.z], dtype=np.float64)
            d = norm2(away)
            if d < 260.0:
                force += 0.28 * away / d * ((260.0 - d) / 260.0)
        if norm2(to_h) < 850.0:
            force = 2.2 * to_h / norm2(to_h) + 0.35 * force / norm2(force)
        point = pos + 650.0 * force / norm2(force)
        actions.append(action_from_point(off, point, speed_bias=1.8,
                                         yaw_gain=2.4, pitch_gain=2.2,
                                         gamma_limit_deg=18.0))
    return actions


def _roll_point_score(env, off, point: np.ndarray, horizon_s: float = 7.0) -> float:
    h = env.hvt
    pos = np.asarray([off.x, off.y, off.z], dtype=np.float64)
    direction = point - pos
    direction = direction / norm2(direction)
    future = pos + direction * float(off.v) * horizon_s
    d_hvt = norm2(future - np.asarray([h.x, h.y, h.z], dtype=np.float64))
    risk = 0.0
    for defender in env.defensives:
        if not defender.alive:
            continue
        dpos = np.asarray([defender.x, defender.y, defender.z], dtype=np.float64)
        ddir = np.asarray([
            math.cos(defender.gamma) * math.cos(defender.heading),
            math.cos(defender.gamma) * math.sin(defender.heading),
            math.sin(defender.gamma),
        ], dtype=np.float64)
        dfuture = dpos + ddir * float(defender.v) * horizon_s
        sep = norm2(future - dfuture)
        risk += math.exp(-sep / 280.0)
    lateral = abs(float(point[1] - h.y)) / 500.0
    return d_hvt + 420.0 * risk + 35.0 * lateral


def actions_predictive_mpc(env) -> list[np.ndarray]:
    """Prediction-enhanced online planning baseline.

    A light-weight deterministic MPC surrogate for recent online-planning
    MARL PE work: each attacker evaluates a small set of heading/elevation
    candidates against predicted defender motion and chooses the lowest-risk
    HVT-progress waypoint.
    """
    h = env.hvt
    actions = []
    heading_offsets = [-0.55, -0.30, 0.0, 0.30, 0.55]
    gamma_offsets = [-0.14, 0.0, 0.10]
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        to_h = np.asarray([h.x - off.x, h.y - off.y, h.z - off.z], dtype=np.float64)
        base_psi = math.atan2(to_h[1], to_h[0])
        base_gamma = math.atan2(to_h[2], max(math.hypot(to_h[0], to_h[1]), 1.0))
        best_point = None
        best_score = float("inf")
        for ho in heading_offsets:
            for go in gamma_offsets:
                psi = base_psi + ho
                gam = float(np.clip(base_gamma + go, math.radians(-18), math.radians(18)))
                vec = np.asarray([
                    math.cos(gam) * math.cos(psi),
                    math.cos(gam) * math.sin(psi),
                    math.sin(gam),
                ], dtype=np.float64)
                point = np.asarray([off.x, off.y, off.z], dtype=np.float64) + 720.0 * vec
                score = _roll_point_score(env, off, point)
                if score < best_score:
                    best_score = score
                    best_point = point
        if off.distance_to(h.x, h.y, h.z) < 650.0:
            best_point = np.asarray([h.x, h.y, h.z], dtype=np.float64)
        actions.append(action_from_point(off, best_point, speed_bias=2.0,
                                         yaw_gain=2.7, pitch_gain=2.35,
                                         gamma_limit_deg=20.0))
    return actions


def actions_hetero_guard(env) -> list[np.ndarray]:
    """Heterogeneous cooperative-attack target-guarding baseline.

    Inspired by Lee-Das-Shishika-Bakolas target-area guarding with a
    heterogeneous group of cooperative attackers.  High-weight attackers focus
    on target proximity; lower-weight attackers take wider routes and draw
    defender assignments.
    """
    h = env.hvt
    alive = alive_indices(env.offensives)
    if not alive:
        return [np.zeros(3, dtype=np.float32) for _ in env.offensives]
    dists = {
        i: env.offensives[i].distance_to(h.x, h.y, h.z)
        for i in alive
    }
    sorted_alive = sorted(alive, key=lambda i: dists[i])
    n_alive = len(sorted_alive)
    strike_set = set(sorted_alive[:max(1, n_alive // 3)])
    decoy_set = set(sorted_alive[-max(1, n_alive // 3):])

    actions = []
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        pos = np.asarray([off.x, off.y, off.z], dtype=np.float64)
        hpos = np.asarray([h.x, h.y, h.z], dtype=np.float64)
        to_h = hpos - pos
        radial = np.asarray([to_h[0], to_h[1], 0.0], dtype=np.float64)
        if norm2(radial) < 1e-6:
            radial = np.asarray([1.0, 0.0, 0.0])
        tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) / norm2(radial)
        side = 1.0 if (i % 2 == 0) else -1.0

        if i in strike_set:
            # The highest-weight attacker uses the geometric target-guarding
            # aimpoint against its nearest active defender; this is the local
            # saddle-point analogue under our point-HVT simulator.
            j = greedy_nearest_defender(env, i, set())
            if j is not None:
                point = garcia_apollonius_point(env, off, env.defensives[j])
            else:
                point = hpos
            if norm2(to_h) < 850.0:
                point = hpos
            actions.append(action_from_point(off, point, speed_bias=2.0,
                                             yaw_gain=2.45, pitch_gain=2.25,
                                             gamma_limit_deg=19.0))
            continue
        elif i in decoy_set:
            # Low-priority attackers deliberately present a lateral threat,
            # attracting assignments while keeping enough HVT progress.
            lane = hpos + side * tangent * 420.0 + np.asarray([0.0, 0.0, 150.0])
            collapse = np.clip((1350.0 - norm2(to_h)) / 950.0, 0.0, 1.0)
            point = (1.0 - collapse) * lane + collapse * hpos
            actions.append(action_from_point(off, point, speed_bias=1.3,
                                             yaw_gain=2.35, pitch_gain=2.1,
                                             gamma_limit_deg=18.0))
            continue
        else:
            lane = hpos + side * tangent * 240.0 + np.asarray([0.0, 0.0, 100.0])
            collapse = np.clip((1450.0 - norm2(to_h)) / 1050.0, 0.0, 1.0)
            point = (1.0 - collapse) * lane + collapse * hpos
            actions.append(action_from_point(off, point, speed_bias=1.6,
                                             yaw_gain=2.35, pitch_gain=2.1,
                                             gamma_limit_deg=18.0))
            continue

        point = pos + 760.0 * force / norm2(force)
        if norm2(to_h) < 700.0:
            point = hpos
        actions.append(action_from_point(off, point, speed_bias=speed_bias,
                                         yaw_gain=2.65, pitch_gain=2.35,
                                         gamma_limit_deg=20.0))
    return actions


def actions_split_risk(env) -> list[np.ndarray]:
    """Risk-taking/risk-averse split swarm attack baseline.

    Inspired by Chipade-Panagou aerial swarm defense scenarios: attackers can
    split into high-speed risk-taking penetrators and risk-averse sub-swarms
    that maneuver laterally to complicate assignment and interception.
    """
    h = env.hvt
    hpos = np.asarray([h.x, h.y, h.z], dtype=np.float64)
    actions = []
    n = max(env.n_offensive, 1)
    for i, off in enumerate(env.offensives):
        if not off.alive or off.hit_hvt:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        pos = np.asarray([off.x, off.y, off.z], dtype=np.float64)
        to_h = hpos - pos
        dist = norm2(to_h)
        risk_taker = (i % 4 in (0, 1))
        radial = np.asarray([to_h[0], to_h[1], 0.0], dtype=np.float64)
        tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) / norm2(radial)
        side = -1.0 if (i % 2) else 1.0

        if risk_taker:
            j = greedy_nearest_defender(env, i, set())
            point = garcia_apollonius_point(env, off, env.defensives[j]) if j is not None else hpos
            # Small weave only before final approach; terminal phase collapses.
            if 750.0 < dist < 1450.0:
                point = 0.72 * point + 0.28 * (hpos + side * tangent * 120.0)
            if dist < 760.0:
                point = hpos
            speed_bias = 2.0
            yaw_gain = 2.55
        else:
            group = (i // 2) % 3
            lane_offset = (group - 1) * 280.0 + side * 120.0
            lane = hpos + tangent * lane_offset + np.asarray([0.0, 0.0, 130.0])
            collapse = np.clip((1350.0 - dist) / 980.0, 0.0, 1.0)
            point = (1.0 - collapse) * lane + collapse * hpos
            speed_bias = 1.4
            yaw_gain = 2.3

        actions.append(action_from_point(off, point, speed_bias=speed_bias,
                                         yaw_gain=yaw_gain, pitch_gain=2.25,
                                         gamma_limit_deg=20.0))
    return actions


METHODS = {
    "garcia_bddg": actions_garcia_bddg,
    "weiyang_ta": actions_weiyang_ta,
    "dualcl_field": actions_dualcl_field,
    "predictive_mpc": actions_predictive_mpc,
    "hetero_guard": actions_hetero_guard,
    "split_risk": actions_split_risk,
}


def make_env(n: int, seed: int) -> FOVPenetrationEnv:
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    env = FOVPenetrationEnv(
        config={"n_offensive": n, "n_defensive": n},
        scenario="scenario_1",
    )
    env.seed(seed)
    return env


def run_episode(method: str, case: str, out_dir: Path, max_steps: int) -> dict:
    meta = CASES[case]
    n = int(meta["n"])
    seed = int(meta["seed"])
    env = make_env(n, seed)
    env.reset()
    rec, game = init_record(n, n)
    final_info = {}
    action_fn = METHODS[method]

    started = time.time()
    for step in range(max_steps):
        actions = action_fn(env)
        _, _, _, _, dones, infos, _ = env.step(actions)
        final_info = infos[0] if infos else {}
        cur_step = step + 1
        append_trajectory(rec, env, actions, cur_step)
        append_game(game, env)
        if all(dones):
            break

    rec["hvt_x"] = env.hvt.x
    rec["hvt_y"] = env.hvt.y
    rec["hvt_z"] = env.hvt.z
    rec["hit_count"] = env.hit_count
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
    np.savez_compressed(out_dir / "trajectory_data.npz", **finalize_npz_dict(rec))
    np.savez_compressed(out_dir / "game_data.npz", **finalize_npz_dict(game))

    min_d = [float(np.min(np.asarray(x, dtype=float))) if len(x) else float("inf")
             for x in rec["off_d_hvt"]]
    best_agent = int(np.argmin(min_d))
    summary = {
        "case": f"{method}_{case}",
        "source_case": case,
        "method": method,
        "seed": seed,
        "n_offensive": n,
        "n_defensive": n,
        "success": bool(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": [int(i) for i in env.hit_indices],
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
        "elapsed_s": time.time() - started,
        "files": ["game_data.npz", "summary.json", "trajectory_data.npz"],
        "method_notes": {
            "garcia_bddg": "Apollonius aimpoint state feedback adapted from N-vs-M BDDG to an HVT-centered border frame.",
            "weiyang_ta": "Multi-attacker/one-target encirclement guidance with tracker/interceptor role split and terminal collapse.",
            "dualcl_field": "Recent multi-UAV PE potential-field evader component adapted to HVT attraction and defender repulsion.",
            "predictive_mpc": "Recent online-planning PE idea adapted as a deterministic candidate-rollout attacker controller.",
            "hetero_guard": "Heterogeneous cooperative attacker target-guarding policy: high-weight strikers plus low-weight decoys.",
            "split_risk": "Aerial-swarm risk-taking/risk-averse split attack pattern adapted to HVT penetration.",
        }[method],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--out-root", default="outputs/paper_baseline_methods")
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("PAPER_BASELINE_MAX_STEPS", "8000")))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    summaries = {}
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        for case in args.cases:
            if case not in CASES:
                raise ValueError(f"unknown case: {case}")
            case_dir = out_root / f"{method}_{case}"
            print(f"=== {method} / {case} -> {case_dir} ===", flush=True)
            summaries[f"{method}_{case}"] = run_episode(method, case, case_dir, args.max_steps)
            print(json.dumps(summaries[f"{method}_{case}"], indent=2, ensure_ascii=False), flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary_all.json").open("w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"summary_all={out_root / 'summary_all.json'}", flush=True)


if __name__ == "__main__":
    main()
