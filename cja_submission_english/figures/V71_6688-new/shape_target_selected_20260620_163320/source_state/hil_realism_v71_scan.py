#!/usr/bin/env python
"""Scan V71 MAPPO under a defensible semi-physical HIL realism wrapper.

The wrapper models common HIL/networked-control nonidealities:
sensor sampling, stochastic sensor latency, packet dropout with sample hold,
sensor noise/bias/quantization, controller sample rate, command latency/dropout,
and actuator first-order lag with rate/quantization limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import types
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from scripts import collect_v71_4v4_deterministic as collect


HIDDEN = collect.HIDDEN
MAX_STEPS = collect.MAX_STEPS

_RAW_ENV = None
_ENV = None
_POLICIES = None
_CASE = None
_CFG = None


@dataclass
class HILConfig:
    sensor_sample_steps: int = 5       # 50 ms at dt=0.01 s
    sensor_delay_steps: int = 8        # base 80 ms
    sensor_jitter_steps: int = 4       # additional 0..40 ms
    sensor_dropout_prob: float = 0.02
    obs_noise_std: float = 0.01
    obs_bias_std: float = 0.002
    obs_bias_rw_std: float = 0.00002
    obs_quant_step: float = 0.001
    policy_sample_steps: int = 5       # embedded policy at 20 Hz
    command_delay_steps: int = 2       # base 20 ms
    command_jitter_steps: int = 2      # additional 0..20 ms
    command_dropout_prob: float = 0.01
    action_quant_step: float = 0.002
    actuator_tau_s: float = 0.08
    action_rate_limit_per_s: float = 6.0
    enable_defense_hil: bool = False
    defense_sample_steps: int = 5          # defender seeker/track manager at 20 Hz
    defense_delay_steps: int = 10          # base 100 ms target-state latency
    defense_jitter_steps: int = 5          # additional 0..50 ms
    defense_dropout_prob: float = 0.03
    defense_pos_noise_m: float = 8.0
    defense_vel_noise_mps: float = 2.0
    defense_pos_quant_m: float = 1.0
    defense_vel_quant_mps: float = 0.1
    defense_fov_false_negative_prob: float = 0.04
    max_steps: int = MAX_STEPS


def parse_case_seed(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items:
        case, seed_s = item.split(":", 1)
        out.append((case.strip().lower(), int(seed_s)))
    return out


def quantize(x: np.ndarray, q: float) -> np.ndarray:
    if q <= 0.0:
        return x
    return np.round(x / q) * q


class HILChannel:
    def __init__(self, cfg: HILConfig, n_agents: int, action_shapes, dt: float, seed: int):
        self.cfg = cfg
        self.n_agents = n_agents
        self.dt = dt
        self.rng = np.random.default_rng(seed + 104729)
        self.sensor_events: list[tuple[int, list[np.ndarray]]] = []
        self.command_events: list[tuple[int, list[np.ndarray]]] = []
        self.current_obs: list[np.ndarray] | None = None
        self.commanded_action = [np.zeros(shape, dtype=np.float32) for shape in action_shapes]
        self.applied_action = [np.zeros(shape, dtype=np.float32) for shape in action_shapes]
        self.obs_bias: list[np.ndarray] | None = None
        self.stats = {
            "sensor_generated": 0,
            "sensor_delivered": 0,
            "sensor_dropped": 0,
            "command_generated": 0,
            "command_delivered": 0,
            "command_dropped": 0,
            "sensor_latency_steps": [],
            "command_latency_steps": [],
        }

    def _corrupt_obs(self, obs) -> list[np.ndarray]:
        arrays = [np.asarray(item, dtype=np.float32).copy() for item in obs]
        if self.obs_bias is None:
            self.obs_bias = [
                self.rng.normal(0.0, self.cfg.obs_bias_std, size=a.shape).astype(np.float32)
                for a in arrays
            ]
        out = []
        for i, arr in enumerate(arrays):
            self.obs_bias[i] += self.rng.normal(
                0.0, self.cfg.obs_bias_rw_std, size=arr.shape
            ).astype(np.float32)
            noisy = arr + self.obs_bias[i]
            if self.cfg.obs_noise_std > 0:
                noisy = noisy + self.rng.normal(
                    0.0, self.cfg.obs_noise_std, size=arr.shape
                ).astype(np.float32)
            out.append(quantize(noisy, self.cfg.obs_quant_step).astype(np.float32))
        return out

    def init_obs(self, obs):
        self.current_obs = self._corrupt_obs(obs)

    def sensor_tick(self, step: int, obs):
        if step % max(1, self.cfg.sensor_sample_steps) != 0:
            return
        self.stats["sensor_generated"] += 1
        if self.rng.random() < self.cfg.sensor_dropout_prob:
            self.stats["sensor_dropped"] += 1
            return
        jitter = int(self.rng.integers(0, max(0, self.cfg.sensor_jitter_steps) + 1))
        latency = max(0, int(self.cfg.sensor_delay_steps) + jitter)
        self.sensor_events.append((step + latency, self._corrupt_obs(obs)))
        self.stats["sensor_latency_steps"].append(latency)

    def deliver_sensor(self, step: int):
        if not self.sensor_events:
            return
        pending = []
        delivered = None
        for due, packet in self.sensor_events:
            if due <= step:
                delivered = packet
                self.stats["sensor_delivered"] += 1
            else:
                pending.append((due, packet))
        self.sensor_events = pending
        if delivered is not None:
            self.current_obs = delivered

    def command_tick(self, step: int, actions):
        self.stats["command_generated"] += 1
        if self.rng.random() < self.cfg.command_dropout_prob:
            self.stats["command_dropped"] += 1
            return
        jitter = int(self.rng.integers(0, max(0, self.cfg.command_jitter_steps) + 1))
        latency = max(0, int(self.cfg.command_delay_steps) + jitter)
        packet = [
            quantize(np.asarray(a, dtype=np.float32), self.cfg.action_quant_step).astype(np.float32)
            for a in actions
        ]
        self.command_events.append((step + latency, packet))
        self.stats["command_latency_steps"].append(latency)

    def deliver_command(self, step: int):
        if not self.command_events:
            return
        pending = []
        delivered = None
        for due, packet in self.command_events:
            if due <= step:
                delivered = packet
                self.stats["command_delivered"] += 1
            else:
                pending.append((due, packet))
        self.command_events = pending
        if delivered is not None:
            self.commanded_action = delivered

    def actuator_step(self):
        alpha = 1.0 if self.cfg.actuator_tau_s <= 0 else self.dt / (self.cfg.actuator_tau_s + self.dt)
        max_delta = max(0.0, self.cfg.action_rate_limit_per_s) * self.dt
        out = []
        for cur, cmd in zip(self.applied_action, self.commanded_action):
            target = cur + alpha * (cmd - cur)
            if max_delta > 0:
                target = cur + np.clip(target - cur, -max_delta, max_delta)
            target = quantize(target, self.cfg.action_quant_step).astype(np.float32)
            out.append(target)
        self.applied_action = [a.copy() for a in out]
        return out

    def packed_stats(self):
        out = dict(self.stats)
        for key in ("sensor_latency_steps", "command_latency_steps"):
            vals = np.asarray(out[key], dtype=float)
            if vals.size:
                out[key] = {
                    "mean": float(np.mean(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "max": int(np.max(vals)),
                }
            else:
                out[key] = {"mean": 0.0, "p95": 0.0, "max": 0}
        return out


class DefenseHILChannel:
    """Runtime-only defender perception and target-manager nonidealities."""

    def __init__(self, cfg: HILConfig, raw_env, seed: int):
        self.cfg = cfg
        self.raw_env = raw_env
        self.n_def = raw_env.n_defensive
        self.n_off = raw_env.n_offensive
        self.rng = np.random.default_rng(seed + 130363)
        self.events: list[tuple[int, int, int, np.ndarray]] = []
        self.perceived = [
            [None for _ in range(self.n_off)]
            for _ in range(self.n_def)
        ]
        self.stats = {
            "defense_samples_generated": 0,
            "defense_samples_delivered": 0,
            "defense_samples_dropped": 0,
            "defense_latency_steps": [],
            "defense_fov_false_negatives": 0,
        }

    def _off_idx(self, target) -> int | None:
        uid = getattr(target, "uid", None)
        if uid is not None and 0 <= int(uid) < self.n_off:
            return int(uid)
        for i, off in enumerate(self.raw_env.offensives):
            if off is target:
                return i
        return None

    def _state_from_target(self, target) -> np.ndarray:
        cos_g = np.cos(target.gamma)
        vx = target.v * cos_g * np.cos(target.heading)
        vy = target.v * cos_g * np.sin(target.heading)
        vz = target.v * np.sin(target.gamma)
        state = np.array([target.x, target.y, target.z, vx, vy, vz], dtype=np.float32)
        state[:3] += self.rng.normal(0.0, self.cfg.defense_pos_noise_m, size=3).astype(np.float32)
        state[3:] += self.rng.normal(0.0, self.cfg.defense_vel_noise_mps, size=3).astype(np.float32)
        state[:3] = quantize(state[:3], self.cfg.defense_pos_quant_m)
        state[3:] = quantize(state[3:], self.cfg.defense_vel_quant_mps)
        return state.astype(np.float32)

    def observe(self, step: int):
        if step % max(1, self.cfg.defense_sample_steps) != 0:
            return
        for di, d in enumerate(self.raw_env.defensives):
            if not d.alive:
                continue
            for oi, off in enumerate(self.raw_env.offensives):
                if not off.alive or off.hit_hvt:
                    continue
                self.stats["defense_samples_generated"] += 1
                if self.rng.random() < self.cfg.defense_dropout_prob:
                    self.stats["defense_samples_dropped"] += 1
                    continue
                jitter = int(self.rng.integers(0, max(0, self.cfg.defense_jitter_steps) + 1))
                latency = max(0, int(self.cfg.defense_delay_steps) + jitter)
                self.events.append((step + latency, di, oi, self._state_from_target(off)))
                self.stats["defense_latency_steps"].append(latency)

    def deliver(self, step: int):
        if not self.events:
            return
        pending = []
        for due, di, oi, state in self.events:
            if due <= step:
                self.perceived[di][oi] = state
                self.stats["defense_samples_delivered"] += 1
            else:
                pending.append((due, di, oi, state))
        self.events = pending

    def state_for(self, policy, target):
        oi = self._off_idx(target)
        if oi is None:
            return None
        di = int(getattr(policy, "patrol_idx", 0))
        if di < 0 or di >= self.n_def:
            return None
        return self.perceived[di][oi]

    def perceived_distance(self, policy, target) -> float:
        state = self.state_for(policy, target)
        if state is None:
            return float("inf")
        intc = policy.interceptor
        dx = float(state[0]) - intc.x
        dy = float(state[1]) - intc.y
        dz = float(state[2]) - intc.z
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def packed_stats(self):
        out = dict(self.stats)
        vals = np.asarray(out["defense_latency_steps"], dtype=float)
        if vals.size:
            out["defense_latency_steps"] = {
                "mean": float(np.mean(vals)),
                "p95": float(np.percentile(vals, 95)),
                "max": int(np.max(vals)),
            }
        else:
            out["defense_latency_steps"] = {"mean": 0.0, "p95": 0.0, "max": 0}
        return out


def _hil_update_known_position(self, target):
    channel = getattr(self, "_hil_defense_channel", None)
    state = channel.state_for(self, target) if channel is not None else None
    if state is None:
        return self._hil_original_update_known_position(target)
    self.target_pos_known = [float(x) for x in state]


def _hil_get_best_alive_target(self, offensives):
    channel = getattr(self, "_hil_defense_channel", None)
    if channel is None:
        return self._hil_original_get_best_alive_target(offensives)
    best = None
    best_dist = float("inf")
    for off in offensives:
        if not off.alive or off.hit_hvt:
            continue
        d = channel.perceived_distance(self, off)
        if d < best_dist:
            best_dist = d
            best = off
    if best is not None and np.isfinite(best_dist):
        return best
    return self._hil_original_get_best_alive_target(offensives)


def _hil_try_fov_lock(self, off_idx, offensive, current_step, already_locked_offensives=None):
    channel = getattr(self, "_hil_defense_channel", None)
    if channel is None:
        return self._hil_original_try_fov_lock(
            off_idx, offensive, current_step, already_locked_offensives
        )
    if self.lock_mode in (self.STATE_LOCKED, self.STATE_MISSED, self.STATE_ABANDONED):
        return False
    if already_locked_offensives is not None and off_idx in already_locked_offensives:
        return False
    intc = self.interceptor
    if not intc.alive or not offensive.alive:
        return False

    state = channel.state_for(self, offensive)
    if state is None:
        return False
    fov_half = self.lock_rules.get("lock_fov_threshold", self.config["fov_half_angle"])
    lock_range = self.lock_rules.get("lock_range_threshold", self.config["detection_range"])
    in_fov = intc.is_in_fov(float(state[0]), float(state[1]), float(state[2]), fov_half, lock_range)
    if not in_fov:
        return False
    if channel.rng.random() < channel.cfg.defense_fov_false_negative_prob:
        channel.stats["defense_fov_false_negatives"] += 1
        return False

    self.current_locked_target_idx = off_idx
    self.target = offensive
    self.lock_mode = self.STATE_LOCKED
    self.engagement_state = self.STATE_LOCKED
    self.first_lock_time = current_step
    self.has_ever_locked = True
    self.fov_loss_counter = 0
    self.tracking_steps = 0
    self.prev_los_az = None
    self.prev_los_el = None
    self.engagement_min_dist = float("inf")
    self._update_known_position(offensive)
    return True


def install_defense_hil(raw_env, cfg: HILConfig, seed: int) -> DefenseHILChannel:
    channel = DefenseHILChannel(cfg, raw_env, seed)
    for policy in raw_env.defensive_policies:
        if not getattr(policy, "_hil_defense_patched", False):
            policy._hil_original_update_known_position = policy._update_known_position
            policy._hil_original_get_best_alive_target = policy._get_best_alive_target
            policy._hil_original_try_fov_lock = policy.try_fov_lock
            policy._update_known_position = types.MethodType(_hil_update_known_position, policy)
            policy._get_best_alive_target = types.MethodType(_hil_get_best_alive_target, policy)
            policy.try_fov_lock = types.MethodType(_hil_try_fov_lock, policy)
            policy._hil_defense_patched = True
        policy._hil_defense_channel = channel
    return channel


def make_actions(policies, obs, rnn_states, masks):
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
        actions.append(action.cpu().numpy().flatten().astype(np.float32))
        new_rnn.append(hidden)
    return actions, new_rnn


def init_worker(case: str, cfg_dict: dict):
    global _RAW_ENV, _ENV, _POLICIES, _CASE, _CFG
    torch.set_num_threads(1)
    _CASE = case
    _CFG = HILConfig(**cfg_dict)
    n_off, n_def = collect.parse_case(case)
    _RAW_ENV, _ENV = collect.make_raw_env(n_off, n_def)
    _POLICIES = collect.load_cloned_policies(_RAW_ENV)


def run_hil_episode(seed: int, record_dir: Path | None = None) -> dict:
    raw_env, env, policies, cfg = _RAW_ENV, _ENV, _POLICIES, _CFG
    if raw_env is None or env is None or policies is None or cfg is None:
        raise RuntimeError("worker not initialized")

    env.seed(seed)
    obs, _, _ = env.reset()
    n_agents = env.n_agents
    hvt = env.hvt
    action_shapes = [raw_env.action_space[i].shape for i in range(n_agents)]
    channel = HILChannel(cfg, n_agents, action_shapes, raw_env.dt, seed)
    defense_channel = install_defense_hil(raw_env, cfg, seed) if cfg.enable_defense_hil else None
    channel.init_obs(obs)

    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    rec = game = None
    if record_dir is not None:
        rec, game = collect.init_record(raw_env.n_offensive, raw_env.n_defensive)

    last_policy_actions = [np.zeros(shape, dtype=np.float32) for shape in action_shapes]
    for step in range(int(cfg.max_steps)):
        if defense_channel is not None:
            defense_channel.observe(step)
            defense_channel.deliver(step)
        channel.sensor_tick(step, obs)
        channel.deliver_sensor(step)

        if step % max(1, cfg.policy_sample_steps) == 0:
            last_policy_actions, rnn_states = make_actions(
                policies, channel.current_obs, rnn_states, masks
            )
            channel.command_tick(step, last_policy_actions)

        channel.deliver_command(step)
        applied_actions = channel.actuator_step()
        obs, _, _, _, dones, infos, _ = env.step(applied_actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}

        if rec is not None:
            collect.append_trajectory(rec, raw_env, applied_actions, final_step)
            collect.append_game(game, raw_env)

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
    summary = {
        "case": _CASE,
        "seed": int(seed),
        "model_dir": str(collect.MODEL_DIR),
        "clone_map": {str(i): int(i % 4) for i in range(raw_env.n_offensive)},
        "n_offensive": raw_env.n_offensive,
        "n_defensive": raw_env.n_defensive,
        "hil_config": asdict(cfg),
        "hil_stats": channel.packed_stats(),
        "defense_hil_stats": (
            defense_channel.packed_stats() if defense_channel is not None else {"enabled": False}
        ),
        "success": bool(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": [int(i) for i in getattr(raw_env, "hit_indices", [])],
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * raw_env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_hvt_distance_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
        "min_dist_per_agent_m": [round(float(x), 3) for x in min_d],
    }

    if rec is not None and record_dir is not None:
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

        summary["hit_step"] = {str(k): int(v) for k, v in rec["hit_step"].items()}
        summary["death_step"] = {str(k): int(v) for k, v in rec["death_step"].items()}
        record_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(record_dir / "trajectory_data.npz", **collect.finalize_npz_dict(rec))
        np.savez_compressed(record_dir / "game_data.npz", **collect.finalize_npz_dict(game))
        with (record_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def scan_one(seed: int) -> dict:
    return run_hil_episode(seed, record_dir=None)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "seed", "success", "hit_count", "hit_indices", "done_reason",
        "final_step", "offensive_alive", "defensive_alive", "best_agent",
        "best_hvt_distance_m", "best_min_step", "min_dist_per_agent_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["seed"])))


def rank_recorded(case_dir: Path) -> dict:
    sm = json.loads((case_dir / "summary.json").read_text())
    d = np.load(case_dir / "trajectory_data.npz", allow_pickle=True)
    gd = np.load(case_dir / "game_data.npz", allow_pickle=True)

    hitter = int((sm.get("hit_indices") or [sm.get("best_agent", 0)])[0])
    ltgt = np.asarray(d["def_ltgt"], dtype=int)
    lmode = np.asarray(d["def_lmode"], dtype=int)
    alive = np.asarray(d["def_alive"], dtype=int)
    t = np.asarray(d["time"], dtype=float)
    valid = (ltgt >= 0) & (alive == 1)
    late = valid & (t[None, :] >= 10.0)
    locked = valid & (lmode == 2)
    hitter_share = float(np.mean(ltgt[valid] == hitter)) if valid.any() else 1.0
    late_share = float(np.mean(ltgt[late] == hitter)) if late.any() else hitter_share
    locked_share = float(np.mean(ltgt[locked] == hitter)) if locked.any() else hitter_share

    lock = np.asarray(gd["decoy_lock_pressure"], dtype=float) if "decoy_lock_pressure" in gd.files else None
    if lock is not None and lock.ndim >= 2 and hitter < lock.shape[0]:
        h_lock = float(np.nanmean(lock[hitter]))
        others = [i for i in range(lock.shape[0]) if i != hitter]
        d_lock = float(np.nanmean(lock[others])) if others else 0.0
    else:
        h_lock = d_lock = 0.0

    # Quality score: precise success first, then assignment disturbance and survivability.
    precision = max(0.0, 5.5 - float(sm["best_hvt_distance_m"]))
    score = (
        100.0 * int(bool(sm["success"]))
        + 3.0 * precision
        + 2.0 * (1.0 - late_share)
        + 1.0 * (1.0 - locked_share)
        + 0.5 * max(d_lock - h_lock, 0.0)
        + 0.002 * float(sm["final_step"])
        + 0.5 * max(int(sm.get("offensive_alive", 0)), 0)
    )
    out = dict(sm)
    out.update({
        "case_dir": str(case_dir),
        "hitter": hitter,
        "hitter_target_share": hitter_share,
        "late_hitter_target_share": late_share,
        "locked_hitter_share": locked_share,
        "hitter_lock_pressure_mean": h_lock,
        "decoy_lock_pressure_mean": d_lock,
        "score": float(score),
    })
    return out


def run_case(
    case: str,
    seed_start: int,
    seed_end: int,
    workers: int,
    record_top: int,
    out_root: Path,
    cfg: HILConfig,
    stop_success_count: int = 0,
):
    seeds = list(range(seed_start, seed_end))
    print(f"=== HIL scan {case}: seeds={seed_start}..{seed_end - 1}, workers={workers} ===", flush=True)
    rows = []
    with mp.Pool(
        processes=max(1, min(workers, len(seeds))),
        initializer=init_worker,
        initargs=(case, asdict(cfg)),
    ) as pool:
        for idx, row in enumerate(pool.imap_unordered(scan_one, seeds), 1):
            rows.append(row)
            print(
                f"{case} {idx:04d}/{len(seeds):04d} seed={row['seed']} "
                f"success={int(row['success'])} best={row['best_hvt_distance_m']:.2f} "
                f"agent={row['best_agent']} step={row['final_step']}",
                flush=True,
            )
            if stop_success_count > 0 and sum(1 for r in rows if r["success"]) >= stop_success_count:
                print(
                    f"{case}: stop after {stop_success_count} successes "
                    f"({idx}/{len(seeds)} seeds returned)",
                    flush=True,
                )
                pool.terminate()
                break

    case_root = out_root / case
    write_csv(case_root / "scan_all.csv", rows)
    successes = [r for r in rows if r["success"]]
    if successes:
        candidates = sorted(
            successes,
            key=lambda r: (
                float(r["best_hvt_distance_m"]),
                -int(r.get("offensive_alive", 0)),
                int(r["final_step"]),
            ),
        )[:record_top]
    else:
        candidates = sorted(rows, key=lambda r: float(r["best_hvt_distance_m"]))[:min(record_top, len(rows))]

    metrics = []
    # Reinitialize in the main process for recording selected candidates.
    init_worker(case, asdict(cfg))
    for row in candidates:
        seed = int(row["seed"])
        rec_dir = case_root / f"seed{seed}"
        print(f"{case}: record seed={seed}", flush=True)
        run_hil_episode(seed, record_dir=rec_dir)
        metrics.append(rank_recorded(rec_dir))

    metrics.sort(key=lambda r: r["score"], reverse=True)
    (case_root / "ranked_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    best = metrics[0] if metrics else None
    return {
        "case": case,
        "n_scanned": len(rows),
        "success_count": len(successes),
        "best": best,
        "ranked_metrics": str(case_root / "ranked_metrics.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["4v4", "6v6", "8v8"])
    parser.add_argument("--seed-start-4v4", type=int, default=90000)
    parser.add_argument("--seed-end-4v4", type=int, default=90400)
    parser.add_argument("--seed-start-6v6", type=int, default=60000)
    parser.add_argument("--seed-end-6v6", type=int, default=60120)
    parser.add_argument("--seed-start-8v8", type=int, default=80000)
    parser.add_argument("--seed-end-8v8", type=int, default=80120)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--record-top", type=int, default=12)
    parser.add_argument("--stop-success-count", type=int, default=0)
    parser.add_argument("--out-root", default="/tmp/v71_hil_realism_scan")

    # HIL realism parameters.
    parser.add_argument("--sensor-sample-steps", type=int, default=5)
    parser.add_argument("--sensor-delay-steps", type=int, default=8)
    parser.add_argument("--sensor-jitter-steps", type=int, default=4)
    parser.add_argument("--sensor-dropout-prob", type=float, default=0.02)
    parser.add_argument("--obs-noise-std", type=float, default=0.01)
    parser.add_argument("--obs-bias-std", type=float, default=0.002)
    parser.add_argument("--obs-bias-rw-std", type=float, default=0.00002)
    parser.add_argument("--obs-quant-step", type=float, default=0.001)
    parser.add_argument("--policy-sample-steps", type=int, default=5)
    parser.add_argument("--command-delay-steps", type=int, default=2)
    parser.add_argument("--command-jitter-steps", type=int, default=2)
    parser.add_argument("--command-dropout-prob", type=float, default=0.01)
    parser.add_argument("--action-quant-step", type=float, default=0.002)
    parser.add_argument("--actuator-tau-s", type=float, default=0.08)
    parser.add_argument("--action-rate-limit-per-s", type=float, default=6.0)
    parser.add_argument("--enable-defense-hil", action="store_true")
    parser.add_argument("--defense-sample-steps", type=int, default=5)
    parser.add_argument("--defense-delay-steps", type=int, default=10)
    parser.add_argument("--defense-jitter-steps", type=int, default=5)
    parser.add_argument("--defense-dropout-prob", type=float, default=0.03)
    parser.add_argument("--defense-pos-noise-m", type=float, default=8.0)
    parser.add_argument("--defense-vel-noise-mps", type=float, default=2.0)
    parser.add_argument("--defense-pos-quant-m", type=float, default=1.0)
    parser.add_argument("--defense-vel-quant-mps", type=float, default=0.1)
    parser.add_argument("--defense-fov-false-negative-prob", type=float, default=0.04)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    cfg = HILConfig(
        sensor_sample_steps=args.sensor_sample_steps,
        sensor_delay_steps=args.sensor_delay_steps,
        sensor_jitter_steps=args.sensor_jitter_steps,
        sensor_dropout_prob=args.sensor_dropout_prob,
        obs_noise_std=args.obs_noise_std,
        obs_bias_std=args.obs_bias_std,
        obs_bias_rw_std=args.obs_bias_rw_std,
        obs_quant_step=args.obs_quant_step,
        policy_sample_steps=args.policy_sample_steps,
        command_delay_steps=args.command_delay_steps,
        command_jitter_steps=args.command_jitter_steps,
        command_dropout_prob=args.command_dropout_prob,
        action_quant_step=args.action_quant_step,
        actuator_tau_s=args.actuator_tau_s,
        action_rate_limit_per_s=args.action_rate_limit_per_s,
        enable_defense_hil=args.enable_defense_hil,
        defense_sample_steps=args.defense_sample_steps,
        defense_delay_steps=args.defense_delay_steps,
        defense_jitter_steps=args.defense_jitter_steps,
        defense_dropout_prob=args.defense_dropout_prob,
        defense_pos_noise_m=args.defense_pos_noise_m,
        defense_vel_noise_mps=args.defense_vel_noise_mps,
        defense_pos_quant_m=args.defense_pos_quant_m,
        defense_vel_quant_mps=args.defense_vel_quant_mps,
        defense_fov_false_negative_prob=args.defense_fov_false_negative_prob,
        max_steps=args.max_steps,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"{stamp}_hil_realism"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries = {}
    for case in args.cases:
        if case == "4v4":
            start, end = args.seed_start_4v4, args.seed_end_4v4
        elif case == "6v6":
            start, end = args.seed_start_6v6, args.seed_end_6v6
        elif case == "8v8":
            start, end = args.seed_start_8v8, args.seed_end_8v8
        else:
            raise ValueError(case)
        summaries[case] = run_case(
            case,
            start,
            end,
            args.workers,
            args.record_top,
            out_root,
            cfg,
            stop_success_count=args.stop_success_count,
        )

    top = {
        "created_at": stamp,
        "elapsed_s": time.time() - started,
        "out_root": str(out_root),
        "hil_config": asdict(cfg),
        "model_dir": str(collect.MODEL_DIR),
        "cases": summaries,
    }
    (out_root / "summary_all.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    print("=== all done ===", flush=True)
    print(json.dumps(top, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
