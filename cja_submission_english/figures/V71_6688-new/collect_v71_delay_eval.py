#!/usr/bin/env python
"""Evaluate V71 MAPPO cases with semi-physical sensing/action delay."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
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


def parse_case_seed(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"case seed must be CASE:SEED, got {item!r}")
        case, seed_s = item.split(":", 1)
        case = case.strip().lower()
        if case not in {"4v4", "6v6", "8v8"}:
            raise ValueError(f"unsupported case {case!r}")
        out.append((case, int(seed_s)))
    return out


def obs_with_noise(obs, rng: np.random.Generator, noise_std: float):
    if noise_std <= 0.0:
        return obs
    noisy = []
    for item in obs:
        arr = np.asarray(item, dtype=np.float32)
        noisy.append(arr + rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32))
    return noisy


def actor_actions(policies, obs, rnn_states, masks, rng, noise_std: float):
    policy_obs = obs_with_noise(obs, rng, noise_std)
    actions = []
    new_rnn = []
    for agent_id, policy in enumerate(policies):
        obs_tensor = torch.FloatTensor(np.asarray(policy_obs[agent_id]).flatten()).unsqueeze(0)
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


def delayed_rollout(
    case: str,
    seed: int,
    out_dir: Path,
    obs_delay_steps: int,
    action_delay_steps: int,
    obs_noise_std: float,
):
    n_off, n_def = collect.parse_case(case)
    raw_env, env = collect.make_raw_env(n_off, n_def, seed=seed)
    policies = collect.load_cloned_policies(raw_env)

    rng = np.random.default_rng(seed + 1729)
    obs, _, _ = env.reset()
    obs_delay = max(0, int(obs_delay_steps))
    action_delay = max(0, int(action_delay_steps))
    obs_queue = deque([[np.asarray(item, dtype=np.float32).copy() for item in obs]
                       for _ in range(obs_delay + 1)],
                      maxlen=obs_delay + 1)
    zero_action = [np.zeros(raw_env.action_space[i].shape, dtype=np.float32) for i in range(n_off)]
    action_queue = deque([zero_action for _ in range(action_delay)], maxlen=max(action_delay, 1))

    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_off)]
    masks = [torch.ones(1, 1) for _ in range(n_off)]
    rec, game = collect.init_record(n_off, n_def)
    final_info = {}

    for step in range(MAX_STEPS):
        delayed_obs = list(obs_queue[0])
        computed_actions, rnn_states = actor_actions(
            policies, delayed_obs, rnn_states, masks, rng, float(obs_noise_std)
        )
        if action_delay > 0:
            env_actions = action_queue.popleft()
            action_queue.append(computed_actions)
        else:
            env_actions = computed_actions

        obs, _, _, _, dones, infos, _ = env.step(env_actions)
        obs_queue.append([np.asarray(item, dtype=np.float32).copy() for item in obs])
        final_info = infos[0] if infos else {}
        cur_step = step + 1
        collect.append_trajectory(rec, raw_env, env_actions, cur_step)
        collect.append_game(game, raw_env)
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
    np.savez_compressed(out_dir / "trajectory_data.npz", **collect.finalize_npz_dict(rec))
    np.savez_compressed(out_dir / "game_data.npz", **collect.finalize_npz_dict(game))

    min_d = [float(np.min(np.asarray(x, dtype=float))) for x in rec["off_d_hvt"]]
    best_agent = int(np.argmin(min_d))
    summary = {
        "case": case,
        "seed": int(seed),
        "model_dir": str(collect.MODEL_DIR),
        "clone_map": {str(i): int(i % 4) for i in range(n_off)},
        "n_offensive": n_off,
        "n_defensive": n_def,
        "delay_model": {
            "obs_delay_steps": int(obs_delay),
            "obs_delay_s": float(obs_delay * raw_env.dt),
            "action_delay_steps": int(action_delay),
            "action_delay_s": float(action_delay * raw_env.dt),
            "obs_noise_std": float(obs_noise_std),
        },
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
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-seed", nargs="+", default=["4v4:90019", "6v6:60015", "8v8:80047"])
    parser.add_argument("--obs-delay-steps", type=int, default=10)
    parser.add_argument("--action-delay-steps", type=int, default=2)
    parser.add_argument("--obs-noise-std", type=float, default=0.005)
    parser.add_argument("--out-root", default="/tmp/v71_delay_eval")
    args = parser.parse_args()

    torch.set_num_threads(1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / (
        f"{stamp}_obs{args.obs_delay_steps}_act{args.action_delay_steps}_noise{args.obs_noise_std:g}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries = {}
    for case, seed in parse_case_seed(args.case_seed):
        print(f"=== delayed rollout {case} seed={seed} ===", flush=True)
        summaries[case] = delayed_rollout(
            case,
            seed,
            out_root / case,
            args.obs_delay_steps,
            args.action_delay_steps,
            args.obs_noise_std,
        )
        print(json.dumps(summaries[case], ensure_ascii=False, indent=2), flush=True)

    top = {
        "created_at": stamp,
        "elapsed_s": time.time() - started,
        "out_root": str(out_root),
        "cases": summaries,
    }
    with (out_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    print("=== all done ===", flush=True)
    print(json.dumps(top, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
