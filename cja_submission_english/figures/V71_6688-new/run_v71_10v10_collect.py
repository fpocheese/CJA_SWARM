#!/usr/bin/env python3
"""Standalone 10v10 V71 success collector.

This script is intended to run on the remote swarm_attack_v2 repository
without modifying the repository's checked-in evaluation scripts.

It scans random seeds for successful 10v10 episodes, then reruns the selected
successful seeds with full recording so the result directories can be copied
back locally and plotted with plot_all.py.
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


PROJECT_ROOT = Path(os.environ.get("V71_PROJECT_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from scripts import collect_v71_4v4_deterministic as collect


RAW_ENV = None
ENV = None
POLICIES = None


def init_worker(model_dir: str):
    global RAW_ENV, ENV, POLICIES
    torch.set_num_threads(1)
    os.environ["V71_MODEL_DIR"] = model_dir
    RAW_ENV, ENV = collect.make_raw_env(10, 10)
    POLICIES = collect.load_cloned_policies(RAW_ENV)


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
        actions.append(action.cpu().numpy().flatten().astype(np.float32))
        new_rnn.append(hidden)
    return actions, new_rnn


def run_episode(seed: int, record_dir: Path | None = None) -> dict:
    raw_env, env, policies = RAW_ENV, ENV, POLICIES
    if raw_env is None or env is None or policies is None:
        raise RuntimeError("worker not initialized")

    env.seed(seed)
    obs, _, _ = env.reset()
    n_agents = env.n_agents
    rnn_states = [torch.zeros(1, 1, collect.HIDDEN) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    min_d = [off.distance_to(raw_env.hvt.x, raw_env.hvt.y, raw_env.hvt.z) for off in env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    rec = game = None
    if record_dir is not None:
        rec, game = collect.init_record(raw_env.n_offensive, raw_env.n_defensive)

    for step in range(collect.MAX_STEPS):
        actions, rnn_states = actor_actions(policies, obs, rnn_states, masks)
        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}
        if rec is not None:
            collect.append_trajectory(rec, raw_env, actions, final_step)
            collect.append_game(game, raw_env)
        for i, off in enumerate(env.offensives):
            d = off.distance_to(raw_env.hvt.x, raw_env.hvt.y, raw_env.hvt.z)
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
        "case": "10v10",
        "seed": int(seed),
        "model_dir": str(os.environ.get("V71_MODEL_DIR", "")),
        "clone_map": {str(i): int(i % 4) for i in range(raw_env.n_offensive)},
        "n_offensive": raw_env.n_offensive,
        "n_defensive": raw_env.n_defensive,
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
        (record_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def scan_seeds(seed_start: int, seed_end: int, workers: int, model_dir: str, stop_success_count: int):
    seeds = list(range(seed_start, seed_end))
    rows = []
    successes = []
    with mp.Pool(
        processes=max(1, min(workers, len(seeds))),
        initializer=init_worker,
        initargs=(model_dir,),
    ) as pool:
        for idx, row in enumerate(pool.imap_unordered(run_episode, seeds), 1):
            rows.append(row)
            print(
                f"seed={row['seed']} success={int(row['success'])} "
                f"hits={row['hit_count']} best={row['best_hvt_distance_m']:.2f} "
                f"step={row['final_step']}",
                flush=True,
            )
            if row["success"]:
                successes.append(row)
            if stop_success_count > 0 and len(successes) >= stop_success_count:
                pool.terminate()
                break
    return rows, successes


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "success", "hit_count", "hit_indices", "done_reason",
        "final_step", "final_time_s", "offensive_alive", "defensive_alive",
        "best_agent", "best_hvt_distance_m", "best_min_step",
        "min_dist_per_agent_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["seed"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--seed-end", type=int, default=101000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stop-success-count", type=int, default=3)
    parser.add_argument(
        "--model-dir",
        default="outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models",
    )
    parser.add_argument("--out-root", default="/tmp/v71_10v10_success")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"{stamp}_10v10"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows, successes = scan_seeds(
        args.seed_start,
        args.seed_end,
        args.workers,
        args.model_dir,
        args.stop_success_count,
    )

    write_csv(out_root / "scan_all.csv", rows)
    selected = sorted(successes, key=lambda r: (float(r["best_hvt_distance_m"]), int(r["final_step"])))[:3]

    selected_summaries = []
    if not args.scan_only:
        init_worker(args.model_dir)
        for row in selected:
            seed = int(row["seed"])
            rec_dir = out_root / f"seed{seed}"
            print(f"record seed={seed}", flush=True)
            selected_summaries.append(run_episode(seed, record_dir=rec_dir))
        (out_root / "summary_all.json").write_text(
            json.dumps(
                {
                    "created_at": stamp,
                    "elapsed_s": time.time() - started,
                    "out_root": str(out_root),
                    "model_dir": args.model_dir,
                    "scan_count": len(rows),
                    "success_count": len(successes),
                    "selected": selected_summaries,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(json.dumps(
        {
            "out_root": str(out_root),
            "scan_count": len(rows),
            "success_count": len(successes),
            "selected_seeds": [int(r["seed"]) for r in selected],
        },
        indent=2,
        ensure_ascii=False,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
