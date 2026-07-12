#!/usr/bin/env python
"""Monte Carlo evaluation for selected paper-baseline attack methods.

Runs lightweight episode statistics only. It deliberately does not save full
trajectories or overwrite the single-case reproduction folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_paper_baseline_methods import CASES, METHODS, make_env


_METHOD = None
_CASE = None
_MAX_STEPS = None
_JITTER = None
_ENV = None


def init_worker(method: str, case: str, max_steps: int, jitter: dict):
    global _METHOD, _CASE, _MAX_STEPS, _JITTER, _ENV
    _METHOD = method
    _CASE = case
    _MAX_STEPS = int(max_steps)
    _JITTER = dict(jitter)
    meta = CASES[case]
    _ENV = make_env(int(meta["n"]), int(meta["seed"]))


def perturb_initial_state(env, seed: int):
    """Add MC randomness beyond the scenario RNG, including speed jitter."""
    rng = np.random.RandomState(int(seed) ^ 0x5EEDC0DE)
    pos_xy = float(_JITTER.get("pos_xy_m", 35.0))
    pos_z = float(_JITTER.get("pos_z_m", 8.0))
    heading = float(_JITTER.get("heading_rad", 0.06))
    gamma = float(_JITTER.get("gamma_rad", 0.015))
    speed_frac = float(_JITTER.get("speed_frac", 0.035))

    for ac in list(env.offensives) + list(env.defensives):
        ac.x += rng.uniform(-pos_xy, pos_xy)
        ac.y += rng.uniform(-pos_xy, pos_xy)
        ac.z = float(np.clip(ac.z + rng.uniform(-pos_z, pos_z),
                             env.config["z_min"], env.config["z_max"]))
        ac.heading = float(np.arctan2(
            np.sin(ac.heading + rng.uniform(-heading, heading)),
            np.cos(ac.heading + rng.uniform(-heading, heading)),
        ))
        ac.gamma = float(np.clip(ac.gamma + rng.uniform(-gamma, gamma),
                                 env.config.get("gamma_min", -0.7),
                                 env.config.get("gamma_max", 0.7)))
        ac.v = float(np.clip(ac.v * (1.0 + rng.uniform(-speed_frac, speed_frac)),
                             ac.params["v_min"], ac.params["v_max"]))


def run_one(ep_idx: int) -> dict:
    method = _METHOD
    case = _CASE
    meta = CASES[case]
    seed = int(meta["seed"]) + int(ep_idx)
    n = int(meta["n"])
    env = _ENV
    if env is None:
        env = make_env(n, seed)
    env.seed(seed)
    env.reset()
    perturb_initial_state(env, seed)

    action_fn = METHODS[method]
    hvt = env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in range(n)]
    final_info = {}
    final_step = 0

    for step in range(_MAX_STEPS):
        actions = action_fn(env)
        _, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}
        for i, off in enumerate(env.offensives):
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
                min_step[i] = final_step
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    hit_step = getattr(env, "hit_indices", [])
    return {
        "method": method,
        "case": case,
        "episode": int(ep_idx),
        "seed": seed,
        "success": int(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ";".join(str(int(i)) for i in getattr(env, "hit_indices", [])),
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_hvt_distance_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
    }


def write_rows_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["episode"]))


def summarize(rows: list[dict], elapsed_s: float, workers: int) -> dict:
    n = len(rows)
    successes = [r for r in rows if int(r["success"]) == 1]
    best = np.asarray([float(r["best_hvt_distance_m"]) for r in rows], dtype=float)
    final_t = np.asarray([float(r["final_time_s"]) for r in rows], dtype=float)
    hit_counts = np.asarray([int(r["hit_count"]) for r in rows], dtype=float)
    reasons = Counter(str(r["done_reason"]) for r in rows)
    success_times = np.asarray([float(r["final_time_s"]) for r in successes], dtype=float)
    return {
        "method": rows[0]["method"],
        "case": rows[0]["case"],
        "n_episodes": n,
        "workers": workers,
        "success_count": len(successes),
        "success_rate": len(successes) / max(n, 1),
        "hit_count_mean": float(hit_counts.mean()),
        "best_hvt_distance_mean_m": float(best.mean()),
        "best_hvt_distance_median_m": float(np.median(best)),
        "best_hvt_distance_min_m": float(best.min()),
        "best_hvt_distance_p05_m": float(np.percentile(best, 5)),
        "best_hvt_distance_p95_m": float(np.percentile(best, 95)),
        "final_time_mean_s": float(final_t.mean()),
        "success_time_mean_s": float(success_times.mean()) if len(success_times) else None,
        "done_reason_counts": dict(reasons),
        "elapsed_s": float(elapsed_s),
    }


def run_condition(method: str, case: str, n_eps: int, max_steps: int, workers: int,
                  out_root: Path, jitter: dict) -> dict:
    out_dir = out_root / f"{method}_{case}_mc{n_eps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict] = []
    print(f"=== MC {method}/{case}: n={n_eps} workers={workers} out={out_dir} ===", flush=True)
    with mp.Pool(processes=workers, initializer=init_worker,
                 initargs=(method, case, max_steps, jitter)) as pool:
        for row in pool.imap_unordered(run_one, range(n_eps), chunksize=1):
            rows.append(row)
            done = len(rows)
            if done <= 5 or done % 25 == 0 or done == n_eps:
                succ = sum(int(r["success"]) for r in rows)
                best_min = min(float(r["best_hvt_distance_m"]) for r in rows)
                print(
                    f"{method}/{case} done={done}/{n_eps} succ={succ} "
                    f"rate={succ / done:.3f} best_min={best_min:.2f}m",
                    flush=True,
                )
    elapsed_s = time.time() - started
    write_rows_csv(out_dir / "episodes.csv", rows)
    summary = summarize(rows, elapsed_s, workers)
    summary["jitter"] = jitter
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["garcia_bddg", "weiyang_ta"])
    parser.add_argument("--cases", nargs="+", default=["caseB_seed50042_torch1", "6v6", "8v8"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-root", default="/tmp/v71_mc_selected_methods_1000")
    parser.add_argument("--pos-xy-m", type=float, default=35.0)
    parser.add_argument("--pos-z-m", type=float, default=8.0)
    parser.add_argument("--heading-rad", type=float, default=0.06)
    parser.add_argument("--gamma-rad", type=float, default=0.015)
    parser.add_argument("--speed-frac", type=float, default=0.035)
    args = parser.parse_args()

    jitter = {
        "pos_xy_m": args.pos_xy_m,
        "pos_z_m": args.pos_z_m,
        "heading_rad": args.heading_rad,
        "gamma_rad": args.gamma_rad,
        "speed_frac": args.speed_frac,
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    started = time.time()
    workers = max(1, int(args.workers))
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        for case in args.cases:
            if case not in CASES:
                raise ValueError(f"unknown case: {case}")
            summary = run_condition(method, case, args.episodes, args.max_steps,
                                    workers, out_root, jitter)
            all_summaries[f"{method}_{case}"] = summary
            with (out_root / "summary_all_partial.json").open("w") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    top = {
        "elapsed_s": time.time() - started,
        "episodes_per_condition": args.episodes,
        "methods": args.methods,
        "cases": args.cases,
        "summaries": all_summaries,
    }
    with (out_root / "summary_all.json").open("w") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    print(f"summary_all={out_root / 'summary_all.json'}", flush=True)


if __name__ == "__main__":
    main()
