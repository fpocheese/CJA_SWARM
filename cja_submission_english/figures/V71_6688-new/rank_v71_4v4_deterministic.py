#!/usr/bin/env python
"""Scan and rank deterministic V71 4v4 success cases by assignment disturbance."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from scripts import collect_v71_4v4_deterministic as collect


def scan_seed(seed: int) -> dict:
    return collect.run_scan_one(seed)


def load_case_metrics(case_dir: Path) -> dict:
    sm = json.loads((case_dir / "summary.json").read_text())
    d = np.load(case_dir / "trajectory_data.npz", allow_pickle=True)
    gd = np.load(case_dir / "game_data.npz", allow_pickle=True)

    hitter = int((sm.get("hit_indices") or [sm.get("best_agent", 0)])[0])
    t = np.asarray(d["time"], dtype=float)
    ht = float(sm.get("final_time_s", t[-1] if len(t) else 0.0))
    ltgt = np.asarray(d["def_ltgt"], dtype=int)
    lmode = np.asarray(d["def_lmode"], dtype=int)
    alive = np.asarray(d["def_alive"], dtype=int)
    valid = (ltgt >= 0) & (alive == 1)
    locked = (lmode == 2) & valid

    if valid.any():
        hitter_share = float(np.mean(ltgt[valid] == hitter))
    else:
        hitter_share = 1.0
    late = (t >= min(10.0, ht * 0.25))[None, :]
    late_mask = valid & late
    late_hitter_share = float(np.mean(ltgt[late_mask] == hitter)) if late_mask.any() else hitter_share
    locked_hitter_share = float(np.mean(ltgt[locked] == hitter)) if locked.any() else hitter_share

    decoy_lock = np.asarray(gd["decoy_lock_pressure"], dtype=float)
    if decoy_lock.ndim >= 2 and decoy_lock.shape[0] > hitter:
        hitter_lock_mean = float(np.nanmean(decoy_lock[hitter]))
        others = [i for i in range(decoy_lock.shape[0]) if i != hitter]
        decoy_lock_mean = float(np.nanmean(decoy_lock[others])) if others else 0.0
    else:
        hitter_lock_mean = 0.0
        decoy_lock_mean = 0.0

    # Prefer low exposure of the hitter, high decoy pressure, later survival, and precise hit.
    score = (
        2.0 * (1.0 - late_hitter_share)
        + 1.0 * (1.0 - hitter_share)
        + 0.5 * max(decoy_lock_mean - hitter_lock_mean, 0.0)
        + 0.2 * min(ht / 50.0, 1.2)
        - 0.02 * float(sm.get("best_hvt_distance_m", 10.0))
    )

    return {
        "case_dir": str(case_dir),
        "seed": int(sm["seed"]),
        "success": bool(sm.get("success", True)),
        "hit_indices": sm.get("hit_indices"),
        "hit_step": sm.get("hit_step"),
        "final_step": int(sm.get("final_step", 0)),
        "best_hvt_distance_m": float(sm.get("best_hvt_distance_m", np.nan)),
        "hitter": hitter,
        "hitter_target_share": hitter_share,
        "late_hitter_target_share": late_hitter_share,
        "locked_hitter_share": locked_hitter_share,
        "hitter_lock_pressure_mean": hitter_lock_mean,
        "decoy_lock_pressure_mean": decoy_lock_mean,
        "score": float(score),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=90000)
    ap.add_argument("--seed-end", type=int, default=90400)
    ap.add_argument("--extra-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--record-top", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-root", default="/tmp/v71_4v4_ranked")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_end)) + [
        s for s in args.extra_seeds if s < args.seed_start or s >= args.seed_end
    ]

    rows = []
    n_workers = max(1, min(args.workers, len(seeds)))
    with mp.Pool(processes=n_workers, initializer=collect.init_worker, initargs=("4v4",)) as pool:
        for idx, row in enumerate(pool.imap_unordered(scan_seed, seeds), 1):
            rows.append(row)
            print(
                f"scan {idx:04d}/{len(seeds):04d} seed={row['seed']} "
                f"success={int(row['success'])} best={row['best_min_dist_m']:.2f} "
                f"agent={row['best_agent']} step={row['final_step']}",
                flush=True,
            )

    scan_csv = out_root / "scan_all.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successes = [r for r in rows if r["success"]]
    successes.sort(key=lambda r: (float(r["best_min_dist_m"]), int(r["final_step"])))
    candidates = successes[: args.record_top]
    # Include known deterministic successes even if they were not in the top precision set.
    by_seed = {int(r["seed"]): r for r in successes}
    for seed in args.extra_seeds:
        if seed in by_seed and all(int(r["seed"]) != seed for r in candidates):
            candidates.append(by_seed[seed])

    metrics = []
    for row in candidates:
        seed = int(row["seed"])
        case_dir = out_root / f"seed{seed}"
        print(f"record seed={seed}", flush=True)
        collect.record_success("4v4", seed, case_dir)
        metrics.append(load_case_metrics(case_dir))

    metrics.sort(key=lambda r: r["score"], reverse=True)
    (out_root / "ranked_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics[: min(10, len(metrics))], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
