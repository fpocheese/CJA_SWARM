#!/usr/bin/env python
"""Merge sharded v69 Monte Carlo evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v69_monte_carlo")
    parser.add_argument("--tag-prefix", default="paper_mc1000_s")
    parser.add_argument("--scenario-case", default="baseline")
    parser.add_argument("--expected", type=int, default=1000)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def as_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def wilson_interval(success: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = success / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(rows: list[dict], shard_dirs: list[Path], args: argparse.Namespace) -> dict:
    total = len(rows)
    success = sum(to_int(row.get("success")) for row in rows)
    hits = sum(to_int(row.get("hit_count")) for row in rows)
    best_values = [to_float(row.get("best_min_dist_m")) for row in rows]
    best_values = [value for value in best_values if not math.isnan(value)]
    seeds = [to_int(row.get("seed")) for row in rows]
    success_rate = success / total if total else 0.0
    normal_se = math.sqrt(success_rate * (1.0 - success_rate) / total) if total else 0.0
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "tag_prefix": args.tag_prefix,
        "scenario_case": args.scenario_case,
        "expected_episodes": args.expected,
        "episodes_completed": total,
        "success_count": success,
        "failure_count": total - success,
        "total_hits": hits,
        "success_rate": success_rate,
        "success_rate_ci95_normal": [
            max(0.0, success_rate - 1.96 * normal_se),
            min(1.0, success_rate + 1.96 * normal_se),
        ],
        "success_rate_ci95_wilson": wilson_interval(success, total),
        "best_min_dist_m": min(best_values) if best_values else None,
        "mean_best_min_dist_m": sum(best_values) / len(best_values) if best_values else None,
        "median_best_min_dist_m": sorted(best_values)[len(best_values) // 2] if best_values else None,
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "unique_seed_count": len(set(seeds)),
        "complete": total >= args.expected and len(set(seeds)) >= args.expected,
        "shard_dirs": [str(path.relative_to(PROJECT_ROOT)) for path in shard_dirs],
    }


def main():
    args = parse_args()
    root = as_path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing MC root: {root}")

    shard_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir()
        and args.tag_prefix in path.name
        and path.name.endswith(f"_{args.scenario_case}")
        and (path / "episodes.csv").exists()
    )
    if not shard_dirs:
        raise SystemExit(f"No shard episodes.csv files found under {root}")

    rows_by_seed: dict[int, dict] = {}
    for shard_dir in shard_dirs:
        for row in read_rows(shard_dir / "episodes.csv"):
            seed = to_int(row.get("seed"), default=-1)
            if seed >= 0:
                row["source_dir"] = str(shard_dir.relative_to(PROJECT_ROOT))
                rows_by_seed[seed] = row

    rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
    out_dir = as_path(args.out_dir) if args.out_dir else root / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag_prefix.rstrip('_')}_{args.scenario_case}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows, shard_dirs, args)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    latest_link = root / f"latest_merged_{args.scenario_case}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(out_dir.resolve())
    print(json.dumps({"out_dir": str(out_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()