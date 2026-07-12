#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_single_algo.py"


BASE_REWARD = {
    "reward_w_dist": 0.10,
    "reward_w_angle": 1.00,
    "reward_w_hit": 1.00,
    "reward_w_coord": 1.00,
    "reward_w_energy": 1.00,
}

BASE_HYPER = {
    "clip_param": 0.10,
    "entropy_coef": 0.01,
    "gae_lambda": 0.95,
    "target_kl": 0.02,
}


def value_id(value):
    text = ("%g" % value).replace("-", "m").replace(".", "p")
    return text


def build_cases(include_reward=True, include_hyper=True):
    cases = [{
        "case_id": "base",
        "group": "base",
        "param": "base",
        "value": "base",
        "args": {},
    }]

    if include_reward:
        reward_grid = {
            "reward_w_dist": [0.05, 0.20],
            "reward_w_angle": [0.50, 2.00],
            "reward_w_hit": [0.50, 2.00],
            "reward_w_coord": [0.50, 2.00],
            "reward_w_energy": [0.50, 2.00],
        }
        for param, values in reward_grid.items():
            for value in values:
                cases.append({
                    "case_id": "%s_%s" % (param.replace("reward_w_", "w_"), value_id(value)),
                    "group": "reward",
                    "param": param,
                    "value": value,
                    "args": {param: value},
                })

    if include_hyper:
        hyper_grid = {
            "clip_param": [0.05, 0.20],
            "entropy_coef": [0.005, 0.02],
            "gae_lambda": [0.90, 0.98],
            "target_kl": [0.01, 0.04],
        }
        for param, values in hyper_grid.items():
            for value in values:
                cases.append({
                    "case_id": "%s_%s" % (param, value_id(value)),
                    "group": "hyper",
                    "param": param,
                    "value": value,
                    "args": {param: value},
                })

    return cases


def command_for_case(case, seed, args, case_dir):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algo", args.algo,
        "--seed", str(seed),
        "--compare_steps", str(args.compare_steps),
        "--save_dir", str(case_dir),
        "--sensitivity_tag", case["case_id"],
    ]

    merged = {}
    merged.update(BASE_REWARD)
    merged.update(BASE_HYPER)
    merged.update(case["args"])
    for key, value in merged.items():
        cmd.extend(["--" + key, str(value)])
    return cmd


def write_case_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "group", "param", "value", "seed", "status", "command"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run one-factor sensitivity experiments for reward weights and PPO hyperparameters."
    )
    parser.add_argument("--algo", default="Advanced-MAPPO",
                        choices=["Advanced-MAPPO", "MAPPO", "IPPO", "IA2C", "IQL"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--compare_steps", type=int, default=150000)
    parser.add_argument("--output_dir", type=str,
                        default=str(SCRIPT_DIR / "results" / "reward_sensitivity"))
    parser.add_argument("--scope", choices=["all", "reward", "hyper"], default="all")
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(
        include_reward=args.scope in ("all", "reward"),
        include_hyper=args.scope in ("all", "hyper"),
    )
    rows = []
    case_csv = output_dir / "cases.csv"

    for case in cases:
        case_dir = output_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            cmd = command_for_case(case, seed, args, case_dir)
            row = {
                "case_id": case["case_id"],
                "group": case["group"],
                "param": case["param"],
                "value": case["value"],
                "seed": seed,
                "status": "planned" if args.dry_run else "running",
                "command": " ".join(cmd),
            }
            rows.append(row)
            write_case_rows(case_csv, rows)

            print("\n[%s seed=%s] %s" % (case["case_id"], seed, row["command"]))
            if args.dry_run:
                continue

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            code = subprocess.call(cmd, cwd=str(SCRIPT_DIR.parent.parent), env=env)
            row["status"] = "done" if code == 0 else "failed:%d" % code
            write_case_rows(case_csv, rows)
            if code != 0:
                print("Case failed: %s seed=%s" % (case["case_id"], seed), file=sys.stderr)

    write_case_rows(case_csv, rows)
    print("\nCase manifest saved to: %s" % case_csv)
    print("Plot with: python %s --input_dir %s" % (SCRIPT_DIR / "plot_reward_sensitivity.py", output_dir))


if __name__ == "__main__":
    main()
