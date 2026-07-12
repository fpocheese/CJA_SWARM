#!/usr/bin/env python3
"""Clean entrypoint for 2v2 / 3v3 single-policy evaluation.

This script reuses the evaluation logic from eval_multi_single_policy.py and
keeps the workflow source-level only:
  1) run the simulation
  2) save summary.json, trajectory_data.npz, game_data.npz directly during the run
  3) generate the evaluation plots

Use this file as the preferred entrypoint going forward.
"""

from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_multi_single_policy import (  # noqa: E402
    evaluate_scenario,
    load_single_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/results/fov_penetration/macpo/v22_1v1_penetration/run1/models",
        help="Directory containing actor_agent0.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/multi_single_policy_eval",
        help="Output root for scenario_2v2_single_policy and scenario_3v3_single_policy",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=8)
    parser.add_argument("--seed_start", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    import torch

    device = torch.device("cpu")
    policy, ref_env = load_single_policy(args.model_dir, args.hidden_size, args.layer_N, device)
    print(f"[fullsave] Loaded single-attacker actor from {args.model_dir}")

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
