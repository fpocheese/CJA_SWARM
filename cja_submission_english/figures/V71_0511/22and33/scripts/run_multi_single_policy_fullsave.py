#!/usr/bin/env python3
"""End-to-end workflow for multi-single-policy evaluation.

This is the professional one-command entrypoint for the current pipeline:
  1) run 2v2 / 3v3 evaluation
  2) save trajectory_data.npz, game_data.npz, summary.json during simulation
  3) draw fig1_offensive.pdf, fig2_defensive.pdf, fig3_game.pdf, fig4_traj3d.pdf
  4) verify the output artifacts exist

It intentionally does not contain any post-processing patch logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "results" / "fov_penetration" / "macpo" / "v22_1v1_penetration" / "run1" / "models"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "multi_single_policy_eval"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _check_artifacts(output_root: Path) -> None:
    required = [
        "trajectory_data.npz",
        "game_data.npz",
        "summary.json",
        "fig1_offensive.pdf",
        "fig2_defensive.pdf",
        "fig3_game.pdf",
        "fig4_traj3d.pdf",
    ]
    for scen in ["scenario_2v2_single_policy", "scenario_3v3_single_policy"]:
        scen_dir = output_root / scen
        missing = [name for name in required if not (scen_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in {scen_dir}: {missing}")
        print(f"[pipeline] verified {scen_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=8)
    parser.add_argument("--seed_start", type=int, default=300)
    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    model_dir = Path(args.model_dir).resolve()

    print(f"[pipeline] model_dir={model_dir}")
    print(f"[pipeline] output={output_root}")

    _run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_multi_single_policy_fullsave.py"),
        "--model_dir",
        str(model_dir),
        "--output",
        str(output_root),
        "--hidden_size",
        str(args.hidden_size),
        "--layer_N",
        str(args.layer_N),
        "--n_episodes",
        str(args.n_episodes),
        "--seed_start",
        str(args.seed_start),
    ])

    _run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_multi_single_policy_fullsave.py"),
        "--input_root",
        str(output_root),
    ])

    _check_artifacts(output_root)
    print("[pipeline] all artifacts ready")


if __name__ == "__main__":
    main()
