#!/usr/bin/env python3
"""Plot the latest full-save evaluation outputs using the existing plot code.

This script does not run simulation and does not rewrite trajectory/game data.
It only prepares a tiny temporary plotting workspace with symlinks to the
selected scenario folders, then executes remote_data_save_code/plot_all.py.

Default input root:
  outputs/fullsave_complete_test

Expected scenario folders:
  scenario_2v2_single_policy
  scenario_3v3_single_policy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = PROJECT_ROOT / "remote_data_save_code" / "plot_all.py"


def _link_scenario(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/fullsave_complete_test",
        help="Root directory containing scenario_2v2_single_policy and scenario_3v3_single_policy",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["scenario_2v2_single_policy", "scenario_3v3_single_policy"],
        help="Scenario folder names to plot",
    )
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"plot script not found: {PLOT_SCRIPT}")

    with tempfile.TemporaryDirectory(prefix="plot_fullsave_", dir=str(PROJECT_ROOT)) as tmpdir:
        tmp_root = Path(tmpdir)
        _link_scenario(PLOT_SCRIPT, tmp_root / "plot_all.py")
        for scen in args.scenarios:
            src = input_root / scen
            if not src.exists():
                raise FileNotFoundError(f"scenario folder not found: {src}")
            _link_scenario(src, tmp_root / scen)

        print(f"[plot-only] plotting from {input_root}")
        subprocess.run([sys.executable, str(tmp_root / "plot_all.py")], cwd=str(tmp_root), check=True)
        print(f"[plot-only] done")


if __name__ == "__main__":
    main()
