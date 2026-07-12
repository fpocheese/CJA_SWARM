#!/usr/bin/env python3
"""Plot all successful mixed HIL results saved in this directory."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PLOT = REPO / 'swarm_attack_v2' / 'hil_v71_split' / 'plot_hil_result.py'
PYTHON = '/home/uav/anaconda3/envs/rlgpu/bin/python'

SUCCESS = [
    ('6v6', 60031),
    ('8v8', 80009),
    ('10v10', 100005),
]


def main() -> int:
    made = []
    for case, seed in SUCCESS:
        summary = ROOT / f'v71_hil_{case}_2nx_restlocal_seed{seed}_summary.json'
        trajectory = ROOT / f'v71_hil_{case}_2nx_restlocal_seed{seed}_trajectory.npz'
        if not summary.exists() or not trajectory.exists():
            print(f'skip {case} seed={seed}: missing data')
            continue
        with summary.open() as f:
            doc = json.load(f)
        item = doc['summaries'][0]
        if not item.get('success'):
            print(f'skip {case} seed={seed}: not success')
            continue
        out = ROOT / f'{case}_seed{seed}_hil_result.png'
        cmd = [PYTHON, str(PLOT), '--trajectory', str(trajectory), '--summary', str(summary), '--out', str(out), '--title', f'V71 HIL {case} 2NX+rest-local MAPPO']
        subprocess.run(cmd, check=True)
        made.append(out)
        print(f'{case} seed={seed}: {out}')
    print('\nGenerated:')
    for p in made:
        print(p)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
