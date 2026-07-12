# V71 latest shape-target selected results

Generated on 2026-06-20 with the latest interceptor strategy and offensive-side HIL environment.

## Model and environment

- Remote repo: `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`
- Model: `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`
- Reward profile: `FOV_REWARD_PROFILE=v69teamsurvive`
- Offensive-side HIL enabled:
  - sensor delay: 4 steps
  - sensor jitter: 2 steps
  - observation noise: 0.0015
  - observation bias: 0.0003
  - observation quantization: 0.001
- Defensive-side HIL disabled: `defense_hil_stats.enabled=false`
- Interceptor target plotting uses `def_current_attack_target`.

## Selected cases

| Case | Seed | Hit agent | Final step | HVT distance (m) | Early main selected | Main unassigned while others targeted (s) |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 4v4 | 92195 | [0] | 4689 | 4.620 | yes | 0.00 |
| 4v4 | 92290 | [0] | 4679 | 4.628 | yes | 0.00 |
| 6v6 | 60139 | [4] | 5018 | 4.786 | yes | 10.35 |
| 6v6 | 60180 | [4] | 4995 | 4.766 | yes | 10.21 |
| 6v6 | 60193 | [0] | 4769 | 4.980 | yes | 37.35 |
| 8v8 | 80132 | [4] | 5095 | 4.598 | yes | 36.42 |
| 8v8 | 80139 | [4] | 5211 | 4.988 | yes | 40.80 |
| 8v8 | 80149 | [4] | 5295 | 4.611 | yes | 42.33 |

Notes:

- Per your instruction, 4v4 uses the two newly found successful cases. A third 4v4 candidate, seed 92764, is retained only in the candidate metrics on the remote side and was not selected for plotting here.
- 6v6 and 8v8 were selected from eight successful candidates per case using the current-attack-target assignment shape score.
- Total generated PDFs: 128.

## Layout

- `selected/4v4/seed92195`
- `selected/4v4/seed92290`
- `selected/6v6/seed60139`
- `selected/6v6/seed60180`
- `selected/6v6/seed60193`
- `selected/8v8/seed80132`
- `selected/8v8/seed80139`
- `selected/8v8/seed80149`
- `candidate_metrics/`
- `source_state/`
