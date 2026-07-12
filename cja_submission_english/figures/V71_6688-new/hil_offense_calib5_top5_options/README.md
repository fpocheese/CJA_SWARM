# V71 offense-side HIL top-5 options

Generated on 2026-06-19 from the remote server `a2rl@192.168.1.91`.

Model:

- `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`

Environment/code state:

- Remote repo: `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`
- Reward profile: `FOV_REWARD_PROFILE=v69teamsurvive`
- Current interceptor policy changes were preserved.
- Defense-side HIL is disabled in every recorded summary: `defense_hil_stats.enabled=false`.

Offense-side HIL settings:

- `sensor_sample_steps=1`
- `sensor_delay_steps=4`
- `sensor_jitter_steps=2`
- `sensor_dropout_prob=0.0`
- `obs_noise_std=0.0015`
- `obs_bias_std=0.0003`
- `obs_bias_rw_std=0.000003`
- `obs_quant_step=0.001`
- `policy_sample_steps=1`
- `command_delay_steps=0`
- `command_jitter_steps=0`
- `command_dropout_prob=0.0`
- `action_quant_step=0.001`
- `actuator_tau_s=0.0`
- `action_rate_limit_per_s=0.0`
- `max_steps=7000`

Each candidate directory contains `trajectory_data.npz`, `game_data.npz`, `summary.json`, and the 16 generated PDF panels.

## 4v4

| Option | Seed | Hit agent | Hit step | Hit time (s) | Best HVT distance (m) | Alive off/def |
|---|---:|---:|---:|---:|---:|---:|
| `4v4/case01_seed90543` | 90543 | 0 | 4456 | 44.56 | 4.581 | 4/4 |
| `4v4/case02_seed90359` | 90359 | 0 | 4500 | 45.00 | 4.727 | 3/3 |
| `4v4/case03_seed90472` | 90472 | 0 | 4683 | 46.83 | 4.975 | 4/4 |
| `4v4/case04_seed91527` | 91527 | 0 | 4947 | 49.47 | 4.737 | 3/3 |
| `4v4/case05_seed90748` | 90748 | 0 | 4610 | 46.10 | 4.983 | 3/3 |

Remote sources:

- `/tmp/v71_hil_offense_calib5_scan_current/20260619_212648_hil_realism/4v4/seed90359`
- `/tmp/v71_hil_offense_calib5_top5_4v4_extra/20260619_223205_hil_realism/4v4/seed90543`
- `/tmp/v71_hil_offense_calib5_top5_4v4_extra/20260619_223205_hil_realism/4v4/seed90472`
- `/tmp/v71_hil_offense_calib5_top5_4v4_extra/20260619_223205_hil_realism/4v4/seed91527`
- `/tmp/v71_hil_offense_calib5_top5_4v4_extra/20260619_223205_hil_realism/4v4/seed90748`

## 6v6

| Option | Seed | Hit agent | Hit step | Hit time (s) | Best HVT distance (m) | Alive off/def |
|---|---:|---:|---:|---:|---:|---:|
| `6v6/case01_seed60031` | 60031 | 4 | 5185 | 51.85 | 4.502 | 3/3 |
| `6v6/case02_seed60081` | 60081 | 4 | 5118 | 51.18 | 4.767 | 3/3 |
| `6v6/case03_seed60030` | 60030 | 4 | 5162 | 51.62 | 4.711 | 3/3 |
| `6v6/case04_seed60050` | 60050 | 4 | 5113 | 51.13 | 4.866 | 3/3 |
| `6v6/case05_seed60097` | 60097 | 4 | 5183 | 51.83 | 4.779 | 3/3 |

Remote source:

- `/tmp/v71_hil_offense_calib5_scan_current/20260619_212648_hil_realism/6v6`

## 8v8

| Option | Seed | Hit agent | Hit step | Hit time (s) | Best HVT distance (m) | Alive off/def |
|---|---:|---:|---:|---:|---:|---:|
| `8v8/case01_seed80017` | 80017 | 4 | 5373 | 53.73 | 4.606 | 5/5 |
| `8v8/case02_seed80005` | 80005 | 4 | 5307 | 53.07 | 4.718 | 4/4 |
| `8v8/case03_seed80009` | 80009 | 4 | 5056 | 50.56 | 4.644 | 4/4 |
| `8v8/case04_seed80003` | 80003 | 4 | 5404 | 54.04 | 4.791 | 3/3 |
| `8v8/case05_seed80023` | 80023 | 0 | 4774 | 47.74 | 4.750 | 3/3 |

Remote source:

- `/tmp/v71_hil_offense_calib5_top5_8v8/20260619_222111_hil_realism/8v8`
