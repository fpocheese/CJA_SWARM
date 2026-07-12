# V71 latest attack-target simulation results

Generated on 2026-06-20 with the latest interceptor target-selection/plotting logic.

## Model and environment

- Remote repo: `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`
- Model: `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`
- Reward profile: `FOV_REWARD_PROFILE=v69teamsurvive`
- Detection/lock range: 500 m
- Max steps: 7000
- Offensive-side HIL is enabled.
- Defensive-side HIL is disabled: `defense_hil_stats.enabled=false`

## Offensive-side HIL configuration

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

## Successful cases

| Case | Seed | Hit agent | Final step | Final time (s) | Best HVT distance (m) | Offensive alive | Defensive alive |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4v4 | 92010 | [0] | 4645 | 46.45 | 4.816 | 4 | 4 |
| 6v6 | 60109 | [4] | 5157 | 51.57 | 4.971 | 3 | 3 |
| 8v8 | 80109 | [4] | 5105 | 51.05 | 4.555 | 3 | 3 |

These seeds are new for this batch and do not reuse the previously selected seeds.

## Target-assignment plotting check

The trajectory and game data include `def_current_attack_target`. For compatibility, `def_ltgt` was written from the same current attack-target series. The local plotting script prefers `def_current_attack_target`, falling back to `def_ltgt` only if needed.

The plotted interceptor target-assignment figures therefore show each interceptor's current selected attack target, not only the lock state.

## Output layout

- `4v4/seed92010/`
- `6v6/seed60109/`
- `8v8/seed80109/`
- `source_state/`

Each seed directory contains the raw `.npz`/`summary.json` files and 16 generated PDF figures.
