# V71 Offensive-Side HIL Best Cases

Remote code: `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`

Model: `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`

HIL scope: offensive-side perception only. Defender/interceptor logic is the current remote working tree and is not patched by the HIL wrapper.

HIL parameters used for all three cases:

- Sensor latency: 4-6 steps at `dt=0.01 s`, mean about `0.05 s`
- Sensor dropout: `0.0`
- Observation noise std: `0.0015`
- Observation bias std: `0.0003`
- Observation bias random walk std: `0.000003`
- Observation quantization: `0.001`
- Command latency/dropout: disabled
- Actuator lag/rate limit: disabled
- `defense_hil_stats.enabled = false`

Selected cases:

| Case | Seed | Hit agent | Hit step | Hit distance (m) | Offensive alive | Defensive alive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4v4 | 90359 | 0 | 4500 | 4.727 | 3 | 3 |
| 6v6 | 60031 | 4 | 5185 | 4.502 | 3 | 3 |
| 8v8 | 80066 | 4 | 5210 | 4.507 | 4 | 4 |

Selection notes:

- `4v4` was scanned over `90300..90399`; only `90359` succeeded under this HIL setting.
- `6v6` was scanned over `60000..60119`; `60031` ranked highest among recorded successes.
- `8v8` was observed during the interrupted scan and then replayed/recorded directly with seed `80066`.
- The old `4v4` seed `90019` is not used here because, under the current remote interceptor working tree, it fails even with all HIL perturbations disabled.

