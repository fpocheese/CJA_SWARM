# V71 10v10 Successful Results

Generated on 2026-06-20 from remote host `192.168.1.91` using user `a2rl`.

## Summary

- Task: collect three successful V71 `10v10` episodes and generate paper figures.
- Remote project: `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`
- Model directory: `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`
- Reward profile: `FOV_REWARD_PROFILE=v69teamsurvive`
- Policy clone map: `agent_id % 4`
- Collector script: `/tmp/run_v71_10v10_collect.py` on the remote host
- Plotter: `plot_all.py`

## Selected Episodes

| Seed | Hit agent | Final step | Final time (s) | Best HVT distance (m) |
| ---: | --- | ---: | ---: | ---: |
| 100007 | [4] | 5192 | 51.92 | 4.805 |
| 100013 | [4] | 5189 | 51.89 | 4.896 |
| 100015 | [8] | 4978 | 49.78 | 4.513 |

## Recorded Configuration

```text
n_offensive = 10
n_defensive = 10
clone_map = {0:0,1:1,2:2,3:3,4:0,5:1,6:2,7:3,8:0,9:1}
success = true
```

## Files

- `seed100007/trajectory_data.npz`
- `seed100007/game_data.npz`
- `seed100007/summary.json`
- `seed100013/trajectory_data.npz`
- `seed100013/game_data.npz`
- `seed100013/summary.json`
- `seed100015/trajectory_data.npz`
- `seed100015/game_data.npz`
- `seed100015/summary.json`
- `scan_all.csv`
- `summary_all.json`
- `plot_all.py`

## Plot Output

Each seed directory contains the standard 16 PDFs:

- `fig1a_speed.pdf`
- `fig1b_pitch_overload.pdf`
- `fig1c_yaw_overload.pdf`
- `fig1d_distance.pdf`
- `fig2a_gantt.pdf`
- `fig2b_assignment_cost.pdf`
- `fig2c_ratio.pdf`
- `fig2d_pitch_overload.pdf`
- `fig2e_yaw_overload.pdf`
- `fig3a_role_prob.pdf`
- `fig3b_lock_pressure.pdf`
- `fig3c_phi_neff.pdf`
- `fig3d_pen_prob.pdf`
- `fig3e_gamma_xi.pdf`
- `fig3f_hit_escape.pdf`
- `fig4_traj3d.pdf`

Total PDFs generated: 48.

## Notes For Future Work

- The remote default `python3` did not have `torch`; the working interpreter
  was `/home/a2rl/miniconda3/envs/rlgpu/bin/python`.
- The dedicated 10v10 collector was run without modifying the remote repo.
- The scan stopped after finding three successes in the seed window
  `100000..100199`.

