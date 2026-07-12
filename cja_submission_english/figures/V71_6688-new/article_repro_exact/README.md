# V71 Article Reproduction Notes

This directory contains the reproduced data and plots for the three cases used
by the current article figures:

- `caseB_seed50042_torch1`: 4v4 article case, env seed 50042, torch seed 1.
- `6v6`: env seed 60015.
- `8v8`: env seed 80047.

The reproduction was generated on `a2rl@192.168.1.91` using the V71 MAPPO
checkpoint:

`outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`

Important code-state note: the current remote working tree has uncommitted
changes in `envs/fov_penetration/policies_interceptor.py`. Those changes make
the interceptor keep normal rear/after-pass guidance capability and do not
reproduce the article trajectories. The data here was regenerated in a
temporary remote worktree using the current V71 observation/reward environment
plus the committed `eeb63ac` version of `policies_interceptor.py`. No current
uncommitted remote changes were deleted or rolled back.
