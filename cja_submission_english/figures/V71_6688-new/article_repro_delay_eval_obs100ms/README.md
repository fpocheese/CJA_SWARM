# V71 Delay/Noise Evaluation

This directory contains a semi-physical delay evaluation for the three
deterministic V71 cases.

Delay model:

- Offensive policy observation delay: `10` simulation steps = `0.10 s`.
- Action execution delay: `2` simulation steps = `0.02 s`.
- Observation noise: zero-mean Gaussian, `std = 0.005` in observation units.
- Actor rollout: `deterministic=True`.
- Model: `outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models`.

Results:

- `4v4`: seed `90019`, hitter `A0`, hit step `4658`, best distance `4.8397 m`.
- `6v6`: seed `60015`, hitter `A4`, hit step `4952`, best distance `4.7013 m`.
- `8v8`: seed `80047`, hitter `A4`, hit step `5128`, best distance `4.7620 m`.

The delayed runs remain successful under the same seeds as the deterministic
baseline cases. The delay/noise model is implemented in the evaluation wrapper
`collect_v71_delay_eval.py`; it does not overwrite the original environment or
the remote working tree.
