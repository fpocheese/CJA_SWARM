# Consistent Deterministic V71 Cases

This directory is a three-case set with consistent deterministic MAPPO rollout
style:

- `4v4`: deterministic V71 MAPPO, seed `90019`, hitter `A0`, hit step `4657`.
- `6v6`: deterministic V71 MAPPO, seed `60015`, hitter `A4`, hit step `4952`.
- `8v8`: deterministic V71 MAPPO, seed `80047`, hitter `A4`, hit step `5128`.

The previous article 4v4 case `caseB_seed50042_torch1` was reproducible but
used stochastic actor sampling with fixed `torch_seed=1`; this made the
overload curves visibly more jittery. The new 4v4 case uses
`deterministic=True`, matching the 6v6 and 8v8 collection path.

Tradeoff: the original stochastic 4v4 case has stronger target-selection
disturbance of the hitter, while this deterministic 4v4 case has smoother
overload curves and consistent rollout semantics across all three scales.
