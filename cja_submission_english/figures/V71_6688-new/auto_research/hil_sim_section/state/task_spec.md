# Task: HIL simulation section publication iteration

## Objective
Improve the simulation-results section of `/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/newswarm_no_first_order_lag_final_from_complete.tex` to a submission-ready academic style, with emphasis on the HIL experiments under `figures/V71_6688-new/hil_outputs/original_success_5nx_restlocal`.

## Scope
- Section 5.1 HIL platform and non-ideal measurement/communication modeling.
- Representative 6v6, 8v8, and 10v10 HIL case-study analysis.
- Consistency between text, figures, seeds, data files, and physical units.
- Scholarly tone suitable for a top journal.

## Success Criteria
- No report-style wording such as ad hoc hardware counts or implementation notes in the main narrative.
- HIL architecture is described as an experimental methodology, not as a lab log.
- Time delay, noise, bias, and measured physical quantities are stated with units and statistical definitions.
- Case analyses use the same style as the existing manuscript: stage decomposition, target reassignment, role preferences, lock pressure, and allocation-ratio interpretation.
- All numerical claims are traceable to `summary.json`, `trajectory_data.npz`, `game_data.npz`, or figure-generation logic.
- The section can be compiled without obvious LaTeX syntax errors.

## Current Artifacts
- Main manuscript: `/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/newswarm_no_first_order_lag_final_from_complete.tex`
- HIL results: `/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/hil_outputs/original_success_5nx_restlocal`
- Installed protocol skill: `/home/uav/.codex-accounts/work/skills/Deli_AutoResearch/SKILL.md`
