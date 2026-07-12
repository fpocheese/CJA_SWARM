# V71 Swarm Simulation AI Handoff

Last updated: 2026-06-20 (Asia/Shanghai)

This document is the operational handoff for the V71 multi-UAV penetration
simulation, HIL experiments, remote execution, result collection, and plotting.
It is intended to let a new AI continue the work without reconstructing the
project context from chat history.

## 1. Current User Goal

The active request is:

1. The existing 4v4 result is not satisfactory.
2. Run and find three successful 10v10 episodes.
3. Copy the three complete result directories back to this local workspace.
4. Generate the standard paper figures for all three episodes.
5. Preserve enough documentation for another AI to continue immediately.

Current status:

- Local code and prior result structure have been inspected.
- The local directory is primarily a paper/result workspace. It does not contain
  the complete simulation project, `envs/`, `scripts/`, MACPO source, or model
  checkpoints required to run a new episode.
- The complete project was previously used remotely at:
  `/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2`
- On 2026-06-20, connection attempts to all seven known EC2 hosts reached port
  22 but were closed during SSH key exchange. No 10v10 run has completed yet.
- The 10v10 simulation has now been completed on `192.168.1.91` and copied back
  locally. The result root is:
  `10v10_success_20260620/20260620_171139_10v10`

Selected successful seeds:

- `100007`
- `100013`
- `100015`

The handoff document itself is complete, and the remaining work is only to
inspect, reuse, or extend the existing 10v10 outputs.

Do not report the 10v10 task as complete until all three local result directories
contain `trajectory_data.npz`, `game_data.npz`, `summary.json`, and generated
PDF figures, and every `summary.json` has `"success": true`.

## 2. Local Workspace

Current local workspace:

```text
/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new
```

Important local files:

| Path | Purpose |
| --- | --- |
| `plot_all.py` | Main IEEE-style figure generator. |
| `hil_realism_v71_scan.py` | HIL nonideality wrapper, seed scanner, recorder, and candidate ranker. |
| `collect_v71_delay_eval.py` | Simpler fixed sensor/action delay evaluator. |
| `hil_v71_split/` | TCP-separated environment server and per-aircraft policy-node prototype. |
| `shape_target_selected_20260620_163320/` | Best documented current V71 results and source snapshots. |
| `shape_target_selected_20260620_163320/source_state/` | Frozen copies of critical remote source files. |
| `repro_success_4v4_6v6_8v8/` | Earlier successful 4v4, 6v6, and 8v8 result format. |

The git worktree contains many unrelated changes outside this directory. Do not
revert or overwrite them. Restrict new work to this workspace unless the user
explicitly requests publication to the paper directory.

## 3. Verified Remote Project and Runtime

Previously verified remote repository:

```text
/home/a2rl/000000GSY_mutiUAV/swarm_attack_v2
```

Known Python interpreter from the split-HIL documentation:

```text
~/miniconda3/envs/rlgpu/bin/python
```

The repository must contain at least:

```text
envs/fov_penetration.py
scripts/collect_v71_4v4_deterministic.py
scripts/phase_obs_wrapper.py
scripts/terminal_pn_action_wrapper.py
third_party/MACPO/MACPO/
outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models/
```

Before running anything, verify the actual remote user and repository:

```bash
hostname
whoami
pwd
find /home -maxdepth 4 -type d -name swarm_attack_v2 2>/dev/null
```

The current local snapshots say the owner/path is `/home/a2rl/...`, while the
historical SSH commands use user `ubuntu`. Therefore always locate the project
after login instead of assuming `~/000000GSY_mutiUAV/swarm_attack_v2`.

## 4. Known Remote Hosts and SSH

Historical host list:

```text
ec2-44-192-55-188.compute-1.amazonaws.com
ec2-3-89-68-252.compute-1.amazonaws.com
ec2-44-201-35-158.compute-1.amazonaws.com
ec2-3-237-82-245.compute-1.amazonaws.com
ec2-44-204-219-14.compute-1.amazonaws.com
ec2-98-81-99-73.compute-1.amazonaws.com
ec2-44-204-202-166.compute-1.amazonaws.com
```

Historical key and login:

```bash
ssh -i ~/.ssh/gsy_keys -o BatchMode=yes -o ConnectTimeout=8 ubuntu@HOST
```

Current failure observed on 2026-06-20:

```text
kex_exchange_identification: Connection closed by remote host
Connection closed by HOST port 22
```

This is not a Python or repository error. It occurs before authentication.
Likely causes include stopped/replaced instances, stale public DNS names, an
SSH/security-group restriction, or server-side connection filtering. Ask the
user for current instance addresses or to start/authorize the instances if the
same failure persists.

New reachable host supplied by the user:

```text
192.168.1.91
```

Login user:

```text
a2rl
```

Use the supplied password out of band. Do not store it in the repository or in
this document.

Safe host probe:

```bash
for h in HOST1 HOST2 HOST3; do
  echo "=== $h ==="
  ssh -i ~/.ssh/gsy_keys -o BatchMode=yes -o ConnectTimeout=8 "ubuntu@$h" \
    'hostname; whoami; find /home -maxdepth 4 -type d -name swarm_attack_v2 2>/dev/null'
done
```

Do not place private-key contents, passwords, or cloud credentials in this file.

## 5. Core Evaluation Chain

The verified V71 closed loop is:

```text
FOVPenetrationEnv
  -> PhaseMaskedFOVWrapper(mode="v65_strict_los")
  -> TerminalPNActionWrapper(gain=3.0, max_action=0.8)
  -> deterministic recurrent MAPPO actor actions
  -> environment step, defender logic, dynamics, hit/kill logic
```

Environment construction:

```python
os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
raw_env = FOVPenetrationEnv(
    config={"n_offensive": n_off, "n_defensive": n_def},
    scenario="scenario_1",
)
env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
```

Important dimensions and policy defaults:

```text
observation dimension: 30
shared observation dimension: 77
action dimension: 3
RNN hidden size: 256
MAPPO layer count: 3
default maximum episode length: 8000 steps
environment dt: 0.01 s in existing results
```

Success criterion:

```python
success = bool(raw_env.hit_count > 0)
```

Typical successful termination:

```json
{
  "success": true,
  "hit_count": 1,
  "done_reason": "success"
}
```

## 6. Policies and Model Checkpoints

Verified model directory, relative to the remote project root:

```text
outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models
```

Override variable:

```bash
export V71_MODEL_DIR=/absolute/path/to/models
```

The V71 run has four trained offensive actor checkpoints:

```text
actor_agent0.pt
actor_agent1.pt
actor_agent2.pt
actor_agent3.pt
```

There is no separately trained actor for offensive agents 4 through 9. Larger
formations clone the four actors cyclically:

```python
source_agent = agent_id % 4
actor_path = MODEL_DIR / f"actor_agent{source_agent}.pt"
```

Therefore the intended 10v10 clone map is:

```json
{
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 0,
  "5": 1,
  "6": 2,
  "7": 3,
  "8": 0,
  "9": 1
}
```

Each policy is an `R_MAPPOPolicy`. Only the actor state is loaded for evaluation:

```python
policy.actor.load_state_dict(
    torch.load(actor_path, map_location="cpu"),
    strict=False,
)
policy.actor.eval()
```

Actions are deterministic and recurrent. Maintain one RNN state and mask per
offensive aircraft. Reset the mask to zero for a done aircraft.

## 7. Important Source Modules

Remote project modules:

| Module | Responsibility |
| --- | --- |
| `envs/fov_penetration.py` | Aircraft environment, offensive/defensive populations, HVT, reward, termination, hit count. |
| `envs/components/policies_interceptor.py` or its actual repository equivalent | Defender target assignment, FOV lock, target switching, known target state, interceptor guidance. Locate with `find envs -name 'policies_interceptor.py'`. |
| `scripts/phase_obs_wrapper.py` | Applies phase-aware masked observations used by V71. |
| `scripts/terminal_pn_action_wrapper.py` | Adds terminal proportional-navigation behavior before the environment step. |
| `scripts/collect_v71_4v4_deterministic.py` | Model loading, seed scanning, result recording, and NPZ serialization. |
| `third_party/MACPO/MACPO/macpo/...` | MAPPO policy implementation and argument definitions. |

Local source snapshots:

| Snapshot | Use |
| --- | --- |
| `shape_target_selected_20260620_163320/source_state/collect_v71_4v4_deterministic.py` | Best reference for collector behavior and data schema. |
| `shape_target_selected_20260620_163320/source_state/hil_realism_v71_scan.py` | Best reference for the HIL scan configuration used in selected results. |
| `shape_target_selected_20260620_163320/source_state/policies_interceptor.py` | Snapshot of current defender attack-target behavior. |
| `uncommitted_interceptor_v71_success_20260620/source_state/worktree.diff` | Previous uncommitted interceptor changes; inspect before replacing remote code. |

Important defender target fields:

```text
initial_assigned_target_idx
current_locked_target_idx
current_attack_target_idx
assigned_target_idx
```

For plotting actual interceptor behavior, use `current_attack_target_idx`.
Existing recorders store it as:

```text
def_current_attack_target
```

Do not silently replace it with only the initial assignment. The latest plots
and candidate selection intentionally use the current attack target.

## 8. HIL Simulation Options

There are two different HIL approaches in this workspace.

### 8.1 In-process nonideality wrapper

File:

```text
hil_realism_v71_scan.py
```

It models:

- Sensor sampling.
- Base sensor delay and random jitter.
- Sensor packet dropout with sample hold.
- Observation noise, fixed bias, bias random walk, and quantization.
- Policy/controller sample rate.
- Command delay, jitter, dropout, and quantization.
- First-order actuator lag and action-rate limiting.
- Optional defender-side sensing/target-manager nonidealities.

The most recent selected-result configuration was intentionally mild and used
offensive-side HIL only:

```text
sensor_sample_steps=1
sensor_delay_steps=4
sensor_jitter_steps=2
sensor_dropout_prob=0
obs_noise_std=0.0015
obs_bias_std=0.0003
obs_bias_rw_std=0.000003
obs_quant_step=0.001
policy_sample_steps=1
command_delay_steps=0
command_jitter_steps=0
command_dropout_prob=0
action_quant_step=0.001
actuator_tau_s=0
action_rate_limit_per_s=0
enable_defense_hil=false
max_steps=7000
```

This exact configuration should be the default for the requested 10v10 results
unless the user asks for a harsher HIL setting. It is the closest match to the
latest accepted 4v4/6v6/8v8 result family.

### 8.2 TCP-separated split HIL

Directory:

```text
hil_v71_split/
```

Components:

- `hil_env_server.py`: owns environment, defenders, dynamics, PN wrapper,
  observations, terminal logic, and episode summary.
- `hil_policy_node.py`: owns one offensive recurrent actor.
- `hil_protocol.py`: JSON-lines TCP protocol.
- `fixed_initial_state.py`: resets from the first frame of an existing NPZ.
- `eval_v71_fixed_initial.py`: single-process fixed-initial-state evaluator.

The server sends each policy node a masked 30-D observation and receives one
raw 3-D action. The server applies the terminal PN wrapper.

For 10v10, add:

```python
CASES["10v10"] = (10, 10, 100000)
```

Then launch 10 policy nodes, each using:

```bash
--source-agent "$((agent_id % 4))"
```

This split prototype is useful for a real NX/embedded deployment, but the
in-process HIL scanner is faster for finding three successful random seeds.

## 9. Required 10v10 Code Extension

The current collector and HIL CLI explicitly recognize only 4v4, 6v6, and 8v8.
Do not change the generic training scenario registry unless necessary. Extend
the dedicated evaluation scripts only.

Required edits in the remote collector:

```python
def parse_case(case: str) -> tuple[int, int]:
    token = case.strip().lower()
    if token == "4v4":
        return 4, 4
    if token == "6v6":
        return 6, 6
    if token == "8v8":
        return 8, 8
    if token == "10v10":
        return 10, 10
    raise ValueError(...)
```

Add a CLI seed base:

```python
parser.add_argument(
    "--seed-base-10v10",
    type=int,
    default=int(os.environ.get("V71_10V10_SEED_BASE", "100000")),
)
```

Add the case selection branch:

```python
elif token == "10v10":
    seed_base = args.seed_base_10v10
```

Required edits in `hil_realism_v71_scan.py`:

```python
parser.add_argument("--seed-start-10v10", type=int, default=100000)
parser.add_argument("--seed-end-10v10", type=int, default=101000)
```

And:

```python
elif case == "10v10":
    start, end = args.seed_start_10v10, args.seed_end_10v10
```

The environment itself is already constructed from numeric `n_offensive` and
`n_defensive`, so 10v10 should not require a new trained model or a global
scenario definition. Verify this with a one-seed smoke test before a long scan.

The standard plotter has 12 offensive and 12 defensive colors and derives
population sizes from NPZ array shapes, so it should support 10v10 without a
structural change. Check legends visually because 20 trajectories are dense.

## 10. Recommended 10v10 Execution Procedure

### Step 1: Verify source and preserve state

On the remote host:

```bash
cd /home/a2rl/000000GSY_mutiUAV/swarm_attack_v2
git status --short
git rev-parse HEAD
```

Do not discard uncommitted changes. Copy critical files into the output
directory before running:

```bash
mkdir -p /tmp/v71_10v10_success/source_state
cp scripts/collect_v71_4v4_deterministic.py /tmp/v71_10v10_success/source_state/
cp /path/to/hil_realism_v71_scan.py /tmp/v71_10v10_success/source_state/
cp "$(find envs -name policies_interceptor.py -print -quit)" \
  /tmp/v71_10v10_success/source_state/
git diff > /tmp/v71_10v10_success/source_state/worktree.diff
git rev-parse HEAD > /tmp/v71_10v10_success/source_state/git_commit.txt
```

### Step 2: Smoke-test one 10v10 episode

Use one worker and one or two seeds. Confirm:

- Environment creates 10 offensive and 10 defensive aircraft.
- Ten actor policies load using the cyclic clone map.
- Observation and action shapes are valid.
- The episode reaches a normal terminal reason.
- The recorder writes NPZ arrays with first dimension 10.

### Step 3: Parallel seed scan

Preferred strategy:

- Use three working hosts if available.
- Assign non-overlapping seed ranges such as
  `100000-100999`, `101000-101999`, and `102000-102999`.
- Use 8 to 12 workers per host, adjusted to CPU and memory.
- Stop each host after at least one success, or let one host continue until the
  global success count reaches three.
- Record more than three successes if inexpensive, then select the best three.

Use the latest mild offensive HIL configuration from Section 8.1.

Conceptual command after adding 10v10 CLI support:

```bash
cd /home/a2rl/000000GSY_mutiUAV/swarm_attack_v2
export FOV_REWARD_PROFILE=v69teamsurvive
export V71_MODEL_DIR=outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

~/miniconda3/envs/rlgpu/bin/python /path/to/hil_realism_v71_scan.py \
  --cases 10v10 \
  --seed-start-10v10 100000 \
  --seed-end-10v10 101000 \
  --workers 10 \
  --record-top 6 \
  --stop-success-count 3 \
  --out-root /tmp/v71_10v10_success \
  --sensor-sample-steps 1 \
  --sensor-delay-steps 4 \
  --sensor-jitter-steps 2 \
  --sensor-dropout-prob 0 \
  --obs-noise-std 0.0015 \
  --obs-bias-std 0.0003 \
  --obs-bias-rw-std 0.000003 \
  --obs-quant-step 0.001 \
  --policy-sample-steps 1 \
  --command-delay-steps 0 \
  --command-jitter-steps 0 \
  --command-dropout-prob 0 \
  --action-quant-step 0.001 \
  --actuator-tau-s 0 \
  --action-rate-limit-per-s 0 \
  --max-steps 7000
```

Important multiprocessing behavior: when `--stop-success-count` terminates a
pool, some already-running workers may not return their rows. This is acceptable
for fast discovery, but use `scan_all.csv` as the record of returned episodes.

### Step 4: Select and record three successes

Each selected seed must be rerun with recording enabled. A scan-only row is not
enough for plotting. Required files per selected case:

```text
trajectory_data.npz
game_data.npz
summary.json
```

Recommended local layout:

```text
10v10_success_20260620/
  README.md
  summary_all.json
  source_state/
  seed100123/
    trajectory_data.npz
    game_data.npz
    summary.json
  seed100456/
    trajectory_data.npz
    game_data.npz
    summary.json
  seed100789/
    trajectory_data.npz
    game_data.npz
    summary.json
```

Quality preference among successful episodes:

1. `success == true`.
2. Lower `best_hvt_distance_m`.
3. More offensive survivors.
4. Clear defender assignment/decoy behavior.
5. Readable trajectories without obvious numerical discontinuities.

Do not fabricate or manually edit success fields.

## 11. Copy Results Back to Local

Create the local destination first:

```bash
mkdir -p \
  /home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/10v10_success_20260620
```

Copy the selected remote output root:

```bash
scp -i ~/.ssh/gsy_keys -r \
  ubuntu@HOST:/tmp/v71_10v10_success/SELECTED_RUN_ROOT/* \
  /home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/10v10_success_20260620/
```

If the remote project belongs to user `a2rl` but login is `ubuntu`, ensure the
result directory is readable by the SSH login user before copying.

After transfer, verify:

```bash
find 10v10_success_20260620 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
```

And validate summaries:

```bash
find 10v10_success_20260620 -name summary.json -print0 |
  xargs -0 -n1 python3 -c \
  'import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(p,d["success"],d["seed"],d["hit_count"])'
```

## 12. Plotting

The main local plotter is:

```text
plot_all.py
```

It scans every immediate child directory beside the script and processes a
directory only when `trajectory_data.npz` exists. It reads:

```text
trajectory_data.npz
summary.json
game_data.npz (optional, but required for complete game panels)
```

It generates:

```text
fig1a_speed.pdf
fig1b_pitch_overload.pdf
fig1c_yaw_overload.pdf
fig1d_distance.pdf
fig2a_gantt.pdf
fig2b_assignment_cost.pdf
fig2c_ratio.pdf
fig2d_pitch_overload.pdf
fig2e_yaw_overload.pdf
fig3a_role_prob.pdf
fig3b_lock_pressure.pdf
fig3c_phi_neff.pdf
fig3d_pen_prob.pdf
fig3e_gamma_xi.pdf
fig3f_hit_escape.pdf
fig4_traj3d.pdf
```

The easiest safe plotting method is to copy `plot_all.py` into the selected
output root and run it there:

```bash
cp plot_all.py 10v10_success_20260620/plot_all.py
./.venv/bin/python 10v10_success_20260620/plot_all.py
```

Alternatively run it remotely if the same dependencies are available, then
copy both NPZ/JSON data and PDFs back.

Dependencies include NumPy, Matplotlib, and SciPy.

Verification:

```bash
find 10v10_success_20260620 -name '*.pdf' | sort
```

Expected count for three complete episodes:

```text
3 episodes x 16 PDFs = 48 PDFs
```

Inspect at least `fig4_traj3d.pdf`, `fig2a_gantt.pdf`, and
`fig3b_lock_pressure.pdf` for each episode. A successful script exit does not
guarantee readable 10-aircraft legends.

## 13. NPZ Data Schema

`trajectory_data.npz` contains time-series dynamics and assignments:

```text
steps, time, actor_actions
off_x, off_y, off_z, off_v
off_heading, off_gamma
off_an_pitch, off_an_yaw
off_lbc, off_alive, off_hit, off_d_hvt
def_x, def_y, def_z, def_v
def_an, def_an_pitch, def_an_yaw
def_initial_target
def_assigned_target
def_current_attack_target
def_lmode, def_ltgt, def_alive
assign_cost, fov_sat
hvt_x, hvt_y, hvt_z, hit_count
```

`game_data.npz` contains game-theoretic metrics:

```text
decoy_Phi
pen_N_eff
esc_Gamma_mean, esc_Xi_mean
esc_Gamma_matrix, esc_Xi_matrix
decoy_role_decoy
decoy_role_pen
decoy_role_stealth
decoy_lock_pressure
pen_P_pen
esc_E_esc
hvt_P_hit
hvt_rho
hvt_closing
def_lmode
def_current_attack_target
def_ltgt
```

Array population dimensions should be 10 for a 10v10 result. Validate before
plotting:

```bash
python3 - <<'PY'
import numpy as np
d = np.load("trajectory_data.npz", allow_pickle=True)
print("off_x", d["off_x"].shape)
print("def_x", d["def_x"].shape)
print("actor_actions", d["actor_actions"].shape)
assert d["off_x"].shape[0] == 10
assert d["def_x"].shape[0] == 10
PY
```

## 14. Existing Result Families

Use these as references, not as replacements for the requested 10v10 results:

- `shape_target_selected_20260620_163320/`: latest selected successful results
  using current interceptor attack targets and mild offensive HIL.
- `latest_attack_target_newseeds_20260620/`: newer seed scans using the latest
  attack-target behavior.
- `uncommitted_interceptor_v71_success_20260620/`: results tied to uncommitted
  interceptor code; includes source diff.
- `repro_success_4v4_6v6_8v8/`: compact earlier successful result family.
- `hil_offense_calib5_best/` and `hil_offense_calib5_top5_options/`: HIL
  calibration candidates.
- `article_repro_exact/` and `article_repro_delay_eval_obs100ms/`: article
  reproduction and delay-evaluation outputs.
- `mc_selected_methods_1000_reuse_parallel/` and
  `mc_hetero_guard_1000_reuse/`: Monte Carlo method comparisons.

The latest selected results document:

```text
Remote repo: /home/a2rl/000000GSY_mutiUAV/swarm_attack_v2
Model: outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models
Reward: FOV_REWARD_PROFILE=v69teamsurvive
Defender plotting target: def_current_attack_target
```

## 15. Known Risks and Pitfalls

1. The remote hosts may have changed. The old DNS names currently close SSH
   during key exchange.
2. The remote repository may contain uncommitted interceptor changes. Preserve
   `git status`, `git diff`, and source snapshots before editing.
3. The local `source_state/collect_v71_4v4_deterministic.py` snapshot visibly
   contains a duplicated `if attack_target is None:` line in one displayed
   section. Treat snapshots as provenance, not automatically as clean source.
   Inspect and syntax-check the actual remote file.
4. `10v10` is not currently accepted by the dedicated script CLIs. Add it
   explicitly rather than letting an `else` branch silently treat it as 8v8.
5. A scan result is not plottable until the successful seed is rerun with full
   recording.
6. Policy checkpoints are cloned from four actors. Do not claim that a native
   10-agent policy was trained.
7. The plotter can technically handle 10 agents, but legends and line density
   require visual inspection.
8. Keep HIL configuration identical across all three selected episodes.
9. Record exact seed, model directory, git commit, worktree diff, HIL config,
   and clone map in each output package.
10. Do not overwrite prior result directories. Use a dated new output root.

## 16. Acceptance Checklist

The 10v10 task is complete only when all items pass:

- [ ] Remote project and exact source state identified.
- [ ] 10v10 support added to dedicated evaluation scripts.
- [ ] One-seed 10v10 smoke test passes.
- [ ] At least three distinct seeds have `success == true`.
- [ ] Each success is rerun with full trajectory and game recording.
- [ ] Three result directories are copied into this local workspace.
- [ ] Each directory contains both NPZ files and `summary.json`.
- [ ] NPZ offensive and defensive population dimensions are both 10.
- [ ] All three summaries use the same model and HIL configuration.
- [ ] `plot_all.py` completes without error.
- [ ] Exactly 48 expected PDFs are generated, unless the plot set is
  intentionally changed and documented.
- [ ] Key trajectory, assignment, and game figures are visually inspected.
- [ ] A top-level README lists seeds, hit agents, hit times, survivors, model,
  source commit, HIL parameters, and remote origin.

## 17. Immediate Next Action for the Next AI

1. Read this document and
   `shape_target_selected_20260620_163320/README.md`.
2. Obtain current reachable remote host addresses from the user or cloud state.
3. Probe the remote repository and preserve source state.
4. Add dedicated 10v10 support.
5. Run three non-overlapping parallel seed scans.
6. Record, copy, validate, and plot the three best successful episodes.
