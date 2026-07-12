import sys, os, json, multiprocessing as mp, queue
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
TORCH_SEED = 42  # fixed for reproducibility; must be set before env/policy imports
torch.manual_seed(TORCH_SEED)
print(f'torch.manual_seed({TORCH_SEED}) set', flush=True)

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from envs.fov_penetration.policies_interceptor import InterceptorPolicy
from phase_obs_wrapper import PhaseMaskedFOVWrapper
from terminal_pn_action_wrapper import TerminalPNActionWrapper
from eval_v70_gifs import load_policies

MODEL_DIR = 'outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models'
HIDDEN = 256
OUT_BASE = '/tmp/v71_figs'

TARGET_CASES = int(os.environ.get('V71_TARGET_CASES', '3'))
CASE_LABELS = [f'case{chr(ord("A") + i)}' for i in range(TARGET_CASES)]

os.makedirs(OUT_BASE, exist_ok=True)

env0 = FOVPenetrationEnv()
env  = TerminalPNActionWrapper(PhaseMaskedFOVWrapper(env0, 'v65_strict_los'), gain=3.0, max_action=0.8)
policies = load_policies(env0, MODEL_DIR, hidden_size=HIDDEN, layer_N=3)
print(f'Loaded {len(policies)} policies, obs_dim={env0.obs_dim}')

STATE_NAMES = {0:'INIT', 1:'FOV_TRACK', 2:'LOCKED', 3:'MISSED', 4:'ABANDON'}
STATE_COLORS = {0:'#999999', 1:'#3498db', 2:'#e74c3c', 3:'#f39c12', 4:'#7f8c8d'}
OFF_COLORS = ['#e74c3c','#3498db','#2ecc71','#9b59b6']   # A0,A1,A2,A3
DEF_COLORS = ['#1abc9c','#e67e22','#34495e','#d35400']   # D0,D1,D2,D3


def init_plotting():
    global matplotlib, plt, mpatches, Line2D, GridSpec
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec


def save_record_data(rec, out_dir, case_name, ep, seed, torch_seed=None):
    data = {}
    for key, value in rec.items():
        if key in ('death_step', 'hit_step'):
            continue
        if isinstance(value, list):
            data[key] = np.array(value, dtype=object)
        else:
            data[key] = value

    np.savez_compressed(os.path.join(out_dir, 'trajectory_data.npz'), **data)

    hitter = int(next(iter(rec['hit_step'].keys()))) if rec['hit_step'] else -1
    hit_step = int(rec['hit_step'][hitter]) if hitter >= 0 else -1
    best_dist = float(np.min(rec['off_d_hvt'][hitter])) if hitter >= 0 else None
    summary = {
        'case': case_name,
        'episode': int(ep),
        'seed': int(seed),
        'torch_seed': int(TORCH_SEED if torch_seed is None else torch_seed),
        'hit_count': int(rec['hit_count']),
        'hitter': hitter,
        'hit_step': hit_step,
        'hit_time_s': float(hit_step * 0.01) if hit_step >= 0 else None,
        'best_hvt_distance_m': best_dist,
        'hitter_locked_by_count_max': int(np.max(rec['off_lbc'][hitter])) if hitter >= 0 else None,
        'death_step': {str(k): int(v) for k, v in rec['death_step'].items()},
        'hit_step_by_attacker': {str(k): int(v) for k, v in rec['hit_step'].items()},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f'  Saved {os.path.join(out_dir, "trajectory_data.npz")}')
    print(f'  Saved {os.path.join(out_dir, "summary.json")}')

def _run_episode_steps(obs, record):
    """Inner step loop. record=True collects trajectory, False runs fast."""
    n_off = len(env0.offensives)
    n_def = len(env0.defensives)
    rnn_states = [np.zeros((1,1,HIDDEN), np.float32) for _ in range(4)]
    masks      = [np.ones((1,1), np.float32) for _ in range(4)]

    if not record:
        for _ in range(8000):
            acts = []; new_rnn = []
            for i in range(4):
                o = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    act, _, rnn_h = policies[i].actor(
                        o, torch.FloatTensor(rnn_states[i]), torch.FloatTensor(masks[i]))
                acts.append(act.squeeze(0).numpy()); new_rnn.append(rnn_h.numpy())
            rnn_states = new_rnn
            obs, _, _, _, dones, _, _ = env.step(acts)
            masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]
            if all(dones): break
        return None

    rec = {
        'steps': [], 'time': [],
        'off_x':   [[] for _ in range(n_off)], 'off_y':   [[] for _ in range(n_off)],
        'off_z':   [[] for _ in range(n_off)], 'off_v':   [[] for _ in range(n_off)],
        'off_heading': [[] for _ in range(n_off)], 'off_gamma': [[] for _ in range(n_off)],
        'off_an_pitch':[[] for _ in range(n_off)], 'off_an_yaw':  [[] for _ in range(n_off)],
        'off_lbc':     [[] for _ in range(n_off)], 'off_alive':   [[] for _ in range(n_off)],
        'off_hit':     [[] for _ in range(n_off)], 'off_d_hvt':   [[] for _ in range(n_off)],
        'def_x':    [[] for _ in range(n_def)], 'def_y':    [[] for _ in range(n_def)],
        'def_z':    [[] for _ in range(n_def)], 'def_v':    [[] for _ in range(n_def)],
        'def_an':   [[] for _ in range(n_def)], 'def_an_pitch': [[] for _ in range(n_def)], 'def_an_yaw': [[] for _ in range(n_def)], 'def_lmode':[[] for _ in range(n_def)],
        'def_initial_target': [[] for _ in range(n_def)],
        'def_assigned_target': [[] for _ in range(n_def)],
        'def_ltgt': [[] for _ in range(n_def)], 'def_alive':[[] for _ in range(n_def)],
        'assign_cost': [], 'fov_sat': [],
    }
    return rec, rnn_states, masks, obs, n_off, n_def


def replay_episode(seed, record=True):
    """Run one episode. If record=False, run fast without storing trajectory."""
    env0.seed(seed)
    obs, _, _ = env.reset()
    rnn_states = [np.zeros((1,1,HIDDEN), np.float32) for _ in range(4)]
    masks      = [np.ones((1,1), np.float32) for _ in range(4)]

    if not record:
        for _ in range(8000):
            acts = []; new_rnn = []
            for i in range(4):
                o = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    act, _, rnn_h = policies[i].actor(
                        o, torch.FloatTensor(rnn_states[i]), torch.FloatTensor(masks[i]))
                acts.append(act.squeeze(0).numpy()); new_rnn.append(rnn_h.numpy())
            rnn_states = new_rnn
            obs, _, _, _, dones, _, _ = env.step(acts)
            masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]
            if all(dones): break
        return None

    n_off = len(env0.offensives)
    n_def = len(env0.defensives)

    rec = {
        'steps': [], 'time': [],
        'off_x':   [[] for _ in range(n_off)], 'off_y':   [[] for _ in range(n_off)],
        'off_z':   [[] for _ in range(n_off)], 'off_v':   [[] for _ in range(n_off)],
        'off_heading': [[] for _ in range(n_off)], 'off_gamma':   [[] for _ in range(n_off)],
        'off_an_pitch':[[] for _ in range(n_off)], 'off_an_yaw':  [[] for _ in range(n_off)],
        'off_lbc':     [[] for _ in range(n_off)], 'off_alive':   [[] for _ in range(n_off)],
        'off_hit':     [[] for _ in range(n_off)], 'off_d_hvt':   [[] for _ in range(n_off)],
        'def_x':    [[] for _ in range(n_def)], 'def_y':    [[] for _ in range(n_def)],
        'def_z':    [[] for _ in range(n_def)], 'def_v':    [[] for _ in range(n_def)],
        'def_an':   [[] for _ in range(n_def)], 'def_an_pitch': [[] for _ in range(n_def)], 'def_an_yaw': [[] for _ in range(n_def)], 'def_lmode':[[] for _ in range(n_def)],
        'def_initial_target': [[] for _ in range(n_def)],
        'def_assigned_target': [[] for _ in range(n_def)],
        'def_ltgt': [[] for _ in range(n_def)], 'def_alive':[[] for _ in range(n_def)],
        'assign_cost': [], 'fov_sat': [],
    }

    for step in range(8000):
        acts = []; new_rnn = []
        for i in range(4):
            o = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                act, _, rnn_h = policies[i].actor(
                    o, torch.FloatTensor(rnn_states[i]), torch.FloatTensor(masks[i]))
            acts.append(act.squeeze(0).numpy()); new_rnn.append(rnn_h.numpy())
        rnn_states = new_rnn
        obs, _, rews, _, dones, infos, _ = env.step(acts)
        masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]

        t = step * 0.01
        rec['steps'].append(step)
        rec['time'].append(t)

        for i, off in enumerate(env0.offensives):
            rec['off_x'][i].append(off.x)
            rec['off_y'][i].append(off.y)
            rec['off_z'][i].append(off.z)
            rec['off_v'][i].append(off.v)
            rec['off_heading'][i].append(off.heading)
            rec['off_gamma'][i].append(off.gamma)
            rec['off_an_pitch'][i].append(off.an_pitch)
            rec['off_an_yaw'][i].append(off.an_yaw)
            rec['off_lbc'][i].append(off.locked_by_count)
            rec['off_alive'][i].append(int(off.alive))
            rec['off_hit'][i].append(int(off.hit_hvt))
            d_hvt = off.distance_to(env0.hvt.x, env0.hvt.y, env0.hvt.z)
            rec['off_d_hvt'][i].append(d_hvt)

        for j, d in enumerate(env0.defensives):
            rec['def_x'][j].append(d.x)
            rec['def_y'][j].append(d.y)
            rec['def_z'][j].append(d.z)
            rec['def_v'][j].append(d.v)
            an_tot = np.sqrt(d.an_pitch**2 + d.an_yaw**2)
            rec['def_an'][j].append(an_tot)
            rec['def_an_pitch'][j].append(d.an_pitch)
            rec['def_an_yaw'][j].append(d.an_yaw)
            p = env0.defensive_policies[j]
            rec['def_lmode'][j].append(p.lock_mode)
            rec['def_initial_target'][j].append(
                p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1)
            rec['def_assigned_target'][j].append(
                p.assigned_target_idx if p.assigned_target_idx is not None else (
                    p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1))
            rec['def_ltgt'][j].append(
                p.assigned_target_idx if p.assigned_target_idx is not None else (
                    p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1))
            rec['def_alive'][j].append(int(d.alive))

        # assignment cost matrix n_def x n_off
        cost_mat = np.zeros((n_def, n_off))
        for j, dd in enumerate(env0.defensives):
            for ii, off in enumerate(env0.offensives):
                cost_mat[j, ii] = dd.distance_to(off.x, off.y, off.z)
        rec['assign_cost'].append(cost_mat)

        # FOV saturation
        n_locked = sum(1 for p in env0.defensive_policies if p.lock_mode == InterceptorPolicy.STATE_LOCKED)
        rec['fov_sat'].append(n_locked / n_def)

        if all(dones):
            break

    # convert to arrays
    for key in rec:
        if isinstance(rec[key], list) and len(rec[key]) > 0 and isinstance(rec[key][0], list):
            rec[key] = [np.array(x) for x in rec[key]]
        elif isinstance(rec[key], list) and len(rec[key]) > 0 and not isinstance(rec[key][0], np.ndarray):
            rec[key] = np.array(rec[key])

    # find death and hit steps
    rec['death_step'] = {}
    rec['hit_step'] = {}
    for i, alive_arr in enumerate(rec['off_alive']):
        hit_arr = rec['off_hit'][i]
        for s, (a, h) in enumerate(zip(alive_arr, hit_arr)):
            if h and i not in rec['hit_step']:
                rec['hit_step'][i] = s
            if not a and not h and i not in rec['death_step']:
                rec['death_step'][i] = s

    rec['hvt_x'] = env0.hvt.x
    rec['hvt_y'] = env0.hvt.y
    rec['hvt_z'] = env0.hvt.z
    rec['hit_count'] = env0.hit_count
    return rec


def compute_R_optimal(cost_list, n_def, n_off):
    """Compute R(t) = actual_assign_cost / optimal_KM_cost."""
    from scipy.optimize import linear_sum_assignment
    R_arr = []
    actual_arr = []
    opt_arr = []
    for cost_mat in cost_list:
        # KM (linear_sum_assignment minimizes)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        opt_cost = cost_mat[row_ind, col_ind].sum()
        # actual assignment cost: for each defender, use its currently locked target or nearest
        actual_cost = 0
        for j in range(n_def):
            assigned = np.argmin(cost_mat[j])  # greedy nearest
            actual_cost += cost_mat[j, assigned]
        if opt_cost > 0:
            R = actual_cost / opt_cost
        else:
            R = 1.0
        R_arr.append(R)
        actual_arr.append(actual_cost)
        opt_arr.append(opt_cost)
    return np.array(R_arr), np.array(opt_arr)


def plot_fig1_kinematics(rec, case_name, out_dir):
    """Fig1: all 8 aircraft kinematics time series."""
    t = rec['time']
    T = len(t)
    n_off = len(rec['off_v'])
    n_def = len(rec['def_v'])

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(8, 1, figure=fig, hspace=0.45)

    # ---- OFFENSIVE: speed ----
    ax1 = fig.add_subplot(gs[0])
    for i in range(n_off):
        arr = rec['off_v'][i]
        mask = np.array(rec['off_alive'][i], dtype=bool) | np.array(rec['off_hit'][i], dtype=bool)
        alive_t = np.array(t[:len(arr)])
        ax1.plot(alive_t[mask], arr[mask], color=OFF_COLORS[i], lw=1.4, label=f'A{i}')
    ax1.set_ylabel('$V$ (m/s)', fontsize=9)
    ax1.axhline(40, color='gray', ls='--', lw=0.7); ax1.axhline(50, color='gray', ls='--', lw=0.7)
    ax1.set_ylim(35, 55); ax1.legend(loc='upper right', fontsize=7, ncol=4)
    ax1.set_title(f'{case_name} — Offensives Kinematics', fontsize=10)
    ax1.tick_params(labelbottom=False)

    # ---- OFFENSIVE: heading rate ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for i in range(n_off):
        h_arr = np.array(rec['off_heading'][i])
        mask = np.array(rec['off_alive'][i], dtype=bool) | np.array(rec['off_hit'][i], dtype=bool)
        if len(h_arr) > 1:
            dh = np.gradient(np.unwrap(h_arr), 0.01)
            alive_t = np.array(t[:len(dh)])
            ax2.plot(alive_t[mask[:len(dh)]], np.degrees(dh)[mask[:len(dh)]], color=OFF_COLORS[i], lw=1.2)
    ax2.set_ylabel('$\\dot{\\psi}$ (°/s)', fontsize=9)
    ax2.axhline(0, color='gray', ls='-', lw=0.5)
    ax2.tick_params(labelbottom=False)

    # ---- OFFENSIVE: pitch load ----
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for i in range(n_off):
        arr = np.array(rec['off_an_pitch'][i]) / 9.81
        mask = np.array(rec['off_alive'][i], dtype=bool) | np.array(rec['off_hit'][i], dtype=bool)
        alive_t = np.array(t[:len(arr)])
        ax3.plot(alive_t[mask], arr[mask], color=OFF_COLORS[i], lw=1.2)
    ax3.set_ylabel('$n_p$ (g)', fontsize=9)
    ax3.axhline(3, color='red', ls='--', lw=0.7, alpha=0.5); ax3.axhline(-3, color='red', ls='--', lw=0.7, alpha=0.5)
    ax3.tick_params(labelbottom=False)

    # ---- OFFENSIVE: yaw load ----
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for i in range(n_off):
        arr = np.array(rec['off_an_yaw'][i]) / 9.81
        mask = np.array(rec['off_alive'][i], dtype=bool) | np.array(rec['off_hit'][i], dtype=bool)
        alive_t = np.array(t[:len(arr)])
        ax4.plot(alive_t[mask], arr[mask], color=OFF_COLORS[i], lw=1.2)
    ax4.set_ylabel('$n_y$ (g)', fontsize=9)
    ax4.axhline(3, color='red', ls='--', lw=0.7, alpha=0.5); ax4.axhline(-3, color='red', ls='--', lw=0.7, alpha=0.5)
    ax4.tick_params(labelbottom=False)

    # ---- HITTER: distance to HVT and nearest defender ----
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    hitter = list(rec['hit_step'].keys())[0] if rec['hit_step'] else 0
    d_hvt = rec['off_d_hvt'][hitter]
    alive_t_h = np.array(t[:len(d_hvt)])
    ax5.semilogy(alive_t_h, d_hvt, color=OFF_COLORS[hitter], lw=1.5, label=f'$d_{{A{hitter},H}}$')
    # nearest defender distance
    d_def_min = []
    for s in range(len(d_hvt)):
        off_x = rec['off_x'][hitter][s]; off_y = rec['off_y'][hitter][s]; off_z = rec['off_z'][hitter][s]
        min_d = 9999
        for j in range(n_def):
            if s < len(rec['def_x'][j]):
                dx = rec['def_x'][j][s]-off_x; dy = rec['def_y'][j][s]-off_y; dz = rec['def_z'][j][s]-off_z
                min_d = min(min_d, np.sqrt(dx*dx+dy*dy+dz*dz))
        d_def_min.append(min_d)
    ax5.semilogy(alive_t_h, d_def_min, color='#e67e22', lw=1.2, ls='--', label=f'$d_{{A{hitter},D^*}}$')
    ax5.axhline(500, color='gray', ls=':', lw=1.0, label='detect 500m')
    ax5.axhline(5, color='red', ls=':', lw=1.0, label='hit 5m')
    ax5.set_ylabel('Distance (m)', fontsize=9)
    ax5.legend(loc='upper right', fontsize=7)

    # mark hit time with green shade
    if rec['hit_step']:
        hit_t = rec['hit_step'][hitter] * 0.01
        ax5.axvspan(hit_t - 5, hit_t, alpha=0.15, color='green')
    ax5.tick_params(labelbottom=False)

    # ---- DEFENSIVE: normal load ----
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    for j in range(n_def):
        arr = np.array(rec['def_an'][j])
        alive_t_d = np.array(t[:len(arr)])
        ax6.plot(alive_t_d, arr, color=DEF_COLORS[j], lw=1.2, label=f'D{j}')
    ax6.axhline(5, color='red', ls='--', lw=0.7, alpha=0.6, label='5g limit')
    ax6.set_ylabel('$|n_d|$ (g)', fontsize=9)
    ax6.legend(loc='upper right', fontsize=7, ncol=4)
    ax6.set_title('Defenders Kinematics', fontsize=9)
    ax6.tick_params(labelbottom=False)

    # ---- DEFENSIVE: lock mode ----
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    for j in range(n_def):
        modes = np.array(rec['def_lmode'][j])
        alive_t_d = np.array(t[:len(modes)])
        ax7.plot(alive_t_d, modes + j*0.08, color=DEF_COLORS[j], lw=1.5, label=f'D{j}')
    ax7.set_yticks([0,1,2,3,4])
    ax7.set_yticklabels(['INIT','FOV_TRK','LOCKED','MISSED','ABANDON'], fontsize=7)
    ax7.set_ylabel('State', fontsize=9)
    ax7.legend(loc='upper right', fontsize=7, ncol=4)
    ax7.tick_params(labelbottom=False)

    # ---- death/hit markers ----
    for i in range(n_off):
        ds = rec['death_step'].get(i)
        if ds is not None:
            dt_ = ds * 0.01
            for ax_ in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
                ax_.axvline(dt_, color=OFF_COLORS[i], ls=':', lw=0.8, alpha=0.6)

    # ---- time axis ----
    ax7.set_xlabel('Time (s)', fontsize=9)
    ax7.tick_params(labelbottom=True)

    # last (empty) subplot for overall xlabel
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig1_kinematics_all8.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_fig2_def_assignment(rec, case_name, out_dir):
    """Fig2: interceptor target assignment timeline."""
    t = rec['time']
    n_def = len(rec['def_v'])
    n_off = len(rec['off_v'])

    # compute R(t)
    R_arr, opt_arr = compute_R_optimal(rec['assign_cost'], n_def, n_off)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{case_name} — Interceptor Assignment Analysis', fontsize=11, y=0.98)

    # ---- subplot 1: state machine states ----
    ax = axes[0]
    for j in range(n_def):
        modes = np.array(rec['def_lmode'][j])
        alive_t_d = np.array(t[:len(modes)])
        ax.step(alive_t_d, modes, color=DEF_COLORS[j], lw=1.8, label=f'D{j}', where='post')
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['INIT_GUIDE','FOV_TRACK','LOCKED','MISSED','ABANDON'], fontsize=8)
    ax.set_ylabel('State', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.set_title('(a) Defender State Machine Evolution', fontsize=9, loc='left')
    ax.grid(True, alpha=0.3)

    # death markers
    for i in range(n_off):
        ds = rec['death_step'].get(i)
        if ds is not None:
            ax.axvline(ds*0.01, color=OFF_COLORS[i], ls='--', lw=1.0, alpha=0.7, label=f'A{i} dead')

    # ---- subplot 2: assignment cost for each defender → each attacker ----
    ax = axes[1]
    # Show: for each defender j, the cost to the HITTER vs min cost to decoys
    hitter = list(rec['hit_step'].keys())[0] if rec['hit_step'] else 0
    decoys = [i for i in range(n_off) if i != hitter]

    t_assign = np.array(t[:len(rec['assign_cost'])])
    # cost to hitter from D0
    for j in range(n_def):
        cost_to_hitter = np.array([rec['assign_cost'][s][j, hitter] for s in range(len(rec['assign_cost']))])
        ax.plot(t_assign, cost_to_hitter, color=DEF_COLORS[j], lw=1.5, label=f'$C_{{D{j},A{hitter}}}$ (hitter)')
    # min cost to any decoy
    for j in range(n_def):
        min_decoy_cost = np.array([min(rec['assign_cost'][s][j, d] for d in decoys) for s in range(len(rec['assign_cost']))])
        ax.plot(t_assign, min_decoy_cost, color=DEF_COLORS[j], lw=1.0, ls='--', alpha=0.6)
    ax.set_ylabel('Dist cost (m)', fontsize=10)
    ax.set_title(f'(b) Assignment Cost: solid=to hitter A{hitter}, dash=min to decoys', fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # ---- subplot 3: R(t) ----
    ax = axes[2]
    ax.plot(t_assign, R_arr, color='#c0392b', lw=2.0, label='$R(t)$=actual/optimal')
    ax.axhline(1.0, color='gray', ls='--', lw=1.0, label='optimal=1.0')
    ax.axhline(2.0, color='orange', ls=':', lw=1.0, label='R=2')
    ax.fill_between(t_assign, 1.0, R_arr, where=(R_arr>1.0), alpha=0.2, color='red')
    ax.set_ylabel('$R(t)$', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_title('(c) Assignment Sub-optimality Ratio $R(t)$', fontsize=9, loc='left')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # death/hit markers on all subplots
    for i in range(n_off):
        ds = rec['death_step'].get(i)
        if ds is not None:
            for ax_ in axes:
                ax_.axvline(ds*0.01, color=OFF_COLORS[i], ls=':', lw=0.9, alpha=0.6)
    if rec['hit_step']:
        ht = rec['hit_step'][hitter] * 0.01
        for ax_ in axes:
            ax_.axvline(ht, color='green', ls='-', lw=1.5, alpha=0.8, label='HIT')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig2_def_assignment_timeline.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def plot_fig3_attacker_game(rec, case_name, out_dir):
    """Fig3: attacker game-theoretic view."""
    t = rec['time']
    n_off = len(rec['off_v'])
    n_def = len(rec['def_v'])

    hitter = list(rec['hit_step'].keys())[0] if rec['hit_step'] else 0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{case_name} — Attacker Game-Theoretic Metrics', fontsize=11, y=0.98)

    # ---- subplot 1: locked_by_count per offensive ----
    ax = axes[0]
    for i in range(n_off):
        lbc = np.array(rec['off_lbc'][i])
        alive_t = np.array(t[:len(lbc)])
        lw = 2.5 if i == hitter else 1.5
        ls = '-' if i == hitter else '--'
        label = f'A{i} (hitter)' if i == hitter else f'A{i} (decoy)'
        ax.step(alive_t, lbc, color=OFF_COLORS[i], lw=lw, ls=ls, label=label, where='post')
    ax.set_ylabel('locked_by_count', fontsize=10)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title('(a) Interceptors Locking Each Attacker', fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ---- subplot 2: FOV saturation ρ(t) ----
    ax = axes[1]
    rho = np.array(rec['fov_sat'])
    t_rho = np.array(t[:len(rho)])
    ax.fill_between(t_rho, 0, rho, alpha=0.3, color='#2980b9')
    ax.plot(t_rho, rho, color='#2980b9', lw=2.0, label='$\\rho(t)$')
    ax.axhline(1.0, color='green', ls='--', lw=1.0, label='Full saturation')
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('FOV saturation $\\rho$', fontsize=10)
    ax.set_title('(b) FOV Resource Saturation $\\rho(t) = N_{\\rm locked}/N_d$', fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- subplot 3: hitter dist to HVT and nearest defender ----
    ax = axes[2]
    d_hvt = np.array(rec['off_d_hvt'][hitter])
    alive_t_h = np.array(t[:len(d_hvt)])

    d_def_min = []
    for s in range(len(d_hvt)):
        off_x = rec['off_x'][hitter][s]; off_y = rec['off_y'][hitter][s]; off_z = rec['off_z'][hitter][s]
        min_d = 9999
        for j in range(n_def):
            if s < len(rec['def_x'][j]):
                dx = rec['def_x'][j][s]-off_x; dy = rec['def_y'][j][s]-off_y; dz = rec['def_z'][j][s]-off_z
                min_d = min(min_d, np.sqrt(dx*dx+dy*dy+dz*dz))
        d_def_min.append(min_d)
    d_def_min = np.array(d_def_min)

    ax.semilogy(alive_t_h, d_hvt, color=OFF_COLORS[hitter], lw=2.0, label=f'$d_{{A{hitter},\\rm HVT}}$')
    ax.semilogy(alive_t_h, d_def_min, color='#e67e22', lw=1.5, ls='--', label=f'$d_{{A{hitter},D^*}}$')
    ax.axhline(500, color='#7f8c8d', ls=':', lw=1.0, label='Detection range 500m')
    ax.axhline(5, color='red', ls=':', lw=1.0, label='Hit range 5m')
    ax.fill_between(alive_t_h, 1, d_def_min, where=(d_def_min > 500), alpha=0.1, color='green', label='Zero-threat zone')

    if rec['hit_step']:
        ht = rec['hit_step'][hitter] * 0.01
        ax.axvspan(ht - 5, ht, alpha=0.2, color='green')

    ax.set_ylabel('Distance (m)', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_title(f'(c) Striker A{hitter}: Distance to HVT and Nearest Defender', fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # death/hit markers
    for i in range(n_off):
        ds = rec['death_step'].get(i)
        if ds is not None:
            for ax_ in axes:
                ax_.axvline(ds*0.01, color=OFF_COLORS[i], ls=':', lw=0.9, alpha=0.6)
    if rec['hit_step']:
        ht = rec['hit_step'][hitter] * 0.01
        for ax_ in axes:
            ax_.axvline(ht, color='green', ls='-', lw=1.5, alpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_attacker_game_view.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path}')


def scan_episode(seed, torch_seed):
    torch.manual_seed(torch_seed)
    env0.seed(seed)
    obs, _, _ = env.reset()
    rnn_states = [np.zeros((1, 1, HIDDEN), np.float32) for _ in range(4)]
    masks = [np.ones((1, 1), np.float32) for _ in range(4)]
    best_dist = 9999.0

    for step in range(8000):
        acts = []
        new_rnn = []
        for agent_id in range(4):
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                act, _, rnn_h = policies[agent_id].actor(
                    obs_tensor,
                    torch.FloatTensor(rnn_states[agent_id]),
                    torch.FloatTensor(masks[agent_id]),
                )
            acts.append(act.squeeze(0).numpy())
            new_rnn.append(rnn_h.numpy())
        rnn_states = new_rnn
        obs, _, _, _, dones, _, _ = env.step(acts)
        masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]

        for off in env0.offensives:
            dist = off.distance_to(env0.hvt.x, env0.hvt.y, env0.hvt.z)
            if dist < best_dist:
                best_dist = dist

        if env0.hit_count > 0:
            return True, best_dist, step
        if all(dones):
            break

    return False, best_dist, step


def build_trial_list():
    priority_envs = [50001, 50015, 50022, 50034, 50042]
    env_span = list(range(50000, int(os.environ.get('V71_ENV_SEED_END', '50200'))))
    env_seeds = priority_envs + [seed for seed in env_span if seed not in priority_envs]
    torch_seed_end = int(os.environ.get('V71_TORCH_SEED_END', '120'))
    torch_seeds = [42] + [seed for seed in range(torch_seed_end) if seed != 42]
    priority_pairs = [(env_seed, torch_seed) for torch_seed in torch_seeds for env_seed in priority_envs]
    broad_pairs = [
        (env_seed, torch_seed)
        for torch_seed in torch_seeds
        for env_seed in env_seeds
        if env_seed not in priority_envs
    ]
    return priority_pairs + broad_pairs


def worker_loop(worker_id, n_workers, trials, hit_counter, counter_lock, stop_event, result_queue):
    init_plotting()
    checked = 0
    for trial_index in range(worker_id, len(trials), n_workers):
        if stop_event.is_set():
            break
        env_seed, torch_seed = trials[trial_index]
        hit, best_dist, end_step = scan_episode(env_seed, torch_seed)
        checked += 1
        print(
            f'[worker {worker_id:02d}] trial={trial_index:05d} env_seed={env_seed} '
            f'torch_seed={torch_seed} hit={int(hit)} best_dist={best_dist:.1f}m step={end_step}',
            flush=True,
        )
        if not hit:
            continue

        with counter_lock:
            case_idx = hit_counter.value
            if case_idx >= len(CASE_LABELS):
                stop_event.set()
                break
            hit_counter.value += 1
            if hit_counter.value >= len(CASE_LABELS):
                stop_event.set()

        case_name = f'{CASE_LABELS[case_idx]}_seed{env_seed}_torch{torch_seed}'
        out_dir = os.path.join(OUT_BASE, case_name)
        os.makedirs(out_dir, exist_ok=True)

        torch.manual_seed(torch_seed)
        rec = replay_episode(env_seed, record=True)
        if rec['hit_count'] == 0:
            print(f'[worker {worker_id:02d}] replay failed for {case_name}, skipping', flush=True)
            with counter_lock:
                hit_counter.value -= 1
                stop_event.clear()
            continue

        hitter = list(rec['hit_step'].keys())[0]
        hit_s = rec['hit_step'][hitter]
        print(f'\n=== {case_name}: HIT A{hitter} at step={hit_s}, best_dist={best_dist:.3f}m ===', flush=True)
        save_record_data(rec, out_dir, case_name, trial_index, env_seed, torch_seed=torch_seed)
        plot_fig1_kinematics(rec, case_name, out_dir)
        plot_fig2_def_assignment(rec, case_name, out_dir)
        plot_fig3_attacker_game(rec, case_name, out_dir)
        result_queue.put({
            'case': case_name,
            'env_seed': env_seed,
            'torch_seed': torch_seed,
            'best_dist': float(best_dist),
            'hitter': int(hitter),
            'hit_step': int(hit_s),
            'out_dir': out_dir,
        })

    result_queue.put({'worker_done': worker_id, 'checked': checked})


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    trials = build_trial_list()
    n_workers = min(int(os.environ.get('V71_WORKERS', '12')), len(trials))
    print(f'Searching current v71 hits: {len(trials)} trials, workers={n_workers}', flush=True)

    ctx = mp.get_context('fork')
    hit_counter = ctx.Value('i', 0)
    counter_lock = ctx.Lock()
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    workers = []
    for worker_id in range(n_workers):
        proc = ctx.Process(
            target=worker_loop,
            args=(worker_id, n_workers, trials, hit_counter, counter_lock, stop_event, result_queue),
        )
        proc.start()
        workers.append(proc)

    hits = []
    finished_workers = 0
    while finished_workers < n_workers and len(hits) < len(CASE_LABELS):
        try:
            message = result_queue.get(timeout=30)
        except queue.Empty:
            alive = sum(proc.is_alive() for proc in workers)
            print(f'Progress: hits={len(hits)}, workers_alive={alive}', flush=True)
            if alive == 0:
                break
            continue
        if 'worker_done' in message:
            finished_workers += 1
            print(f'Worker {message["worker_done"]} done after {message["checked"]} trials', flush=True)
        else:
            hits.append(message)
            print(f'COLLECTED HIT #{len(hits)}: {message}', flush=True)
            if len(hits) >= len(CASE_LABELS):
                stop_event.set()

    stop_event.set()
    for proc in workers:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

    summary_path = os.path.join(OUT_BASE, 'v71_current_hits_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(hits, handle, ensure_ascii=False, indent=2)
    print(f'\n=== Search finished: {len(hits)} hits saved to {OUT_BASE} ===')
    print(json.dumps(hits, ensure_ascii=False, indent=2), flush=True)
    if len(hits) < len(CASE_LABELS):
        raise RuntimeError(f'Only found {len(hits)} hits; need {len(CASE_LABELS)}')


if __name__ == '__main__':
    main()
