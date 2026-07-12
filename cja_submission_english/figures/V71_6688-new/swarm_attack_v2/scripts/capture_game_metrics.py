#!/usr/bin/env python
"""仅补录博弈值(角色/群体价值)到已有npz, 不重录动力学数据."""
import sys, os, json
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(42)

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from phase_obs_wrapper import PhaseMaskedFOVWrapper
from terminal_pn_action_wrapper import TerminalPNActionWrapper
from eval_v70_gifs import load_policies

MODEL_DIR = 'outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models'
HIDDEN = 256
SEEDS = {
    'caseA_seed50015_torch1': (50015, 1),
    'caseB_seed50042_torch1': (50042, 1),
    'caseC_seed50034_torch7': (50034, 7),
    'caseC_seed50042_torch7': (50042, 7),
    'caseD_seed50015_torch8': (50015, 8),
    'caseD_seed50015_torch9': (50015, 9),
    'caseE_seed50034_torch7': (50034, 7),
    'caseE_seed50042_torch9': (50042, 9),
}

env0 = FOVPenetrationEnv()
env  = TerminalPNActionWrapper(PhaseMaskedFOVWrapper(env0, 'v65_strict_los'), gain=3.0, max_action=0.8)
policies = load_policies(env0, MODEL_DIR, hidden_size=HIDDEN, layer_N=3)

def cap(v, default=0.0):
    try: return float(v) if np.isfinite(float(v)) else default
    except: return default


def vec(info, key, n, default=0.0):
    val = info.get(key)
    if val is None:
        return [default] * n
    arr = np.asarray(val).reshape(-1)
    out = arr.tolist()[:n]
    if len(out) < n:
        out.extend([default] * (n - len(out)))
    return out


def mat(info, key, rows, cols, default=0.0):
    val = info.get(key)
    if val is None:
        return np.full((rows, cols), default, dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return np.full((rows, cols), float(arr), dtype=float)
    arr = np.atleast_2d(arr)
    out = np.full((rows, cols), default, dtype=float)
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return out

def replay_and_capture(env_seed, torch_seed):
    torch.manual_seed(torch_seed)
    env0.seed(env_seed)
    obs, _, _ = env.reset()
    rnn = [np.zeros((1,1,HIDDEN), np.float32) for _ in range(4)]
    masks = [np.ones((1,1), np.float32) for _ in range(4)]
    n_off, n_def = 4, 4

    game = {
        'decoy_Phi': [], 'decoy_role_decoy': [[] for _ in range(n_off)],
        'decoy_role_pen': [[] for _ in range(n_off)],
        'decoy_role_stealth': [[] for _ in range(n_off)],
        'decoy_lock_pressure': [[] for _ in range(n_off)],
        'pen_N_eff': [], 'pen_P_pen': [[] for _ in range(n_off)],
        'esc_Gamma_mean': [], 'esc_Xi_mean': [],
        'esc_E_esc': [[] for _ in range(n_off)],
        'hvt_P_hit': [[] for _ in range(n_off)], 'hvt_rho': [[] for _ in range(n_off)],
        'hvt_closing': [[] for _ in range(n_off)],
        'def_lmode': [[] for _ in range(n_def)], 'def_ltgt': [[] for _ in range(n_def)],
    }

    for step in range(8000):
        acts = []; nr = []
        for i in range(4):
            o = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                act, _, rh = policies[i].actor(o, torch.FloatTensor(rnn[i]), torch.FloatTensor(masks[i]))
            acts.append(act.squeeze(0).numpy()); nr.append(rh.numpy())
        rnn = nr
        obs, _, _, _, dones, _, _ = env.step(acts)
        masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]

        # capture env internal game values
        di = getattr(env0, '_ap_decoy_info', {}) or {}
        pi = getattr(env0, '_ap_pen_info', {}) or {}
        ei = getattr(env0, '_ap_esc_info', {}) or {}
        hi = getattr(env0, '_ap_hvt_info', {}) or {}

        game['decoy_Phi'].append(cap(di.get('Phi_decoy', 0)))
        for i in range(n_off):
            game['decoy_role_decoy'][i].append(cap(vec(di, 'role_decoy_per_agent', n_off)[i]))
            game['decoy_role_pen'][i].append(cap(vec(di, 'role_penetrate_per_agent', n_off)[i]))
            game['decoy_role_stealth'][i].append(cap(vec(di, 'role_stealth_per_agent', n_off)[i]))
            game['decoy_lock_pressure'][i].append(cap(vec(di, 'lock_pressure_per_agent', n_off)[i]))

        game['pen_N_eff'].append(cap(pi.get('N_eff', 0)))
        for i in range(n_off):
            game['pen_P_pen'][i].append(cap(vec(pi, 'P_pen_per_agent', n_off)[i]))

        gm = mat(ei, '_Gamma_matrix', n_off, n_def)
        xm = mat(ei, '_Xi_matrix', n_off, n_def)
        game['esc_Gamma_mean'].append(cap(np.mean(gm) if gm.size else 0))
        game['esc_Xi_mean'].append(cap(np.mean(xm) if xm.size else 0))
        for i in range(n_off):
            game['esc_E_esc'][i].append(cap(vec(ei, 'E_i_esc', n_off)[i]))

        for i in range(n_off):
            game['hvt_P_hit'][i].append(cap(vec(hi, 'P_hit_per_agent', n_off)[i]))
            game['hvt_rho'][i].append(cap(vec(hi, 'rho_per_agent', n_off)[i]))
            game['hvt_closing'][i].append(cap(vec(hi, 'closing_per_agent', n_off)[i]))

        for j in range(n_def):
            p = env0.defensive_policies[j]
            game['def_lmode'][j].append(p.lock_mode)
            game['def_ltgt'][j].append(
                p.assigned_target_idx if p.assigned_target_idx is not None else (
                    p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1))

        if all(dones): break

    for k in game:
        if isinstance(game[k], list) and len(game[k]) > 0 and isinstance(game[k][0], list):
            game[k] = [np.array(x) for x in game[k]]
        elif isinstance(game[k], list):
            game[k] = np.array(game[k])
    return game


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '/tmp/v71_paper'
    for case, (env_s, torch_s) in SEEDS.items():
        base = os.path.join(root, case)
        if not os.path.isdir(base): continue
        print(f'>>> Capturing game metrics for {case}...')
        g = replay_and_capture(env_s, torch_s)
        # save game fields to separate npz
        out = {}
        for k, v in g.items():
            if isinstance(v, list): out[k] = np.array(v, dtype=object)
            else: out[k] = v
        np.savez_compressed(os.path.join(base, 'game_data.npz'), **out)
        print(f'  => {case} saved ({len(out)} fields)')
    print('All done.')
