#!/usr/bin/env python
"""重播3个V71 seed, 补录def_an_pitch/def_an_yaw到npz. 运行后原npz被覆盖."""
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
G = 9.80665
SEEDS = {  # case -> (env_seed, torch_seed)
    'caseA_seed50015_torch1': (50015, 1),
    'caseB_seed50042_torch1': (50042, 1),
    'caseC_seed50034_torch7': (50034, 7),
    'caseD_seed50042_torch7': (50042, 7),
    'caseE_seed50015_torch8': (50015, 8),
}

env0 = FOVPenetrationEnv()
env  = TerminalPNActionWrapper(PhaseMaskedFOVWrapper(env0, 'v65_strict_los'), gain=3.0, max_action=0.8)
policies = load_policies(env0, MODEL_DIR, hidden_size=HIDDEN, layer_N=3)
print(f'Policies loaded, obs_dim={env0.obs_dim}')

def replay(env_seed, torch_seed):
    torch.manual_seed(torch_seed)
    env0.seed(env_seed)
    obs, _, _ = env.reset()
    n_off, n_def = 4, 4
    rnn = [np.zeros((1,1,HIDDEN), np.float32) for _ in range(4)]
    masks = [np.ones((1,1), np.float32) for _ in range(4)]

    rec = {
        'steps':[], 'time':[],
        'off_x':[[] for _ in range(n_off)], 'off_y':[[] for _ in range(n_off)],
        'off_z':[[] for _ in range(n_off)], 'off_v':[[] for _ in range(n_off)],
        'off_heading':[[] for _ in range(n_off)], 'off_gamma':[[] for _ in range(n_off)],
        'off_an_pitch':[[] for _ in range(n_off)], 'off_an_yaw':[[] for _ in range(n_off)],
        'off_lbc':[[] for _ in range(n_off)], 'off_alive':[[] for _ in range(n_off)],
        'off_hit':[[] for _ in range(n_off)], 'off_d_hvt':[[] for _ in range(n_off)],
        'def_x':[[] for _ in range(n_def)], 'def_y':[[] for _ in range(n_def)],
        'def_z':[[] for _ in range(n_def)], 'def_v':[[] for _ in range(n_def)],
        'def_an':[[] for _ in range(n_def)],
        'def_an_pitch':[[] for _ in range(n_def)],  # NEW
        'def_an_yaw':[[] for _ in range(n_def)],    # NEW
        'def_initial_target':[[] for _ in range(n_def)],
        'def_assigned_target':[[] for _ in range(n_def)],
        'def_lmode':[[] for _ in range(n_def)], 'def_ltgt':[[] for _ in range(n_def)],
        'def_alive':[[] for _ in range(n_def)],
        'assign_cost':[], 'fov_sat':[],
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
        tt = step * 0.01
        rec['steps'].append(step); rec['time'].append(tt)

        for i, off in enumerate(env0.offensives):
            rec['off_x'][i].append(off.x); rec['off_y'][i].append(off.y)
            rec['off_z'][i].append(off.z); rec['off_v'][i].append(off.v)
            rec['off_heading'][i].append(off.heading); rec['off_gamma'][i].append(off.gamma)
            rec['off_an_pitch'][i].append(off.an_pitch); rec['off_an_yaw'][i].append(off.an_yaw)
            rec['off_lbc'][i].append(off.locked_by_count)
            rec['off_alive'][i].append(int(off.alive)); rec['off_hit'][i].append(int(off.hit_hvt))
            rec['off_d_hvt'][i].append(off.distance_to(env0.hvt.x, env0.hvt.y, env0.hvt.z))

        for j, ddef in enumerate(env0.defensives):
            rec['def_x'][j].append(ddef.x); rec['def_y'][j].append(ddef.y)
            rec['def_z'][j].append(ddef.z); rec['def_v'][j].append(ddef.v)
            rec['def_an'][j].append(np.sqrt(ddef.an_pitch**2 + ddef.an_yaw**2) / G)
            rec['def_an_pitch'][j].append(ddef.an_pitch / G)   # NEW
            rec['def_an_yaw'][j].append(ddef.an_yaw / G)       # NEW
            p = env0.defensive_policies[j]
            rec['def_initial_target'][j].append(
                p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1)
            rec['def_assigned_target'][j].append(
                p.assigned_target_idx if p.assigned_target_idx is not None else -1)
            rec['def_lmode'][j].append(p.lock_mode)
            rec['def_ltgt'][j].append(
                p.assigned_target_idx if p.assigned_target_idx is not None else (
                    p.initial_assigned_target_idx if p.initial_assigned_target_idx is not None else -1))
            rec['def_alive'][j].append(int(ddef.alive))

        cmat = np.zeros((n_def, n_off))
        for jj, dd2 in enumerate(env0.defensives):
            for ii, off2 in enumerate(env0.offensives):
                cmat[jj, ii] = dd2.distance_to(off2.x, off2.y, off2.z)
        rec['assign_cost'].append(cmat)
        n_locked = sum(1 for pp in env0.defensive_policies if pp.lock_mode == 2)
        rec['fov_sat'].append(n_locked / n_def)

        if all(dones): break

    for k in rec:
        if isinstance(rec[k], list) and len(rec[k]) > 0 and isinstance(rec[k][0], list):
            rec[k] = [np.array(x) for x in rec[k]]
        elif isinstance(rec[k], list) and len(rec[k]) > 0:
            rec[k] = np.array(rec[k])

    rec['death_step'] = {}; rec['hit_step'] = {}
    for i, al in enumerate(rec['off_alive']):
        for s, (a, h) in enumerate(zip(al, rec['off_hit'][i])):
            if h and i not in rec['hit_step']: rec['hit_step'][i] = s
            if not a and not h and i not in rec['death_step']: rec['death_step'][i] = s
    rec['hvt_x'] = env0.hvt.x; rec['hvt_y'] = env0.hvt.y; rec['hvt_z'] = env0.hvt.z
    rec['hit_count'] = env0.hit_count
    return rec

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '/tmp/v71_paper'
    for case, (env_s, torch_s) in SEEDS.items():
        base = os.path.join(root, case)
        if not os.path.isdir(base): continue
        print(f'>>> Replaying {case} (env={env_s}, torch={torch_s})...')
        rec = replay(env_s, torch_s)
        sm_path = os.path.join(base, 'summary.json')
        with open(sm_path) as f: sm = json.load(f)
        # update summary: safely extract hitter from new replay
        if rec['hit_step']:  # if this replay found a hit
            hitter = int(list(rec['hit_step'].keys())[0])
            hit_s = int(rec['hit_step'][hitter])
            best_d = float(np.min(rec['off_d_hvt'][hitter]))
        else:  # fallback to old summary
            hitter = sm.get('hitter', -1)
            hit_s = sm.get('hit_step', -1)
            best_d = sm.get('best_hvt_distance_m', None)
        sm['hit_step'] = hit_s; sm['hit_time_s'] = float(hit_s*0.01) if hit_s>=0 else None
        sm['best_hvt_distance_m'] = best_d
        with open(sm_path, 'w') as f: json.dump(sm, f, indent=2, ensure_ascii=False)
        # save npz
        out = {}
        for k, v in rec.items():
            if k in ('death_step','hit_step'): continue
            if isinstance(v, list): out[k] = np.array(v, dtype=object)
            else: out[k] = v
        np.savez_compressed(os.path.join(base, 'trajectory_data.npz'), **out)
        print(f'  => {case} saved (hits={rec["hit_count"]}, npz keys={len(out)})')
    print('All replays done.')
