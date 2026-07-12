import argparse
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np
import torch

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from phase_obs_wrapper import PhaseMaskedFOVWrapper
from terminal_pn_action_wrapper import TerminalPNActionWrapper
from eval_v70_gifs import load_policies


MODEL_DIR = 'outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models'
HIDDEN = 256


def velocity(aircraft):
    cos_gamma = np.cos(float(aircraft.gamma))
    return np.array([
        float(aircraft.v) * cos_gamma * np.cos(float(aircraft.heading)),
        float(aircraft.v) * cos_gamma * np.sin(float(aircraft.heading)),
        float(aircraft.v) * np.sin(float(aircraft.gamma)),
    ], dtype=np.float64)


def run_case(env_seed, torch_seed, max_steps=8000, max_action=0.8,
             anti_dive=0.6, terminal_only=True):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    raw_env = FOVPenetrationEnv()
    phase_env = PhaseMaskedFOVWrapper(raw_env, 'v65_strict_los')
    env = TerminalPNActionWrapper(
        phase_env,
        gain=3.0,
        max_action=max_action,
        anti_dive=anti_dive,
        terminal_only=terminal_only,
    )
    policies = load_policies(raw_env, MODEL_DIR, hidden_size=HIDDEN, layer_N=3)

    torch.manual_seed(torch_seed)
    raw_env.seed(env_seed)
    obs, _, _ = env.reset()
    rnn_states = [np.zeros((1, 1, HIDDEN), np.float32) for _ in range(4)]
    masks = [np.ones((1, 1), np.float32) for _ in range(4)]

    best = {
        'dist': 9999.0,
        'step': -1,
        'agent': -1,
        'terminal_flag': False,
        'guided_count': 0,
        'phase_terminal_count': 0,
        'base_action': None,
        'guided_action': None,
        'z': None,
        'gamma': None,
        'closing': None,
        'nearest_def_hvt': None,
    }
    terminal_first_step = [None] * 4
    guided_steps = [0] * 4
    terminal_steps = [0] * 4

    print(f'case env_seed={env_seed} torch_seed={torch_seed} max_action={max_action} '
          f'anti_dive={anti_dive} terminal_only={terminal_only}')

    for step in range(max_steps):
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

        flags = phase_env._terminal_flags()
        guided_actions, guided_count = env.guide_actions(acts)

        for agent_id, flag in enumerate(flags):
            if flag:
                terminal_steps[agent_id] += 1
                if terminal_first_step[agent_id] is None:
                    terminal_first_step[agent_id] = step
        for agent_id in range(4):
            if flags[agent_id] and raw_env.offensives[agent_id].alive and not raw_env.offensives[agent_id].hit_hvt:
                guided_steps[agent_id] += 1

        hvt = raw_env.hvt
        nearest_def_hvt = phase_env._nearest_def_hvt_dist()
        for agent_id, off in enumerate(raw_env.offensives):
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            if dist < best['dist']:
                target_vec = np.array([hvt.x - off.x, hvt.y - off.y, hvt.z - off.z], dtype=np.float64)
                closing = float(np.dot(target_vec, velocity(off)) / max(np.linalg.norm(target_vec), 1.0))
                best.update({
                    'dist': float(dist),
                    'step': int(step),
                    'agent': int(agent_id),
                    'terminal_flag': bool(flags[agent_id]),
                    'guided_count': int(guided_count),
                    'phase_terminal_count': int(sum(flags)),
                    'base_action': np.asarray(acts[agent_id], dtype=float).tolist(),
                    'guided_action': np.asarray(guided_actions[agent_id], dtype=float).tolist(),
                    'z': float(off.z),
                    'gamma': float(off.gamma),
                    'closing': closing,
                    'nearest_def_hvt': float(nearest_def_hvt),
                })

        rnn_states = new_rnn
        obs, _, _, _, dones, infos, _ = env.step(acts)
        masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]
        if raw_env.hit_count > 0 or all(dones):
            break
    print(f'  hit_count={raw_env.hit_count}')
    print(f'  best={best}')
    print(f'  terminal_first_step={terminal_first_step}')
    print(f'  terminal_steps={terminal_steps}')
    print(f'  guided_steps={guided_steps}')
    print(f'  hit_flags={[int(off.hit_hvt) for off in raw_env.offensives]}')
    print(f'  alive_flags={[int(off.alive) for off in raw_env.offensives]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', action='append', required=True,
                        help='env_seed:torch_seed, e.g. 50001:7')
    parser.add_argument('--max-action', type=float, default=0.8)
    parser.add_argument('--anti-dive', type=float, default=0.6)
    parser.add_argument('--all-phase-pn', action='store_true')
    args = parser.parse_args()
    for case in args.case:
        env_seed, torch_seed = [int(x) for x in case.split(':')]
        run_case(
            env_seed,
            torch_seed,
            max_action=args.max_action,
            anti_dive=args.anti_dive,
            terminal_only=not args.all_phase_pn,
        )


if __name__ == '__main__':
    main()