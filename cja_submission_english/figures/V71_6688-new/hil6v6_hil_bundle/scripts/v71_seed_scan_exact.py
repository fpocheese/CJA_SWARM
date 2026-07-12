import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np
import torch

torch.manual_seed(42)

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from phase_obs_wrapper import PhaseMaskedFOVWrapper
from terminal_pn_action_wrapper import TerminalPNActionWrapper
from eval_v70_gifs import load_policies

MODEL_DIR = 'outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models'
HIDDEN = 256
N_EPS = int(os.environ.get('V71_N_EPS', '50'))

env0 = FOVPenetrationEnv()
env = TerminalPNActionWrapper(PhaseMaskedFOVWrapper(env0, 'v65_strict_los'), gain=3.0, max_action=0.8)
policies = load_policies(env0, MODEL_DIR, hidden_size=HIDDEN, layer_N=3)
print(f'Loaded {len(policies)} policies', flush=True)

hits = []
for ep in range(N_EPS):
    env0.seed(50000 + ep)
    obs, _, _ = env.reset()
    rnn_states = [np.zeros((1, 1, HIDDEN), np.float32) for _ in range(4)]
    masks = [np.ones((1, 1), np.float32) for _ in range(4)]
    best_dist = 9999.0
    for _ in range(8000):
        acts = []
        new_rnn = []
        for i in range(4):
            o = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                act, _, rnn_h = policies[i].actor(o, torch.FloatTensor(rnn_states[i]), torch.FloatTensor(masks[i]))
            acts.append(act.squeeze(0).numpy())
            new_rnn.append(rnn_h.numpy())
        rnn_states = new_rnn
        obs, _, _, _, dones, _, _ = env.step(acts)
        masks = [np.array([[0.0 if dones[i] else 1.0]], np.float32) for i in range(4)]
        for off in env0.offensives:
            dist = off.distance_to(env0.hvt.x, env0.hvt.y, env0.hvt.z)
            if dist < best_dist:
                best_dist = dist
        if all(dones):
            break
    hit = env0.hit_count > 0
    if hit:
        hits.append((ep, 50000 + ep, best_dist))
    print(f'ep{ep:03d}: hit={int(hit)}  best_dist={best_dist:.1f}m', flush=True)

print(f'=== HIT EPISODES: {hits} ===', flush=True)