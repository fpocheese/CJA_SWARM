#!/usr/bin/env python
"""
eval_v70_gifs.py — v70_team_survive 모델 평가 + GIF 생성
GIF는 episode 종료(done)까지 전체 회합을 그립니다.
PhaseMaskedFOVWrapper(v65_strict_los) + TerminalPNActionWrapper(gain=3.0) 사용.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


def canvas_to_rgb(fig):
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        return buf.reshape((h, w, 3))
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        return buf.reshape((h, w, 4))[:, :, 1:4]


def load_policies(raw_env, model_dir, hidden_size=256, layer_N=3):
    parser = get_macpo_config()
    args = parser.parse_known_args([
        '--algorithm_name', 'mappo',
        '--hidden_size', str(hidden_size),
        '--layer_N', str(layer_N),
        '--lr', '5e-4',
        '--critic_lr', '5e-4',
        '--use_feature_normalization',
        '--use_recurrent_policy',
    ])[0]
    obs_space = raw_env.observation_space[0]
    share_obs_space = raw_env.share_observation_space[0]
    act_space = raw_env.action_space[0]
    policies = []
    for i in range(raw_env.n_agents):
        policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space,
                               device=torch.device('cpu'))
        pt = torch.load(os.path.join(model_dir, f'actor_agent{i}.pt'), map_location='cpu')
        policy.actor.load_state_dict(pt, strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def run_episode(env, raw_env, policies, seed, gif_path,
                hidden_size=256, max_steps=8000,
                fps=10, speedup=4.0, frame_stride=8):
    """运行一个 episode 并导出 GIF（到 done 为止）"""
    env.seed(seed)
    obs, share_obs, _ = env.reset()

    n_agents = env.n_agents
    rnn_states = [np.zeros((1, 1, hidden_size), dtype=np.float32) for _ in range(n_agents)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(n_agents)]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    gif_fps = max(1, int(round(fps * speedup)))
    writer = imageio.get_writer(gif_path, mode='I', fps=gif_fps)

    # 渲染初始帧
    render_frame(ax, raw_env, step_num=0, show_fov=True)
    writer.append_data(canvas_to_rgb(fig))
    frame_count = 1

    ep_reward = 0.0
    info = None

    for step in range(max_steps):
        actions = []
        for i in range(n_agents):
            with torch.no_grad():
                obs_t = np.array(obs[i]).reshape(1, -1)
                action, _, rnn_out = policies[i].actor(
                    obs_t, rnn_states[i], masks[i], deterministic=True)
                actions.append(action.squeeze().numpy())
                rnn_states[i] = rnn_out.numpy()

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        ep_reward += float(np.mean(rewards))
        info = infos[0] if infos else {}

        # 每 frame_stride 步渲染一帧，或最后一帧必须渲染
        if ((step + 1) % frame_stride == 0) or dones[0]:
            render_frame(ax, raw_env, step_num=step + 1, show_fov=True)
            writer.append_data(canvas_to_rgb(fig))
            frame_count += 1

        if dones[0]:
            break

    writer.close()
    plt.close(fig)

    if info is None:
        info = {}
    return {
        'success':        info.get('success', False),
        'done_reason':    info.get('done_reason', 'unknown'),
        'hit_count':      info.get('hit_count', 0),
        'off_alive':      info.get('offensive_alive', 0),
        'steps':          raw_env.current_step,
        'reward':         ep_reward,
        'frames':         frame_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
        default='outputs/results/fov_penetration/mappo/v70_team_survive/run1/models')
    parser.add_argument('--out_dir', type=str,
        default='outputs/gifs/v70_team_survive_eval')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=8000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--speedup', type=float, default=4.0,
        help='GIF 播放加速倍率（默认4x，保证文件不太大）')
    parser.add_argument('--frame_stride', type=int, default=8,
        help='每隔多少仿真步渲染一帧（默认8步=0.08s/帧）')
    parser.add_argument('--obs_mask', type=str, default='v65_strict_los')
    parser.add_argument('--pn_gain', type=float, default=3.0)
    parser.add_argument('--pn_max_action', type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 构建 raw_env（供渲染用）和 wrapped_env（供策略决策用）
    raw_env = FOVPenetrationEnv(scenario='scenario_1')
    wrapped_env = PhaseMaskedFOVWrapper(raw_env, mode=args.obs_mask)
    wrapped_env = TerminalPNActionWrapper(
        wrapped_env, gain=args.pn_gain, max_action=args.pn_max_action)

    policies = load_policies(raw_env, args.model_dir,
                             hidden_size=args.hidden_size, layer_N=args.layer_N)

    print(f'Model:      {args.model_dir}')
    print(f'Out dir:    {args.out_dir}')
    print(f'Episodes:   {args.n_episodes}, seed_base={args.seed_base}')
    print(f'Wrappers:   obs={args.obs_mask}, PN gain={args.pn_gain} max_action={args.pn_max_action}')
    print(f'GIF params: fps={args.fps} x{args.speedup}, stride={args.frame_stride}')

    results = []
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        gif_path = os.path.join(args.out_dir, f'ep{ep:02d}_seed{seed}.gif')
        r = run_episode(wrapped_env, raw_env, policies, seed, gif_path,
                        hidden_size=args.hidden_size,
                        max_steps=args.max_steps,
                        fps=args.fps,
                        speedup=args.speedup,
                        frame_stride=args.frame_stride)
        status = 'SUCCESS' if r['success'] else r['done_reason']
        print(f"  Ep{ep:02d} seed={seed}: {status}, steps={r['steps']}, "
              f"hits={r['hit_count']}, alive={r['off_alive']}, "
              f"reward={r['reward']:.1f}, frames={r['frames']} -> {os.path.basename(gif_path)}")
        results.append(r)

    successes = sum(1 for r in results if r['success'])
    avg_steps = np.mean([r['steps'] for r in results])
    print(f'\n=== SUMMARY ===')
    print(f'Success: {successes}/{args.n_episodes} = {successes*100/max(1,args.n_episodes):.1f}%')
    print(f'Avg steps: {avg_steps:.1f}')

    summary = {
        'model_dir': args.model_dir,
        'n_episodes': args.n_episodes,
        'seed_base': args.seed_base,
        'obs_mask': args.obs_mask,
        'pn_gain': args.pn_gain,
        'success_count': successes,
        'success_rate': successes / max(1, args.n_episodes),
        'episodes': results,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary -> {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
