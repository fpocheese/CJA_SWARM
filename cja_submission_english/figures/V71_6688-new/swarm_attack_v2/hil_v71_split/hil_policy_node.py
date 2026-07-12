#!/usr/bin/env python3
"""NX-side V71 offensive policy node.

The node owns one offensive aircraft policy. It receives the masked observation
from the server-side environment process and returns the raw actor action.
The server remains responsible for terminal PN wrapping, dynamics, defenders,
and hit/kill logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gym
except ModuleNotFoundError:
    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = Box

    class _Gym:
        spaces = _Spaces()

    gym = _Gym()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "on-policy-main"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

try:
    from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from macpo.config import get_config as get_mappo_config
except Exception:
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.config import get_config as get_mappo_config
from hil_v71_split.hil_protocol import connect


def make_policy_args(hidden_size: int, layer_n: int):
    parser = get_mappo_config()
    return parser.parse_known_args([
        "--algorithm_name", "mappo",
        "--hidden_size", str(hidden_size),
        "--layer_N", str(layer_n),
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--use_feature_normalization",
        "--use_recurrent_policy",
    ])[0]


def load_actor(model_dir: Path, source_agent: int, hidden_size: int, layer_n: int,
               obs_dim: int, share_obs_dim: int, action_dim: int):
    args = make_policy_args(hidden_size, layer_n)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    share_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    actor_path = model_dir / f"actor_agent{source_agent}.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"missing actor checkpoint: {actor_path}")
    policy.actor.load_state_dict(torch.load(actor_path, map_location="cpu"), strict=False)
    policy.actor.eval()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5500)
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--source-agent", type=int, default=None,
                        help="V71 checkpoint index to load; defaults to agent_id %% 4")
    parser.add_argument("--model-dir", default="outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layer-n", type=int, default=3)
    parser.add_argument("--obs-dim", type=int, default=30)
    parser.add_argument("--share-obs-dim", type=int, default=77)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    source_agent = args.source_agent if args.source_agent is not None else args.agent_id % 4
    policy = load_actor(Path(args.model_dir), int(source_agent), args.hidden_size, args.layer_n,
                        args.obs_dim, args.share_obs_dim, args.action_dim)
    rnn_state = torch.zeros(1, 1, args.hidden_size)

    peer = connect(args.server_host, args.server_port)
    peer.send({
        "type": "hello",
        "agent_id": int(args.agent_id),
        "source_agent": int(source_agent),
        "pid": os.getpid(),
    })
    try:
        while True:
            msg = peer.recv()
            kind = msg.get("type")
            if kind == "close":
                break
            if kind != "obs":
                raise ValueError(f"unexpected message: {msg}")
            obs = torch.as_tensor(np.asarray(msg["obs"], dtype=np.float32).reshape(1, -1))
            mask_value = float(msg.get("mask", 1.0))
            mask = torch.tensor([[mask_value]], dtype=torch.float32)
            with torch.no_grad():
                action, _, rnn_state = policy.actor(
                    obs,
                    rnn_state,
                    mask,
                    deterministic=bool(args.deterministic),
                )
            peer.send({
                "type": "action",
                "agent_id": int(args.agent_id),
                "episode": int(msg.get("episode", 0)),
                "step": int(msg.get("step", 0)),
                "action": action.cpu().numpy().reshape(-1).astype(float).tolist(),
            })
    finally:
        peer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
