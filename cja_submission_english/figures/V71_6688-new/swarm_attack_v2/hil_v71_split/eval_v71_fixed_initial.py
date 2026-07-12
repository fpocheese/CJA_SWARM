#!/usr/bin/env python3
"""Single-process V71 evaluation from saved fixed initial scenario states."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from macpo.config import get_config as get_macpo_config
from envs.fov_penetration import FOVPenetrationEnv
from hil_v71_split.fixed_initial_state import reset_with_fixed_initial
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


def infer_case(npz_path: Path) -> tuple[str, int]:
    data = np.load(npz_path, allow_pickle=True)
    n = int(data["off_x"].shape[0])
    if n == 4:
        return "4v4", n
    if n == 6:
        return "6v6", n
    if n == 8:
        return "8v8", n
    raise ValueError(f"unsupported offensive count in {npz_path}: {n}")


def make_env(n: int):
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": n, "n_defensive": n},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    return raw_env, env


def make_policy_args(hidden_size: int, layer_n: int):
    parser = get_macpo_config()
    return parser.parse_known_args([
        "--algorithm_name", "mappo",
        "--hidden_size", str(hidden_size),
        "--layer_N", str(layer_n),
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--use_feature_normalization",
        "--use_recurrent_policy",
    ])[0]


def load_policies(raw_env, model_dir: Path, hidden_size: int, layer_n: int):
    args = make_policy_args(hidden_size, layer_n)
    device = torch.device("cpu")
    policies = []
    for agent_id in range(raw_env.n_agents):
        policy = R_MAPPOPolicy(
            args,
            raw_env.observation_space[agent_id],
            raw_env.share_observation_space[agent_id],
            raw_env.action_space[agent_id],
            device=device,
        )
        src_agent = agent_id % 4
        actor_path = model_dir / f"actor_agent{src_agent}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"missing actor checkpoint: {actor_path}")
        policy.actor.load_state_dict(torch.load(actor_path, map_location=device), strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def run_case(npz_path: Path, model_dir: Path, max_steps: int, hidden_size: int, layer_n: int):
    case, n = infer_case(npz_path)
    raw_env, env = make_env(n)
    policies = load_policies(raw_env, model_dir, hidden_size, layer_n)
    obs, _, _ = reset_with_fixed_initial(raw_env, env, npz_path, seed=0)

    rnn_states = [torch.zeros(1, 1, hidden_size) for _ in range(n)]
    masks = [torch.ones(1, 1) for _ in range(n)]
    hvt = raw_env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in raw_env.offensives]
    min_step = [0 for _ in range(n)]
    final_info = {}
    final_step = 0
    actions_log = []

    for step in range(max_steps):
        actions = []
        next_rnn = []
        for agent_id, policy in enumerate(policies):
            obs_tensor = torch.as_tensor(np.asarray(obs[agent_id], dtype=np.float32).reshape(1, -1))
            with torch.no_grad():
                action, _, hidden = policy.actor(
                    obs_tensor,
                    rnn_states[agent_id],
                    masks[agent_id],
                    deterministic=True,
                )
            action_np = action.cpu().numpy().reshape(-1).astype(np.float32)
            actions.append(action_np)
            next_rnn.append(hidden)
        if step < 5:
            actions_log.append(np.asarray(actions).astype(float).tolist())
        rnn_states = next_rnn
        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}
        for i, off in enumerate(raw_env.offensives):
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
                min_step[i] = final_step
        masks = [
            torch.tensor([[0.0 if dones[i] else 1.0]], dtype=torch.float32)
            for i in range(n)
        ]
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    return {
        "case": case,
        "npz_path": str(npz_path),
        "model_dir": str(model_dir),
        "success": bool(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": [int(i) for i in getattr(raw_env, "hit_indices", [])],
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * raw_env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_hvt_distance_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
        "min_dist_per_agent_m": [round(float(x), 3) for x in min_d],
        "initial_action_first5": actions_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", nargs="+", required=True)
    parser.add_argument("--model-dir", default="outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models")
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layer-n", type=int, default=3)
    parser.add_argument("--out", default="/tmp/v71_fixed_initial_eval.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.time()
    results = [
        run_case(Path(path), Path(args.model_dir), args.max_steps, args.hidden_size, args.layer_n)
        for path in args.npz
    ]
    output = {
        "elapsed_s": time.time() - started,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
