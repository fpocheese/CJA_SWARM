#!/usr/bin/env python
"""Dedicated V71 6v6 clone-weight evaluation.

This script intentionally avoids changing the generic training/eval entry
points. It creates a 6v6 FOV penetration environment directly, loads the V71
4-agent actor checkpoints, and clones them cyclically to six attackers:

  agent0 <- actor_agent0.pt
  agent1 <- actor_agent1.pt
  agent2 <- actor_agent2.pt
  agent3 <- actor_agent3.pt
  agent4 <- actor_agent0.pt
  agent5 <- actor_agent1.pt

It runs 50 episodes by default and writes a CSV/JSON summary.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from macpo.config import get_config as get_macpo_config
from envs.fov_penetration import FOVPenetrationEnv
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


MODEL_DIR = Path(os.environ.get(
    "V71_MODEL_DIR",
    "outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models",
))
N_EPS = int(os.environ.get("V71_6V6_N_EPS", "50"))
SEED_BASE = int(os.environ.get("V71_6V6_SEED_BASE", "60000"))
MAX_STEPS = int(os.environ.get("V71_6V6_MAX_STEPS", "8000"))
HIDDEN = int(os.environ.get("V71_HIDDEN", "256"))
LAYER_N = int(os.environ.get("V71_LAYER_N", "3"))
N_WORKERS = int(os.environ.get("V71_6V6_WORKERS", "10"))

OUT_ROOT = Path(os.environ.get("V71_6V6_OUT_ROOT", "outputs/v71_6v6_clone50"))

_ENV = None
_POLICIES = None


def make_raw_env(seed: int | None = None):
    # Keep this test local to this script: no scenario registry changes.
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": 6, "n_defensive": 6},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    if seed is not None:
        raw_env.seed(seed)
    return raw_env, env


def make_policy_args():
    parser = get_macpo_config()
    args = parser.parse_known_args([
        "--algorithm_name", "mappo",
        "--hidden_size", str(HIDDEN),
        "--layer_N", str(LAYER_N),
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--use_feature_normalization",
        "--use_recurrent_policy",
    ])[0]
    return args


def load_cloned_policies(raw_env):
    args = make_policy_args()
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
        actor_path = MODEL_DIR / f"actor_agent{src_agent}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        state_dict = torch.load(actor_path, map_location=device)
        policy.actor.load_state_dict(state_dict, strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def init_worker():
    global _ENV, _POLICIES
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    torch.set_num_threads(1)
    raw_env, env = make_raw_env()
    _ENV = env
    _POLICIES = load_cloned_policies(raw_env)


def run_one(seed: int):
    env = _ENV
    policies = _POLICIES
    if env is None or policies is None:
        raise RuntimeError("Worker was not initialized")

    env.seed(seed)
    obs, _, _ = env.reset()
    n_agents = env.n_agents
    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    hvt = env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    for step in range(MAX_STEPS):
        actions = []
        next_rnn = []
        for agent_id, policy in enumerate(policies):
            obs_tensor = torch.FloatTensor(np.asarray(obs[agent_id]).flatten()).unsqueeze(0)
            with torch.no_grad():
                action, _, hidden = policy.actor(
                    obs_tensor,
                    rnn_states[agent_id],
                    masks[agent_id],
                    deterministic=True,
                )
            actions.append(action.cpu().numpy().flatten())
            next_rnn.append(hidden)
        rnn_states = next_rnn

        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}

        for agent_id, off in enumerate(env.offensives):
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            if dist < min_d[agent_id]:
                min_d[agent_id] = dist
                min_step[agent_id] = final_step

        masks = [
            torch.tensor([[0.0 if dones[i] else 1.0]], dtype=torch.float32)
            for i in range(n_agents)
        ]
        if all(dones):
            break

    return {
        "seed": seed,
        "success": bool(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ",".join(str(i) for i in env.hit_indices),
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_min_dist_m": float(min(min_d)),
        "best_agent": int(np.argmin(min_d)),
        "best_min_step": int(min_step[int(np.argmin(min_d))]),
        "min_dist_per_agent_m": json.dumps([round(float(x), 3) for x in min_d]),
    }


def write_outputs(out_dir: Path, rows: list[dict], elapsed_s: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda item: item["seed"])
    csv_path = out_dir / "episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    successes = [row for row in rows_sorted if row["success"]]
    summary = {
        "model_dir": str(MODEL_DIR),
        "n_episodes": len(rows_sorted),
        "seed_base": SEED_BASE,
        "max_steps": MAX_STEPS,
        "n_workers": N_WORKERS,
        "success_count": len(successes),
        "success_rate": len(successes) / max(len(rows_sorted), 1),
        "success_seeds": [row["seed"] for row in successes],
        "hit_counts": [row["hit_count"] for row in successes],
        "best_min_dist_m": min(row["best_min_dist_m"] for row in rows_sorted),
        "best_episode": min(rows_sorted, key=lambda row: row["best_min_dist_m"]),
        "elapsed_s": elapsed_s,
    }
    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return csv_path, json_path, summary


def main():
    if N_EPS <= 0:
        raise ValueError("V71_6V6_N_EPS must be positive")
    workers = max(1, min(N_WORKERS, N_EPS))
    seeds = [SEED_BASE + i for i in range(N_EPS)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{stamp}_v71_6v6_clone"

    print("=== V71 6v6 clone-weight evaluation ===", flush=True)
    print(f"model_dir={MODEL_DIR}", flush=True)
    print(f"episodes={N_EPS} seed_base={SEED_BASE} max_steps={MAX_STEPS} workers={workers}", flush=True)
    print("clone_map: 0<-0, 1<-1, 2<-2, 3<-3, 4<-0, 5<-1", flush=True)
    start = time.time()

    rows = []
    if workers == 1:
        init_worker()
        for seed in seeds:
            row = run_one(seed)
            rows.append(row)
            print_episode(row, len(rows))
    else:
        with mp.Pool(processes=workers, initializer=init_worker) as pool:
            for row in pool.imap_unordered(run_one, seeds):
                rows.append(row)
                print_episode(row, len(rows))

    elapsed_s = time.time() - start
    csv_path, json_path, summary = write_outputs(out_dir, rows, elapsed_s)

    print("=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"episodes_csv={csv_path}", flush=True)
    print(f"summary_json={json_path}", flush=True)


def print_episode(row: dict, done_count: int):
    print(
        "ep_done={:03d} seed={} success={} hits={} reason={} steps={} "
        "alive={}/def{} best_min={:.2f}m agent={} step={}".format(
            done_count,
            row["seed"],
            int(row["success"]),
            row["hit_count"],
            row["done_reason"],
            row["final_step"],
            row["offensive_alive"],
            row["defensive_alive"],
            row["best_min_dist_m"],
            row["best_agent"],
            row["best_min_step"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
