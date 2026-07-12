#!/usr/bin/env python3
"""Server-side V71 environment for split HIL closed-loop simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.fov_penetration import FOVPenetrationEnv
from hil_v71_split.fixed_initial_state import reset_with_fixed_initial
from hil_v71_split.hil_protocol import JsonLineSocket, listen
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


CASES = {
    "4v4": (4, 4, 90000),
    "6v6": (6, 6, 60000),
    "8v8": (8, 8, 80000),
}


def make_env(case: str):
    n_off, n_def, _ = CASES[case]
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": n_off, "n_defensive": n_def},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    return raw_env, env


def accept_clients(host: str, port: int, n_agents: int) -> dict[int, JsonLineSocket]:
    server = listen(host, port)
    clients: dict[int, JsonLineSocket] = {}
    print(f"[server] listening on {host}:{port}, waiting for {n_agents} policy nodes", flush=True)
    try:
        while len(clients) < n_agents:
            sock, addr = server.accept()
            sock.setsockopt(6, 1, 1)  # IPPROTO_TCP, TCP_NODELAY
            peer = JsonLineSocket(sock)
            hello = peer.recv()
            if hello.get("type") != "hello":
                raise ValueError(f"client {addr} did not send hello: {hello}")
            agent_id = int(hello["agent_id"])
            if agent_id < 0 or agent_id >= n_agents:
                raise ValueError(f"invalid agent_id {agent_id}, expected [0,{n_agents})")
            if agent_id in clients:
                raise ValueError(f"duplicate agent_id {agent_id}")
            clients[agent_id] = peer
            print(f"[server] agent {agent_id} connected from {addr}, hello={hello}", flush=True)
    finally:
        server.close()
    return clients


def run_episode(raw_env, env, clients: dict[int, JsonLineSocket], seed: int,
                max_steps: int, episode: int, fixed_initial_npz: str | None = None):
    if fixed_initial_npz:
        obs, _, _ = reset_with_fixed_initial(raw_env, env, fixed_initial_npz, seed=seed)
    else:
        env.seed(seed)
        obs, _, _ = env.reset()
    n_agents = env.n_agents
    masks = [1.0 for _ in range(n_agents)]
    hvt = raw_env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in raw_env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    for step in range(max_steps):
        for agent_id in range(n_agents):
            clients[agent_id].send({
                "type": "obs",
                "episode": int(episode),
                "step": int(step),
                "seed": int(seed),
                "mask": float(masks[agent_id]),
                "obs": np.asarray(obs[agent_id], dtype=np.float32).reshape(-1).astype(float).tolist(),
            })

        actions = [None for _ in range(n_agents)]
        for agent_id in range(n_agents):
            msg = clients[agent_id].recv()
            if msg.get("type") != "action":
                raise ValueError(f"agent {agent_id} sent unexpected message: {msg}")
            if int(msg.get("agent_id", -1)) != agent_id:
                raise ValueError(f"agent id mismatch: expected {agent_id}, got {msg}")
            actions[agent_id] = np.asarray(msg["action"], dtype=np.float32)

        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_step = step + 1
        final_info = infos[0] if infos else {}
        for i, off in enumerate(raw_env.offensives):
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
                min_step[i] = final_step
        masks = [0.0 if bool(dones[i]) else 1.0 for i in range(n_agents)]
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    return {
        "seed": int(seed),
        "episode": int(episode),
        "success": bool(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": [int(i) for i in getattr(raw_env, "hit_indices", [])],
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * raw_env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_min_dist_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
        "min_dist_per_agent_m": [round(float(x), 3) for x in min_d],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), default="4v4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--fixed-initial-npz", default=None,
                        help="trajectory_data.npz whose first frame is used as the fixed initial scenario")
    parser.add_argument("--out", default="/tmp/v71_hil_split_summary.json")
    args = parser.parse_args()

    n_agents, _, seed_base = CASES[args.case]
    seed0 = seed_base if args.seed is None else int(args.seed)
    raw_env, env = make_env(args.case)
    clients = accept_clients(args.host, args.port, n_agents)
    started = time.time()
    summaries = []
    try:
        for ep in range(int(args.episodes)):
            summary = run_episode(raw_env, env, clients, seed0 + ep, int(args.max_steps), ep,
                                  fixed_initial_npz=args.fixed_initial_npz)
            summaries.append(summary)
            print(f"[server] episode={ep} seed={seed0 + ep} success={int(summary['success'])} "
                  f"reason={summary['done_reason']} best={summary['best_min_dist_m']:.2f}m", flush=True)
    finally:
        for peer in clients.values():
            try:
                peer.send({"type": "close"})
            except Exception:
                pass
            peer.close()

    output = {
        "case": args.case,
        "episodes": len(summaries),
        "success_count": sum(1 for item in summaries if item["success"]),
        "success_rate": sum(1 for item in summaries if item["success"]) / max(len(summaries), 1),
        "elapsed_s": time.time() - started,
        "summaries": summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[server] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
