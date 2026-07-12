#!/usr/bin/env python
"""Monte Carlo ablation for V71 RTA-MAPPO evaluation.

The terminal PN wrapper is intentionally kept for every variant.  The ablation
only changes structured observation channels visible to the trained policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
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


CASES = {
    "6v6": {"n": 6, "seed": 60015},
    "8v8": {"n": 8, "seed": 80047},
    "10v10": {"n": 10, "seed": 100000},
}

NOMINAL_SEEDS = {
    "6v6": [60031],
    "8v8": [80047],
    "10v10": [100007, 100013, 100015],
}

VARIANTS = {
    "full": "Full RTA-MAPPO observation prior",
    "no_threat_margin": "Zero top-K threat and primary-locker safety-margin channels",
    "no_team_pen_hit": "Zero team drawing-fire, penetration and hit-prior channels",
}

MODEL_DIR = Path(os.environ.get(
    "V71_MODEL_DIR",
    "outputs/results/fov_penetration/mappo/v71_locker_obs/run2/models",
))
HIDDEN = int(os.environ.get("V71_HIDDEN", "256"))
LAYER_N = int(os.environ.get("V71_LAYER_N", "3"))

_ENV = None
_RAW_ENV = None
_POLICIES = None
_CASE = None
_VARIANT = None
_MAX_STEPS = None
_JITTER = None
_MC_MODE = None


def make_policy_args():
    parser = get_macpo_config()
    return parser.parse_known_args([
        "--algorithm_name", "mappo",
        "--hidden_size", str(HIDDEN),
        "--layer_N", str(LAYER_N),
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--use_feature_normalization",
        "--use_recurrent_policy",
    ])[0]


def make_env(n: int, seed: int | None = None):
    os.environ.setdefault("FOV_REWARD_PROFILE", "v69teamsurvive")
    raw_env = FOVPenetrationEnv(
        config={"n_offensive": n, "n_defensive": n},
        scenario="scenario_1",
    )
    env = PhaseMaskedFOVWrapper(raw_env, mode="v65_strict_los")
    env = TerminalPNActionWrapper(env, gain=3.0, max_action=0.8)
    if seed is not None:
        raw_env.seed(seed)
    return raw_env, env


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
        policy.actor.load_state_dict(torch.load(actor_path, map_location=device), strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def ablate_obs(obs, variant: str):
    obs_arr = np.asarray(obs, dtype=np.float32).copy()
    if variant == "full":
        pass
    elif variant == "no_threat_margin":
        obs_arr[:, 9:26] = 0.0
    elif variant == "no_team_pen_hit":
        obs_arr[:, 26:29] = 0.0
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if isinstance(obs, list):
        return [obs_arr[i].copy() for i in range(obs_arr.shape[0])]
    return obs_arr


def perturb_initial_state(env, jitter_seed: int, jitter: dict):
    rng = np.random.RandomState(int(jitter_seed) ^ 0x5EEDC0DE)
    pos_xy = float(jitter.get("pos_xy_m", 35.0))
    pos_z = float(jitter.get("pos_z_m", 8.0))
    heading = float(jitter.get("heading_rad", 0.06))
    gamma = float(jitter.get("gamma_rad", 0.015))
    speed_frac = float(jitter.get("speed_frac", 0.035))

    for ac in list(env.offensives) + list(env.defensives):
        ac.x += rng.uniform(-pos_xy, pos_xy)
        ac.y += rng.uniform(-pos_xy, pos_xy)
        ac.z = float(np.clip(ac.z + rng.uniform(-pos_z, pos_z),
                             env.config["z_min"], env.config["z_max"]))
        ac.heading = float(np.arctan2(
            np.sin(ac.heading + rng.uniform(-heading, heading)),
            np.cos(ac.heading + rng.uniform(-heading, heading)),
        ))
        ac.gamma = float(np.clip(ac.gamma + rng.uniform(-gamma, gamma),
                                 env.config.get("gamma_min", -0.7),
                                 env.config.get("gamma_max", 0.7)))
        ac.v = float(np.clip(ac.v * (1.0 + rng.uniform(-speed_frac, speed_frac)),
                             ac.params["v_min"], ac.params["v_max"]))


def actor_actions(policies, obs, rnn_states, masks, variant: str):
    obs = ablate_obs(obs, variant)
    actions = []
    new_rnn = []
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
        new_rnn.append(hidden)
    return actions, new_rnn


def init_worker(case: str, variant: str, max_steps: int, jitter: dict, mc_mode: str):
    global _ENV, _RAW_ENV, _POLICIES, _CASE, _VARIANT, _MAX_STEPS, _JITTER, _MC_MODE
    torch.set_num_threads(1)
    _CASE = case
    _VARIANT = variant
    _MAX_STEPS = int(max_steps)
    _JITTER = dict(jitter)
    _MC_MODE = mc_mode
    meta = CASES[case]
    _RAW_ENV, _ENV = make_env(int(meta["n"]), int(meta["seed"]))
    _POLICIES = load_cloned_policies(_RAW_ENV)


def run_one(ep_idx: int) -> dict:
    case = _CASE
    variant = _VARIANT
    raw_env = _RAW_ENV
    env = _ENV
    policies = _POLICIES
    if env is None or raw_env is None or policies is None:
        raise RuntimeError("Worker not initialized")

    base_seed = int(CASES[case]["seed"])
    if _MC_MODE == "full_random":
        nominal_seed = base_seed + int(ep_idx)
        jitter_seed = nominal_seed
    elif _MC_MODE == "local_perturb":
        nominal_pool = NOMINAL_SEEDS[case]
        nominal_seed = int(nominal_pool[int(ep_idx) % len(nominal_pool)])
        jitter_seed = base_seed + int(ep_idx)
    else:
        raise ValueError(f"Unknown MC mode: {_MC_MODE}")

    raw_env.seed(nominal_seed)
    obs, _, _ = env.reset()

    if _MC_MODE == "local_perturb":
        perturb_initial_state(raw_env, jitter_seed, _JITTER)

        # Recompute masked observation after MC perturbation.
        if hasattr(raw_env, "_update_detection"):
            raw_env._update_detection()
        if hasattr(raw_env, "_update_lock_on_map"):
            raw_env._update_lock_on_map()
        raw_obs = raw_env._get_obs()
        obs = env.env._mask_obs(raw_obs)
        env._last_obs = env._copy_obs(obs)

    n_agents = raw_env.n_agents
    hvt = raw_env.hvt
    rnn_states = [torch.zeros(1, 1, HIDDEN) for _ in range(n_agents)]
    masks = [torch.ones(1, 1) for _ in range(n_agents)]
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in raw_env.offensives]
    min_step = [0 for _ in range(n_agents)]
    final_info = {}
    final_step = 0

    for step in range(_MAX_STEPS):
        actions, rnn_states = actor_actions(policies, obs, rnn_states, masks, variant)
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
            for i in range(n_agents)
        ]
        if all(dones):
            break

    best_agent = int(np.argmin(min_d))
    return {
        "variant": variant,
        "case": case,
        "episode": int(ep_idx),
        "nominal_seed": nominal_seed,
        "jitter_seed": jitter_seed,
        "mc_mode": _MC_MODE,
        "success": int(raw_env.hit_count > 0),
        "hit_count": int(raw_env.hit_count),
        "hit_indices": ";".join(str(int(i)) for i in getattr(raw_env, "hit_indices", [])),
        "done_reason": final_info.get("done_reason", ""),
        "final_step": int(final_step),
        "final_time_s": float(final_step * raw_env.dt),
        "offensive_alive": int(final_info.get("offensive_alive", -1)),
        "defensive_alive": int(final_info.get("defensive_alive", -1)),
        "best_agent": best_agent,
        "best_hvt_distance_m": float(min_d[best_agent]),
        "best_min_step": int(min_step[best_agent]),
    }


def write_rows_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["episode"]))


def summarize(rows: list[dict], elapsed_s: float, workers: int, jitter: dict) -> dict:
    n = len(rows)
    successes = [r for r in rows if int(r["success"]) == 1]
    best = np.asarray([float(r["best_hvt_distance_m"]) for r in rows], dtype=float)
    final_t = np.asarray([float(r["final_time_s"]) for r in rows], dtype=float)
    reasons = Counter(str(r["done_reason"]) for r in rows)
    success_times = np.asarray([float(r["final_time_s"]) for r in successes], dtype=float)
    return {
        "variant": rows[0]["variant"],
        "variant_note": VARIANTS[rows[0]["variant"]],
        "case": rows[0]["case"],
        "n_episodes": n,
        "workers": workers,
        "success_count": len(successes),
        "success_rate": len(successes) / max(n, 1),
        "best_hvt_distance_mean_m": float(best.mean()),
        "best_hvt_distance_median_m": float(np.median(best)),
        "best_hvt_distance_min_m": float(best.min()),
        "best_hvt_distance_p05_m": float(np.percentile(best, 5)),
        "best_hvt_distance_p95_m": float(np.percentile(best, 95)),
        "final_time_mean_s": float(final_t.mean()),
        "success_time_mean_s": float(success_times.mean()) if len(success_times) else None,
        "done_reason_counts": dict(reasons),
        "elapsed_s": float(elapsed_s),
        "mc_mode": rows[0].get("mc_mode", ""),
        "jitter": jitter,
    }


def run_condition(case: str, variant: str, n_eps: int, max_steps: int, workers: int,
                  out_root: Path, jitter: dict, mc_mode: str) -> dict:
    out_dir = out_root / f"{variant}_{case}_mc{n_eps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict] = []
    print(f"=== MC ablation {variant}/{case}: n={n_eps} workers={workers} ===", flush=True)
    with mp.Pool(processes=workers, initializer=init_worker,
                 initargs=(case, variant, max_steps, jitter, mc_mode)) as pool:
        for row in pool.imap_unordered(run_one, range(n_eps), chunksize=1):
            rows.append(row)
            done = len(rows)
            if done <= 5 or done % 25 == 0 or done == n_eps:
                succ = sum(int(r["success"]) for r in rows)
                best_min = min(float(r["best_hvt_distance_m"]) for r in rows)
                print(
                    f"{variant}/{case} done={done}/{n_eps} succ={succ} "
                    f"rate={succ / done:.3f} best_min={best_min:.2f}m",
                    flush=True,
                )
    elapsed_s = time.time() - started
    write_rows_csv(out_dir / "episodes.csv", rows)
    summary = summarize(rows, elapsed_s, workers, jitter)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--cases", nargs="+", default=["6v6", "8v8", "10v10"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-root", default="/tmp/v71_rta_mappo_ablation_mc")
    parser.add_argument(
        "--mc-mode",
        choices=["full_random", "local_perturb"],
        default="full_random",
        help="full_random resets each episode with an independent seed; "
             "local_perturb repeats representative successful seeds with small perturbations.",
    )
    parser.add_argument("--pos-xy-m", type=float, default=35.0)
    parser.add_argument("--pos-z-m", type=float, default=8.0)
    parser.add_argument("--heading-rad", type=float, default=0.06)
    parser.add_argument("--gamma-rad", type=float, default=0.015)
    parser.add_argument("--speed-frac", type=float, default=0.035)
    args = parser.parse_args()

    jitter = {
        "pos_xy_m": args.pos_xy_m,
        "pos_z_m": args.pos_z_m,
        "heading_rad": args.heading_rad,
        "gamma_rad": args.gamma_rad,
        "speed_frac": args.speed_frac,
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    all_summaries = {}
    started = time.time()
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for case in args.cases:
            if case not in CASES:
                raise ValueError(f"Unknown case: {case}")
            summary = run_condition(case, variant, args.episodes, args.max_steps,
                                    workers, out_root, jitter, args.mc_mode)
            all_summaries[f"{variant}_{case}"] = summary
            with (out_root / "summary_all_partial.json").open("w") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    top = {
        "elapsed_s": time.time() - started,
        "episodes_per_condition": args.episodes,
        "model_dir": str(MODEL_DIR),
        "hidden": HIDDEN,
        "layer_n": LAYER_N,
        "variants": args.variants,
        "variant_notes": VARIANTS,
        "cases": args.cases,
        "mc_mode": args.mc_mode,
        "summaries": all_summaries,
    }
    with (out_root / "summary_all.json").open("w") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    print(f"summary_all={out_root / 'summary_all.json'}", flush=True)


if __name__ == "__main__":
    main()
