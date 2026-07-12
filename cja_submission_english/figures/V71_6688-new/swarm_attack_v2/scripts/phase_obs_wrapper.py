"""Phase-gated observation wrapper for FOV penetration training.

This wrapper does not change environment dynamics, hit/kill logic, action space,
or observation dimensionality. It only masks what the policy sees.
"""

from __future__ import annotations

import numpy as np


class PhaseMaskedFOVWrapper:
    """Mask offensive-agent observations by penetration phase.

    V71 obs layout (30):
      0:5   self kinematics
      5:9   HVT guidance: d_az, d_el, closing, align_cos
      9:19  top-2 threat features
      19:26 primary locker: q*, V_c*, |omega*|, Gamma*, rho_norm, sin_az_body, is_locked
      26:29 team / priors
      29    time

    Phase rule per agent:
      - penetration: off-HVT distance > nearest alive defender-HVT distance
      - terminal:    off-HVT distance <= nearest alive defender-HVT distance

    Penetration obs keeps self + threat + primary_locker + team + time + coarse HVT closing/align.
    Terminal obs keeps only HVT LOS angular rates (d_az, d_el), forcing the actor
    to rely on PN-like terminal guidance cues.
    """

    def __init__(self, env, mode: str = "v60_phase"):
        self.env = env
        self.mode = mode
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        obs, share_obs, avail = self.env.reset()
        return self._mask_obs(obs), share_obs, avail

    def step(self, actions):
        obs, share_obs, rewards, costs, dones, infos, avail = self.env.step(actions)
        masked_obs = self._mask_obs(obs)
        terminal_count = int(sum(self._terminal_flags()))
        for info in infos:
            if isinstance(info, dict):
                info["phase_terminal_agents"] = terminal_count
                info["obs_mask_mode"] = self.mode
        return masked_obs, share_obs, rewards, costs, dones, infos, avail

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _nearest_def_hvt_dist(self) -> float:
        alive_dists = []
        hvt = self.env.hvt
        for defender in self.env.defensives:
            if defender.alive:
                alive_dists.append(defender.distance_to(hvt.x, hvt.y, hvt.z))
        if not alive_dists:
            return float("inf")
        return float(min(alive_dists))

    def _terminal_flags(self):
        hvt = self.env.hvt
        nearest_def = self._nearest_def_hvt_dist()
        flags = []
        for off in self.env.offensives:
            if not off.alive or off.hit_hvt:
                flags.append(False)
                continue
            off_dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            flags.append(bool(off_dist <= nearest_def))
        return flags

    def _mask_obs(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32)
        flags = self._terminal_flags()
        masked = np.zeros_like(obs_arr, dtype=np.float32)

        for agent_id in range(min(len(obs_arr), len(flags))):
            source = obs_arr[agent_id]
            target = masked[agent_id]
            if flags[agent_id]:
                # Terminal guidance: HVT LOS angular rates only.
                target[5:7] = source[5:7]
            else:
                # Penetration: self, threats, primary_locker, team priors, time,
                # and coarse HVT closure/alignment; no HVT LOS angular-rate guidance yet.
                target[0:5] = source[0:5]
                target[7:9] = source[7:9]
                target[9:26] = source[9:26]   # threats (9:19) + primary_locker (19:26)
                target[26:30] = source[26:30]  # team priors (26:29) + time (29)

        if isinstance(obs, list):
            return [masked[i].copy() for i in range(masked.shape[0])]
        return masked
