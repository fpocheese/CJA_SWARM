# Revision Report

## Scope

- Main source revised: `newswarm_cja_english.tex`.
- Pre-revision backup: `newswarm_cja_english_before_revision.tex`.
- This round was restricted to (1) the relationship between the latent and executed actor-policy densities and (2) the definition of `a_{t,ij}^{cmd}`.
- No figures, bibliography entries, template files, code, configuration files, experimental data, reward functions, parameters, reference actions, heuristic mappings, labels, or cross-reference commands were modified.

## 1. LOS-tangent-plane acceleration component

**Location:** Cooperative dynamic task reconfiguration subsection, immediately after the burst LOS angular-velocity approximation (source line 505 after revision).

**Original:**

> Here, $a_{t,ij}^{\mathrm{cmd}}(t)$ denotes the acceleration command of offensive UAV $A_i$.

**Revised:**

> Here, $a_{t,ij}^{\mathrm{cmd}}(t)$ denotes the tangential component of the offensive UAV acceleration command in the LOS tangent plane.

**Effect:** The definition now matches the preceding description of $v_{t,ij}$ in the LOS tangent plane. No projection matrix, acceleration decomposition, control-channel mapping, constraint, formula, or parameter was added.

## 2. Latent and executed actor-policy densities

**Location:** TTA-MAPPO subsection, after the componentwise tanh action transformation and before the latent behavior-policy definition (source line 1189 after revision).

**Original issue:** The manuscript used $\widetilde\pi_{\theta_i}(\widetilde u_i\mid\bar o_i)$ for the latent Gaussian actor and $\pi_{\theta_i}(u_i\mid\bar o_i)$ for the bounded executed actor, but did not explicitly state their transformation relationship.

**Added text:**

> The latent actor policy $\widetilde\pi_{\theta_i}(\widetilde u_i\mid\bar o_i)$ is parameterized as a Gaussian distribution in the unconstrained latent-action space. The corresponding executed actor policy $\pi_{\theta_i}(u_i\mid\bar o_i)$ is induced from $\widetilde\pi_{\theta_i}$ through the same componentwise tanh transformation and change-of-variables rule. Therefore, $\widetilde\pi_{\theta_i}$ denotes the latent-action density, whereas $\pi_{\theta_i}$ denotes the density of the bounded maneuver action $u_i=\tanh(\widetilde u_i)$.

**Additional clarification after the existing behavior-density transformation:**

> The same transformation relates the latent actor density $\widetilde\pi_{\theta_i}$ to the executed actor density $\pi_{\theta_i}$.

No additional Jacobian formula was introduced, and the existing behavior-density change-of-variables formula was not altered.

## 3. Deterministic evaluation wording

**Location:** Deterministic HIL-evaluation paragraph following the PPO update (source lines 1305 and 1319 after revision).

**Original:** The text referred ambiguously to the “squashed mean latent action” of $\pi_{\theta_i}$ and defined $\mu_{\theta_i}$ only as the mean of an actor latent Gaussian distribution.

**Revised:** The maneuver command is described as the squashed mean of the latent Gaussian actor policy $\widetilde\pi_{\theta_i}$, and the explanation now states:

> where $\mu_{\theta_i}(\bar o_i)$ is the mean of the latent Gaussian actor policy $\widetilde\pi_{\theta_i}$, and the tanh transformation maps it to the bounded executed action.

The deterministic action formula remains unchanged:

$$
u_i^{\mathrm{eval}}(t)=\tanh\!\left[\mu_{\theta_i}(\bar o_i(t))\right].
$$

## Protected-content verification

- The three reference actions $u_i^{D,\mathrm{ref}}$, $u_i^{P,\mathrm{ref}}$, and $u_i^{S,\mathrm{ref}}$ and the mappings $\mathcal G_D$, $\mathcal G_P$, and $\mathcal G_S$ are byte-identical to the backup.
- $P_i^{\mathrm{hit}}$, $\kappa_h=0.6\,\mathrm{m^{-1}}$, all reward formulas and weights, the tanh mechanism, behavior policies, PPO ratio, entropy statement, CTDE wording, experimental results, and figure paths are unchanged.
- The joint executed actor factorization remains $\pi_\theta(\mathbf u\mid\bar{\mathbf o})=\prod_i\pi_{\theta_i}(u_i\mid\bar o_i)$.
- The PPO ratio continues to use the executed behavior densities $b_\theta/b_{\theta_{\mathrm{old}}}$.

## Search results

Searches were performed in `newswarm_cja_english.tex` after revision.

| Search item | Result |
|---|---|
| `\widetilde\pi_{\theta_i}(u_i` | No occurrence |
| `\pi_{\theta_i}(\widetilde u_i` | Literal substring matches at lines 1189 and 1194 only because it is contained inside the correct `\widetilde\pi_{\theta_i}(\widetilde u_i`; a boundary-aware search finds no un-tilded executed policy paired with `\widetilde u_i` |
| `acceleration command of offensive UAV` | No occurrence |
| `tangential component of the offensive UAV acceleration command` | Line 505 only |

All occurrences of $\widetilde\pi_{\theta_i}$, $\pi_{\theta_i}$, $\widetilde b_{\theta_i}$, and $b_{\theta_i}$ were also inspected. Whenever an action argument is present, latent densities are paired with $\widetilde u_i$ and executed densities are paired with $u_i$.

## Compilation and static checks

- Command: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/cja_actor_revision.ah6llJ newswarm_cja_english.tex`
- Result: successful; 22-page PDF generated at `/tmp/cja_actor_revision.ah6llJ/newswarm_cja_english.pdf`.
- Final log: no undefined control sequence, unresolved reference, unresolved citation, duplicate label, emergency stop, or fatal error.
- Static checks: no duplicate `\label`; every `\ref`/`\eqref` target and every `\cite` key resolves within the source.
- Nonfatal pre-existing layout warnings remain, including several overfull/underfull boxes and a PDF-version inclusion warning for `figures/algorithm_framework.pdf`; none was introduced by the two requested semantic changes.
