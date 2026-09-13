# Formula Condensation Report

## 1. Files and authorized scope

- Revised source: `newswarm_cja_english.tex`
- Complete pre-revision backup: `newswarm_cja_english_before_formula_condensation.tex`
- Compiled output: `newswarm_cja_english.pdf`
- Modified scope: Sections 2, 3, and 4 only, from `Cooperative UAV-Swarm Penetration Modeling under Multi-Interceptor FOV Constraints` to immediately before `Simulation Results and Analysis`.

An automated byte comparison confirms that the preamble, title and author material, abstract, keywords, Introduction, the complete simulation section, Conclusions, and references are unchanged.

## 2. Quantitative results

Displayed-math counts include top-level `equation`, `equation*`, `align`, `align*`, `gather`, `multline`, and `\[...\]` environments. Narrative word counts exclude displayed mathematics, inline mathematics, figures, tables, captions, headings, comments, and LaTeX commands; the same extraction method was applied before and after revision.

| Section | Display environments before | Display environments after | Reduction | Narrative words before | Narrative words after | Word reduction |
|---|---:|---:|---:|---:|---:|---:|
| Section 2 | 21 | 17 | 4 (19.05%) | 362 | 307 | 55 (15.19%) |
| Section 3 | 51 | 36 | 15 (29.41%) | 1,070 | 921 | 149 (13.93%) |
| Section 4 | 71 | 52 | 19 (26.76%) | 1,440 | 1,110 | 330 (22.92%) |
| **Total** | **143** | **105** | **38 (26.57%)** | **2,872** | **2,338** | **534 (18.59%)** |

- PDF before revision: **20 pages**
- PDF after revision: **19 pages**

The result is one full compiled page shorter. Further display reduction was not forced because the safety-margin chain, potential-game derivation, bounded behavior-policy density, and PPO update must remain independently readable and reproducible.

## 3. Section 2 condensation

### Merged display formulas

1. The offensive and defensive state vectors, previously two `equation` environments, were combined into one `align` environment. Both states and their original labels `eq:att-state` and `eq:def-state` remain.
2. The three UAV point-mass dynamics for $\dot V_i$, $\dot\psi_i$, and $\dot\gamma_i$ were combined into one `align` environment. Their equations, controls, gravity terms, and labels `eq:dyn-V`, `eq:dyn-psi`, and `eq:dyn-gamma` remain unchanged in meaning.

### Converted to prose

3. The standalone information-update formula
   $f_{ij}^{\mathrm{info}}=20\delta_{ij}^{\mathrm{det}}+2(1-\delta_{ij}^{\mathrm{det}})$
   was converted to the sentence stating $20\,\mathrm{Hz}$ under onboard detection and $2\,\mathrm{Hz}$ under ground-radar indication. This quantity was not used by any later theoretical formula. Its obsolete label and self-reference were removed.

### Protected content

The body-axis vector, relative position and range, body LOS angle, algebraic FOV margin $q_{ij}$, occupancy $s_{ij}$, detection indicator $\delta_{ij}^{\mathrm{det}}$, detectable-survivor set, Hungarian initialization, nearest-target switching rule, internal defender assignment $m_j(t)$, PN guidance, LOS angular velocity, and destruction/strike criteria remain explicit.

## 4. Section 3 condensation

### Safety-margin derivation

The following same-level formulas were merged without deleting intermediate reasoning:

1. $\xi_{ij}$, $\dot\xi_{ij}=A_c\xi_{ij}$, and the matrix $A_c$ were combined into one local-state formula group.
2. The output vector $c=[1,0]^\top$ was moved inline, while the relations $q_{ij}=c^\top\xi_{ij}$, the state-transition prediction, the predicted FOV margin, $Y_{ij}$, and $Z_{ij}$ were combined into one aligned prediction chain.
3. The analytic constant-integrator result for $Y_{ij}$ and the equivalent expression $Z_{ij}=q_{ij}+t_{go,ij}\dot q_{ij}$ were combined into one display.
4. The radial gate $G_{ij}^{\mathrm{rad}}$ and single-interceptor gated margin $Z_{ij}^{\mathrm{det}}$ were combined into one display.
5. The individual risk penalty $\Psi_i^{\mathrm{cone}}$ and swarm risk $c_t^{\mathrm{cone}}$ were combined into one display.

The complete chain remains explicit:

`FOV geometry -> local state -> finite pairwise horizon -> state transition -> prediction-horizon margin -> radial gating -> normalized smooth maximum -> individual and swarm risk`.

The definitions of $\dot\rho_{ij}$, $V_{c,ij}$, the full piecewise $t_{go,ij}$, $\varepsilon_v$, $T_p$, $1/N_I$, $\beta_Z$, $M_c$, and the non-closing condition were retained.

### Capture inference and task utilities

6. $\bar\rho_{ij}$ and $\bar V_{c,ij}$ were combined into one normalization formula group.
7. The three auxiliary self-cost indicators $E_{ij}$, $D_{ij}$, and $K_{ij}$ were combined into one aligned display; their physical interpretations, thresholds, smoothing gains, and exponential/sigmoid forms remain.
8. The decoy-containment, primary-attack, and concealment-evasion utilities $U_i^D$, $U_i^P$, and $U_i^S$ were combined into a single `align` environment. All three labels, weights, benefit terms, risk terms, and task meanings remain.

### Potential-game protection

No core potential-game derivation was deleted or abbreviated. The following block is byte-for-byte identical to the backup:

- entropy-regularized individual payoff $J_i$;
- task potential $\Phi_{\mathrm{task}}$;
- Proposition 1;
- fixed instantaneous offensive--defensive state;
- unilateral mixed-task deviation;
- equality of individual-payoff and potential differences;
- Lagrange multiplier and simplex constraint;
- Lagrangian expression;
- stationarity condition;
- proportional exponential response and normalized Logit response;
- potential maximization $\mu^\star=\arg\max\Phi_{\mathrm{task}}$.

The capture-intent score, Softmax capture likelihood, integrated capture pressure, three task utilities, task-probability vector, simplex constraint, and Logit update remain connected in their original logical order.

## 5. Section 4 condensation

### Value functional and repeated expansions

1. $e_i$ and $N_{\mathrm{eff}}$ were combined into one aligned display.
2. $J_{\mathrm{sat}}=\lambda_1N_{\mathrm{eff}}+\lambda_2N_{\mathrm{eff}}^2$ was converted to an inline definition, and $J_T$ now references $J_{\mathrm{sat}}$ rather than repeating its two terms.
3. The continuous running utility was named $L(t)$ and displayed once. The operation-level objective is now written as $\mathcal J=\mathbb E[\int_0^T L(t)\,dt+J_T]$.
4. The discrete theoretical utility reward is now $r_t^{\mathrm{util}}=\Delta t\,L(t)$, eliminating a second expansion of the same four running-utility sums.
5. The terminal reward is written as $r^T=J_T$, eliminating a third expansion of the already defined terminal-value components.

The destruction indicator, destruction time, pre-destruction salvageability, piecewise ineffective-sacrifice definition, total ineffective sacrifice, total loss, terminal value, and operation-level utility remain explicit.

### Action mapping

6. The axial, pitch-normal, and yaw-normal physical action maps were combined into one aligned display labeled `eq:act-map`. The redundant standalone symbolic map $\mathcal T$ was removed, while every piecewise branch, acceleration limit, gravity offset, and normalized-action meaning remains.
7. The zero-action trim relation was converted to inline mathematics immediately after the action map.

### Observation table

8. The self, HVT, threat, team, and dynamic-task-prior components were reorganized into a compact, unnumbered `tabularx` summary. It is non-floating and therefore does not alter the numbering of existing tables.
9. The core observation equation $o_i=[o_i^{\mathrm{self}}\mid o_i^H\mid o_i^{\mathrm{thr}}\mid o_i^{\mathrm{team}}]$ and enhanced actor input $\bar o_i=[o_i^\top,\mu_i^\top]^\top$ were combined into one display.
10. The two threat-normalization definitions were combined into one display.

The table retains every feature and dimension: self state (5), HVT geometry (4), Top-$K$ threat tuples ($4K$), team indices (2), and task prior (3). The HVT LOS-rate/closing/alignment formulas, inferred-capture ordering, $K=3$, zero padding, normalization constants, and the distinction between actor observation and centralized critic prior remain in the text.

### Reward and network formulas

11. Individual, team, and total running rewards were combined into one `align` environment without deleting any reward component or weight.
12. The critic encoder, scaled Softmax attention weights, and cross-agent weighted aggregation were combined into one aligned attention group; the residual critic output remains a separate formula. Query/key/value sources, $\sqrt d$, Softmax, aggregation, residual connection, and critic-only use remain explicit.
13. The primary-attack, decoy-containment, and concealment-evasion reference actions, together with the inferred critical interceptor $j^\star$, were combined into one aligned display. Their three distinct mappings and input arguments remain explicit.

No TTA-MAPPO core strategy formula was removed. The actor factorization, centralized critic prior, residual actor processing, trust factor, exploration intensity, latent Gaussian actor, tanh-bounded action, latent/executed policy distinction, mixed latent behavior policy, task-weighted Gaussian mixture, change-of-variables density, Jacobian correction, training sampling rule, PPO behavior-density ratio, clipped objective, critic loss, entropy term, total loss, and deterministic HIL evaluation action remain connected and reproducible.

## 6. Label and reference changes

Removed labels belonged only to displays that were converted to prose/table form or replaced by a consolidated formula:

- `eq:info-update-rate`
- `eq:act-ax`
- `eq:act-an`
- `eq:obs-self`
- `eq:obs-thr`
- `eq:obs-team`

Added labels:

- `eq:running-utility`
- `eq:act-map`

The only affected in-text equation references were removed with the obsolete information-rate explanation or redirected to `eq:act-map`. No dangling or duplicate label remains.

## 7. Compilation and static validation

Compilation command:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex
```

Result: successful; `latexmk` completed the required `pdflatex` passes and produced a 19-page PDF.

Automated checks report:

- protected source outside Sections 2--4: **byte-for-byte identical**;
- all pre-existing `figure`, `figure*`, `table`, `table*`, and `strip` environments: **byte-for-byte identical**;
- labels: **109**, with **0 duplicates**;
- unresolved `\ref`/`\eqref`: **0**;
- unresolved `\cite`: **0**;
- mismatched `\begin`/`\end`: **0**;
- undefined control sequences: **0**;
- LaTeX errors: **0**;
- required safety-margin, potential-game, behavior-policy, and PPO chain tokens missing: **0**.

Successful compilation also confirms balanced braces and valid equation/table syntax. Existing nonfatal overfull-box notices and the imported algorithm-framework PDF-version warning remain; no font, spacing, margin, image, float-position, or template change was made.

## 8. Manual logical-chain validation

1. **Safety chain: complete.** FOV geometry leads to the local margin state, finite horizon, state transition, $Z_{ij}$, radial gating, normalized multi-interceptor aggregation, and risk penalties without a missing intermediate definition.
2. **Dynamic-task chain: complete.** Relative state leads to capture inference, task utilities, entropy-regularized payoff, exact potential, Logit response, and online task reconfiguration. The proposition and optimization proof are intact.
3. **MARL chain: complete.** Group value defines $L(t)$ and $J_T$; $L(t)$ produces the discrete utility reward; task probabilities augment actor and critic inputs; trust controls task-guided latent exploration; tanh and the Jacobian define bounded executed densities; the behavior-density ratio feeds the PPO objective; deterministic evaluation uses the squashed actor mean.

No variable was found to be used before definition, and no formula input or method output was disconnected by the condensation.
