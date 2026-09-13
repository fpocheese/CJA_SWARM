# Logic Fix Report

## Files

- Modified manuscript: `newswarm_cja_english.tex`
- Pre-modification backup: `newswarm_cja_english_before_logic_fix.tex`
- Compilation output: `newswarm_cja_english.pdf`

## 1. State-transition derivation connection

Location: line 306, immediately between the two unchanged state-transition equations.

Before:

> and

After:

> the predicted horizon margin satisfies

The two adjacent equations were not modified.

## 2. Destruction indicator and destruction-time definitions

Location: lines 687 and 696.

Before:

> Let $\ell_i(T)\in\{0,1\}$ indicate preterminal destruction at

After:

> Let $\ell_i(T)\in\{0,1\}$ indicate whether UAV $i$ is destroyed before the terminal time, and define its destruction time as

The following definition of $t_i^{\mathrm{kill}}$ was not modified.

Before:

> The last valid pre-destruction state $t_i^{\mathrm{kill}-}$ is salvageable when detection risk is low, interceptor separation sufficient, and escape capability positive:

After:

> At the last valid pre-destruction instant $t_i^{\mathrm{kill}-}$, the UAV is considered salvageable when detection risk is low, interceptor separation is sufficient, and escape capability is positive:

No formula or subsequent text concerning $\chi_i^{\mathrm{salvage}}$, $U_i^{\mathrm{waste}}$, or $N_{\mathrm{waste}}$ was modified.

## 3. Opening sentence of the Markov decision model

Location: line 779.

Before:

> The preceding quantities define each offensive UAV's action, observation, reward, and policy network:

After:

> Based on the preceding theoretical quantities, each offensive UAV is modeled as a policy agent with policy

The policy equation and its following explanation were not modified.

## 4. Interpretation of the three task utilities

Location: line 592.

Before:

> Equations~\eqref{eq:role-utility-decoy}--\eqref{eq:role-utility-stealth} assign $D$ to attract interception at acceptable cost, $P$ to exploit favorable penetration and strike conditions, and $S$ to preserve concealment, evasion, and switching redundancy.

After:

> Equations~\eqref{eq:role-utility-decoy}--\eqref{eq:role-utility-stealth} associate task $D$ with attracting interception at acceptable cost, task $P$ with favorable penetration and strike conditions, and task $S$ with concealment, evasion, and switching redundancy.

The task-utility equations, task probabilities, Logit response, potential-game proposition, and proof were not modified.

## 5. Ablation-setting explanation before the table

Location: line 1780, immediately before Table~\ref{tab:ablation-rta-prior}.

Before:

> With trained weights fixed, selected actor inputs are masked only during inference. The first ablation removes $q_{ij}$ and $\widetilde\Gamma_{ij}$ from $o_i^{\mathrm{thr}}$; the second removes $\mu_i=[\mu_i^D,\mu_i^P,\mu_i^S]^{\top}$ while retaining all other observations. Each configuration uses $1000$ HIL Monte Carlo episodes; Table~\ref{tab:ablation-rta-prior} reports success.

After:

> With trained weights fixed, selected actor inputs are masked only during inference. The first ablation removes the FOV cone margin $q_{ij}$ and tracking-mismatch margin $\widetilde\Gamma_{ij}$ from $o_i^{\mathrm{thr}}$, thereby suppressing the actor's explicit access to threat geometry and interceptor tracking-mismatch information. The second removes the task-probability vector $\mu_i=[\mu_i^D,\mu_i^P,\mu_i^S]^{\top}$, thereby eliminating the dynamic task-reconfiguration prior from the actor input while retaining all other observations. Each configuration uses $1000$ HIL Monte Carlo episodes; Table~\ref{tab:ablation-rta-prior} reports the penetration success rates.

The ablation table and all text following it were not modified. The Monte Carlo experiment text was not modified.

## Verification

- A unified diff between the backup and revised source contains only the five requested modification items above (six replaced prose lines because item 2 contains two sentences).
- All displayed mathematical environments are byte-for-byte identical and occur in the same order.
- All figure and table environments are byte-for-byte identical and occur in the same order.
- All section titles and the ordered sequences of `\label`, `\ref`, `\eqref`, and `\cite` commands are identical.
- No duplicate label was found among the 113 labels.
- LaTeX environment begin/end counts match; the successful two-pass compilation also confirms balanced environments and braces.
- The compilation log contains no undefined control sequence, LaTeX error, undefined reference, undefined citation, duplicate-label diagnostic, runaway argument, or emergency stop.
- Mathematical formulas, parameters, experimental results, figures, tables, captions, paths, and layout settings were not modified.
- $N_{\mathrm{eff}}$, the Monte Carlo experiment, the ablation table, and the post-table ablation analysis were not modified.
- The abstract, Introduction, Conclusions, and references were not modified.
- No text outside the five specified modification items was changed.

## Compilation and page count

Command:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex
```

Result: successful; `latexmk` completed two `pdflatex` passes and produced `newswarm_cja_english.pdf`.

- Page count before modification: **20 pages**
- Page count after modification: **20 pages**

The existing nonfatal box warnings and the imported PDF-version warning remain; no layout or external asset was changed.
