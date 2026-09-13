# Second Condensation Report

## 1. Files and scope

- Modified source: `newswarm_cja_english.tex`
- Pre-modification backup: `newswarm_cja_english_before_second_condensation.tex`
- Compiled PDF: `newswarm_cja_english.pdf`
- Baseline page count: **21 pages**
- Final page count: **20 pages**

Only the following narrative text was condensed:

1. Section 2, *Cooperative UAV-Swarm Penetration Modeling under Multi-Interceptor FOV Constraints*.
2. Section 3, *Safety-Margin Evaluation and Dynamic Task Reconfiguration under Multi-Interceptor Detection Constraints*.
3. Section 4, *Cooperative UAV-Swarm Penetration Value Evaluation and Intelligent Guidance Policy Optimization*.
4. The Monte Carlo, baseline-comparison, ablation, and associated concluding prose after the three representative scenarios in Section 5.

No layout-based page-reduction measure was used.

## 2. Narrative-text compression

The counts below were obtained with the same automated extraction procedure for the backup and revised files. Displayed mathematics, figures, tables, captions, headings, and LaTeX commands were excluded before counting English words.

| Allowed part | Before | After | Reduction |
|---|---:|---:|---:|
| Section 2 | 534 words | 386 words | 27.72% |
| Section 3 | 1,362 words | 1,095 words | 19.60% |
| Section 4 | 1,949 words | 1,465 words | 24.83% |
| Monte Carlo and baseline-comparison prose | 189 words | 76 words | 59.79% |
| Ablation prose | 229 words | 94 words | 58.95% |
| Post-experiment integrated summary | 89 words | 32 words | 64.04% |

The capped regions remain within the specified maxima: Section 2 below 28%, Section 3 below 20%, Section 4 below 25%, and the Monte Carlo and ablation prose below 60%.

## 3. Condensation performed

### Section 2

- Consolidated repeated descriptions of the coordinate system, vehicle states, relative geometry, FOV variables, detection logic, target selection, and PN guidance.
- Retained every first-use variable definition, capability constraint, defender target-assignment rule, information-update condition, and detection/destruction criterion.

### Section 3

- Removed redundant transitions and repeated interpretations around the safety-margin chain while preserving the full progression from the FOV state to the joint detection-risk penalty.
- Condensed repeated explanations of distance gating, normalized smooth-max aggregation, capture-likelihood inference, and the three task utilities.
- Preserved the Logit response, individual payoff, potential function, proposition, unilateral-deviation argument, exact-potential equality, and Lagrange-multiplier derivation.

### Section 4

- Condensed repeated explanations accompanying the group-value metrics, Dec-POMDP observations, reward construction, and TTA-MAPPO architecture.
- Preserved all definitions required for ineffective-sacrifice evaluation, observation dimensions, Top-K threat ordering, CTDE, physical action mapping, reward composition, latent/executed action densities, Jacobian correction, PPO update, and deterministic evaluation.
- Shortened the descriptions of the three heuristic guidance mappings without changing their definitions or algorithmic roles.

### Monte Carlo, comparison, and ablation analysis

- Reduced the Monte Carlo discussion to three compact paragraphs retaining the three engagement scales, 1,000 HIL episodes per scale, randomized perturbations, both evaluation metrics, proposed-method results, baseline method names, citations, and comparison conclusion.
- Reduced the ablation setup and interpretation while retaining fixed trained weights, inference-only masking, both masked input groups, unchanged remaining observations, 1,000 HIL episodes, and the conclusions supported by the unchanged table values.
- Condensed the integrated experimental summary to two sentences while retaining stable cooperation, decoy-containment effects, protection of the primary-attack channel, online task formation/switching, and scale adaptability.

## 4. Invariance and static checks

Automated comparisons between the backup and revised source confirmed:

- Text outside the authorized ranges is byte-for-byte unchanged.
- The abstract, keywords, Introduction, Section 5.1, all three representative 6-vs-6/8-vs-8/10-vs-10 scenario analyses, Conclusions, and references are unchanged.
- All **137** displayed mathematical environments are identical and remain in the same order.
- All **20** figure/table environments are identical and remain in the same order.
- Mathematical formulas and derivations are unchanged.
- Figure and table contents, captions, labels, paths, values, and formatting are unchanged.
- The merged parameter table is unchanged.
- All parameters, training settings, experimental data, success rates, miss distances, and conclusions are unchanged.
- The ordered sets of `\label`, `\ref`, `\eqref`, and `\cite` commands are unchanged.
- All section, subsection, and subsubsection titles are unchanged.

## 5. Compilation result

Final compilation command:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/cja_second_condensation.bu2XBW/final newswarm_cja_english.tex
```

Compilation completed successfully. `pdfinfo` reports **20 pages**. The final log contains no fatal LaTeX error, undefined control sequence, undefined reference/citation, or duplicate-label diagnostic. Nonfatal overfull-box and imported-PDF-version warnings remain and were not addressed because the task prohibited layout, figure, and template changes.

## 6. Page-target assessment

The manuscript was reduced from 21 to 20 pages. Reaching 18 pages was not possible without exceeding the authorized compression ceilings or deleting definitions and explanatory links needed for the Section 3 derivation and Section 4 algorithmic closure. Section 2, Section 3, Section 4, and the Monte Carlo/ablation prose are already close to their respective maximum permitted reductions; formulas, figures, tables, captions, representative scenario analyses, Section 5.1, Conclusions, and all other protected material could not be altered. The revision therefore stops at 20 pages in accordance with the instruction to prioritize mathematical and technical completeness over forced page reduction.
