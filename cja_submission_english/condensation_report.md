# Condensation Report

## Files and scope

- Revised source: `newswarm_cja_english.tex`
- Pre-condensation backup: `newswarm_cja_english_before_condensation.tex`
- Compiled PDF: `newswarm_cja_english.pdf`
- Permitted source ranges: complete Sections 2--4 and Section 5.1 only.
- All text before Section 2, the Section 5 heading outside Section 5.1, and all content from Section 5.2 to the end of the manuscript were checked byte-for-byte against the backup and are unchanged.

## Page count and narrative-word reduction

Narrative words were counted with one fixed procedure for both files: displayed mathematics, figures, tables, captions, headings, and LaTeX commands were removed before `detex` word counting.

| Part | Before | After | Reduction |
|---|---:|---:|---:|
| Section 2: Cooperative UAV-Swarm Penetration Modeling under Multi-Interceptor FOV Constraints | 736 | 534 | 27.45% |
| Section 3: Safety-Margin Evaluation and Dynamic Task Reconfiguration under Multi-Interceptor Detection Constraints | 1814 | 1362 | 24.92% |
| Section 4: Cooperative UAV-Swarm Penetration Value Evaluation and Intelligent Guidance Policy Optimization | 2535 | 1949 | 23.12% |
| Section 5.1: Parameter Settings and Model Training | 736 | 524 | 28.80% |

- PDF before condensation: **22 pages**.
- PDF after condensation: **21 pages**.

The result does not reach 18 pages because the prescribed narrative-reduction ceilings have been reached closely, especially in Sections 3 and 5.1, while the unchanged equations, figures, and tables occupy a substantial fixed area. Further reduction would require exceeding the safe limits, deleting technical explanations, changing other sections, or altering layout, all of which were explicitly prohibited.

## Condensation by location

### Section 2

- Combined the engagement sets, HVT objective, maneuver-capability asymmetry, and geometry reference into a compact opening paragraph.
- Consolidated coordinate, state, HVT-relative-motion, FOV, sensing-rate, target-selection, PN-guidance, and hit/destruction explanations.
- Retained every first-use definition and the physical distinctions among FOV margin, occupancy, detection, target assignment, LOS motion, and PN control.

### Section 3

- Removed repeated introductions and post-formula paraphrases from the zero-control prediction-horizon derivation while retaining the complete derivation chain and physical meaning of the finite horizon.
- Condensed radial gating, normalized smooth-max aggregation, risk metrics, inferred capture likelihood, and task-utility explanations.
- Preserved the Logit response, individual payoff, potential function, proposition, unilateral-deviation argument, and Lagrange-multiplier derivation in full mathematical form.

### Section 4

- Condensed the chapter introduction and repeated interpretations of effective penetration, pre-destruction salvageability, group loss, terminal value, and the utility functional.
- Shortened repeated explanations of observation groups, reward components, and action mappings while retaining all definitions, dimensions, Top-K ordering, zero padding, reward logic, and CTDE statements.
- Preserved all TTA-MAPPO formulas and the complete latent-action/executed-action, behavior-policy, Jacobian-correction, PPO, entropy, and deterministic-evaluation logic.

### Section 5.1

- Combined the server, five Jetson NX nodes, TCP communication, distributed HIL loop, and deterministic evaluation description.
- Concentrated the training-hyperparameter explanation without repeating table values.
- Preserved every initial-state distribution, uncertainty model, scale case, hardware allocation, and training comparison.
- Reduced the reward and critic-loss discussions to one compact paragraph each and compressed the offline allocation-failure diagnostic explanation.

## Invariance and consistency checks

The following comparisons against `newswarm_cja_english_before_condensation.tex` passed:

- **Mathematical formulas:** 137 displayed equation/align environments are identical in content and order.
- **Mathematical derivations:** no displayed derivation step was deleted, inserted, merged, split, or edited.
- **Figures and tables:** 20 figure/table environments, including captions, paths, dimensions, and all table cells, are identical.
- **Cross-references and citations:** the complete ordered sequence of every `\label`, `\ref`, `\eqref`, and `\cite` command is identical.
- **Section structure:** all section, subsection, and subsubsection titles and their order are identical.
- **Parameters and experimental data:** parameter tables, initial states, noise values, training hyperparameters, success rates, miss distances, task probabilities, curve data, ablation values, and comparison values are unchanged.
- **Protected manuscript content:** title, abstract, keywords, Introduction, Section 5.2 onward, Conclusions, references, author information, affiliations, and preamble are byte-for-byte unchanged.
- No font, margin, line spacing, column spacing, equation spacing, float size, template option, or other layout setting was changed.

## Compilation check

- Command: `latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex` (compiled in an isolated output directory and copied to the project as `newswarm_cja_english.pdf`).
- Result: **successful**; final PDF has 21 pages.
- Final log contains no undefined control sequence, LaTeX error, undefined reference, undefined citation, multiply-defined label, or duplicate-label diagnostic.
- The template still reports non-fatal overfull-box warnings associated mainly with long mathematical content and unchanged later tables/text. Their total decreased from 18 in the baseline build to 16 in the revised build.
