# Goal Revision Report

## 1. Deliverables

- Chinese source (unchanged): `newswarm_no_first_order_lag_final_from_complete.tex`
- Revised English manuscript: `cja_submission_english/newswarm_cja_english.tex`
- Pre-revision backup: `cja_submission_english/newswarm_cja_english_before_goal_revision.tex`
- Final PDF: `cja_submission_english/newswarm_cja_english.pdf`
- Final terminology glossary: `TERM_GLOSSARY_FINAL.md`

The English manuscript was located in the `cja_submission_english` subdirectory rather than at the project root. All edits and compilation were therefore performed in that CJA submission directory.

## 2. Chinese--English Fidelity Review

The Chinese and English manuscripts were mapped and reviewed by section, subsection, paragraph, equation explanation, figure/table caption, and result-analysis block. The review covered the title, abstract, keywords, introduction, engagement model, UAV dynamics, interceptor detection and assignment model, safety-margin model, dynamic task reconfiguration, utility functional, Markov decision model, reward design, RTA-MAPPO, HIL platform and parameters, 6VS6/8VS8/10VS10 cases, Monte Carlo study, ablation study, conclusions, captions, labels, and citations.

The Chinese source remained unchanged. Equation meanings, variable definitions, constraints, labels, citations, figure paths, and bibliography entries were preserved. The citation-key sets in the Chinese and English manuscripts were compared and found to be consistent.

## 3. Fidelity Corrections

- Restored simulation mechanisms that had been compressed into short numerical summaries.
- Corrected the English parameter table to match the Chinese source: `lambda_P=0`, `lambda_C=0`, and `lambda_S=0`.
- Removed wording that implied stronger generality or certainty than the Chinese manuscript.
- Reframed `capture` according to physical meaning: target acquisition, engagement likelihood, active interceptor assignment, or engagement by an interceptor.
- Preserved the unique algorithm expansion `role-trust attention multi-agent proximal policy optimization (RTA-MAPPO)` and used `RTA-MAPPO` thereafter.

## 4. Simulation-Analysis Revisions

### 6VS6

- Preserved the impact time, terminal miss distance, survivor count, acceleration peaks, minimum interceptor separation, and allocation-failure statistics.
- Restored the sequence of interceptor target switches involving `D_0`, `D_1`, `D_4`, and `D_5`.
- Explained how early decoy losses dispersed defensive resources and created spatial advantage for `A_4`.
- Restored the key mechanism that the allocation-failure ratio later returned to approximately 1.00, but this recovery occurred after the decisive penetration window had already formed.
- Connected stagewise task-role probabilities to decoy relay, concealment, and primary-attack persistence under nonzero threat.

### 8VS8

- Preserved the impact time, miss distance, survivor count, minimum separation, and allocation-failure statistics.
- Restored the complete target-switching chain and the diversion of interceptors toward non-primary UAVs.
- Explained why the brief early engagement of `A_4` did not become an effective interception.
- Restored the interpretation of the post-decoy increase in the allocation-failure ratio: reassignment did not recover primary-channel coverage and instead produced persistent defensive-resource misallocation.
- Connected relay containment to the increase in `A_4`'s primary-attack probability and its low direct-engagement fraction.

### 10VS10

- Preserved the impact time, miss distance, survivor count, acceleration limits, minimum separation, stagewise allocation-failure ratios, and task-role probabilities.
- Restored the sequential target switches by `D_7`, `D_8`, and `D_9` and the containment of the remaining interceptors.
- Explained why later engagement by as many as two interceptors could not recover an effective interception geometry.
- Restored the implication of the monotonically increasing allocation-failure ratio: repeated reassignment progressively weakened, rather than restored, defensive coordination.
- Connected the high aggregate threat and engagement fraction of `A_8` to the maintained primary-attack role, decoy relay, concealment dispersion, and large spatial margin.

### Monte Carlo and Ablation Studies

- Retained all Chinese-source success rates and terminal miss-distance data.
- Kept comparison claims restricted to the same defensive strategy and random initial perturbations stated in the source.
- Clarified the two test-time observation-channel ablations without implying retraining.
- Preserved the conclusion that the threat safety-margin and group-benefit channels are important to risk identification, decoy containment, and primary-corridor maintenance.

## 5. Terminology Unification

- Physical study objects are `UAV`/`UAVs`; `agent` is retained only for reinforcement-learning entities.
- `aircraft swarm`, `offensive aircraft`, and related study-object expressions were replaced by UAV terminology in the manuscript body and captions.
- The unchanged internal label `sec:aircraft-model` is retained solely to preserve cross-reference stability.
- `overload` terminology in the body and captions was replaced by acceleration terminology based on direction and physical meaning.
- Figure filenames containing `overload` were not renamed because they are file paths, not manuscript terminology.
- `capture` was disambiguated as `target acquisition`, `engagement likelihood`, `interceptor--target assignment`, `engage`, or `track` according to context.

## 6. Title, Abstract, and Conclusions

- The title now emphasizes UAV swarms, dynamic task reconfiguration, cooperative maneuvering penetration, and multi-interceptor cooperative interception.
- The abstract was rewritten to follow the Chinese sequence of background, limitation, model, dynamic task mechanism, trust-exploration optimization, and simulation conclusion.
- The conclusions were rewritten as four restrained findings supported by the model and simulations; unsupported claims about universal performance, robustness, real-time capability, or generalization were not added.

## 7. Independent Review Passes

Four role-separated review passes were performed:

1. Fidelity audit: Chinese--English section and paragraph mapping; numerical, label, and citation checks.
2. Aerospace terminology audit: UAV dynamics, FOV, proportional navigation, acceleration commands, engagement, and cooperative-penetration terminology.
3. Simulation audit: independent checks of 6VS6, 8VS8, 10VS10, Monte Carlo, and ablation mechanisms and numerical claims.
4. LaTeX audit: clean compilation, references, citations, labels, figures, PDF generation, and warning review.

No subagent execution interface was available in the current environment, so the independent roles were implemented as separate review passes with distinct checklists.

## 8. Compilation

Working directory:

```text
cja_submission_english
```

Final compile command:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex
```

Clean rebuild command used before final compilation:

```bash
latexmk -C newswarm_cja_english.tex
```

Compilation status: **successful**.

Final output: 20 pages, approximately 3.16 MB.

Resolved during compilation:

- Duplicate labels and duplicate PDF destinations caused by a `cuted/strip` full-width algorithm figure. The figure was converted to the CJA-compatible `figure*` environment without changing its content or label.
- Long inline noise-distribution and action-bias expressions were reformatted as multiline mathematics.
- In the follow-up layout correction, the Section 5.1 noise relations were returned to a continuous inline paragraph as requested, and the two consecutive `cuted/strip` tables were converted to standard `table*` floats. A subsequent float-layout pass placed the HIL figure in the Section 5.1 text on page 11, grouped the two wide tables with the training curves on page 12, and reduced Fig. 5 to a compact single-column trajectory plot. This removed the former overlap, float-only stacking, and excessive blank space around Fig. 5.

Final log status:

- Fatal errors: none.
- Undefined references: none.
- Undefined citations: none.
- Multiply defined labels: none.
- Missing figures: none.
- Chinese text in the English body: none detected.

Remaining nonfatal warnings:

- Several overfull boxes remain in long mathematical expressions and dense multi-panel figure layouts; the largest reported excess is approximately `21.69 pt`. These do not obscure content or prevent compilation.
- `gao1_V2.pdf` and `algorithm_framework.pdf` are PDF 1.7 assets included by a PDF 1.5 `pdflatex` engine. They render successfully, but source-asset down-conversion may be considered before final publisher production.

## 9. Items Requiring Author Confirmation

1. Author names, affiliations, city, country, and corresponding-author email are placeholders because the Chinese source does not provide them.
2. The Chinese method text states positive weighting coefficients, whereas its simulation table sets `lambda_P`, `lambda_C`, and `lambda_S` to zero. The English table now follows the Chinese numerical table exactly, but the intended theoretical/experimental relationship should be confirmed.
3. The Chinese source identifies the embedded hardware as `NVIDIA Jetson NX`; the exact commercial model (for example, Xavier NX or Orin NX) should be confirmed.
4. The implementation details used to reproduce the two comparison methods in the Monte Carlo study are not fully specified in the Chinese source. Additional reproducibility details may be needed during peer review, but none were invented here.
5. The remaining overfull mathematical and multi-panel layout warnings may require final typesetting adjustments after author and publisher metadata are inserted.
