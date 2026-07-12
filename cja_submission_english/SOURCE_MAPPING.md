# Source Mapping

## Authoritative Chinese Source

- Actual compile entry: `newswarm_main.tex`.
- Manuscript body: `newswarm_no_first_order_lag_final_from_complete.tex`.
- Relationship: `newswarm_main.tex` is a 26-line `ctexart` wrapper and inputs `newswarm_no_first_order_lag_final_from_complete.tex`; the long file declares `% !TEX root = newswarm_main.tex` and is not a standalone document.
- Authority decision: the complete Chinese manuscript is the wrapper plus the body file. The English CJA manuscript uses the active, uncommented body content from `newswarm_no_first_order_lag_final_from_complete.tex`.

## Source Structure

- Abstract and keywords: `newswarm_no_first_order_lag_final_from_complete.tex`, beginning of file.
- Introduction: `\section{引言}`.
- Problem modeling: `\section{考虑多拦截器视场限制的集群突防问题建模}`.
- Safety margin and task reconfiguration: `\section{多拦截器探测约束下的安全裕度评估与任务动态重构机制}`.
- Value evaluation and policy solution: `\section{集群协同突防价值评估与智能制导策略求解}`.
- Simulation results and analysis: `\section{仿真结果与分析}`.
- Conclusion: `\section{结论}`.
- References: inline `thebibliography` block in the body file.

## Output Mapping

- English CJA main file: `cja_submission_english/newswarm_cja_english.tex`.
- Final PDF: `cja_submission_english/newswarm_cja_english.pdf`.
- Figures: copied from `figures/` to `cja_submission_english/figures/`.
- CJA template files: copied from `cja-template/` to `cja_submission_english/`.

## Version Conflicts

- No evidence was found that the two `.tex` files are competing complete versions. The short file is the actual root; the long file is the manuscript body.
- Commented-out old derivations and old reward-design blocks in the body were not treated as active manuscript content.
- The inline bibliography contains one unused entry, `WeiYang2022CJA`; it was preserved because the task prohibits deleting references.

