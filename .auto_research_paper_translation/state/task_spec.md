# Chinese-to-English CJA Manuscript Translation Task

## Objective

Translate the Chinese LaTeX manuscript in this project into a polished English journal manuscript using the Chinese Journal of Aeronautics (CJA) LaTeX template, without overwriting the original Chinese sources.

## Primary Source Candidates

- `newswarm_main.tex`
- `newswarm_no_first_order_lag_final_from_complete.tex`

The authoritative source must be selected only after checking document structure, dependencies, source completeness, and the relationship between the two files.

## Output Directory

All generated English submission files must be placed under:

- `cja_submission_english/`

The final project must be self-contained and must not rely on absolute paths.

## Non-Expansion Constraints

The task is faithful translation and academic English polishing. The following are disabled:

- Literature search and bibliography expansion.
- Automatic addition, deletion, or replacement of references.
- New experiments, ablations, figures, proofs, methods, algorithms, claims, or limitations.
- Scientific-content changes based on simulated review.

## Priority Order

1. Preserve the technical meaning of the Chinese source.
2. Preserve formulas, data, symbols, experimental settings, results, and conclusions.
3. Maintain terminology and notation consistency.
4. Improve English to formal aerospace-journal style.
5. Migrate correctly to the CJA template.
6. Ensure the LaTeX project compiles completely.

## Required Workflow

1. Analyze the Chinese LaTeX project and determine the authoritative manuscript source.
2. Analyze the CJA template and its compile chain.
3. Create and maintain a terminology glossary.
4. Translate section by section, with faithful translation followed by academic-English polishing.
5. Perform independent reviews for fidelity, English quality, technical consistency, and LaTeX/template correctness.
6. Compile the final manuscript and repair feasible errors or warnings.
7. Generate reports: `SOURCE_MAPPING.md`, `TERM_GLOSSARY.md`, `TRANSLATION_REPORT.md`, `COMPILE_REPORT.md`, and `FIGURE_LANGUAGE_ISSUES.md`.

## Success Criteria

- The authoritative Chinese source is identified and documented.
- All body text, title, abstract, keywords, section headings, captions, and tables are translated.
- Scientific content is neither added nor deleted.
- Terminology is consistent.
- The manuscript is migrated to the actual CJA template.
- Formulas, figures, labels, citations, and bibliography compile normally.
- A PDF is generated successfully.
- The required reports are present.

