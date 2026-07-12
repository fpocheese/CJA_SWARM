# Compile Report

## Environment

- OS context: Linux, project run under the provided workspace.
- Available tools used: `latexmk`, `pdflatex`, `pdfinfo`, `pdftotext`, `rg`.
- Template compile chain followed: `latexmk -pdf`, which drives repeated `pdflatex` runs for this inline-bibliography manuscript.

## Main File

- `cja_submission_english/newswarm_cja_english.tex`

## Actual Compile Command

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex
```

## Result

- Compile status: success.
- Final PDF: `cja_submission_english/newswarm_cja_english.pdf`
- PDF pages: 20.
- PDF version: 1.5.
- Fatal errors: 0 in final compile.
- Undefined references: 0 in final log check.
- Undefined citations: 0 in final log check.
- Missing figures: 0.
- Template example residue in PDF text: 0 occurrences found by text extraction check.
- Chinese CJK text residue in PDF text and English `.tex`: 0 occurrences found.
- Large overfull boxes above 50 pt: 0 in final log check after formula reformatting.

## Fixed Compile Issues

- Removed `amssymb` because CJA's `newtxmath` path already defined `\Bbbk`.
- Inserted `\begin{document}` before CJA front-matter commands, following the template structure.
- Replaced the Chinese figure filename reference with `figures/algorithm_framework.pdf`.
- Suppressed the CJA default placeholder running header with `\frontheader{}`.
- Replaced two-column figure widths based on `\textwidth` with `\linewidth` where needed to avoid clipping in normal `figure` environments.
- Resized the dense Monte Carlo and ablation tables to the column width.
- Revised the manuscript title in both the CJA title field and PDF metadata.
- Reformatted PDF-numbered equations (18), (51), (52), (55), (59), (63), (68), (73), (87), (101), and (111) using aligned multi-line equation layouts.
- Converted Fig. 2 to a nonfloating `strip` block to keep the cross-column figure close to its source location, while keeping Fig. 3 as a cross-column figure.
- Converted Table 1 and Table 2 to source-position `strip` blocks with `\captionof{table}` and full `\textwidth` scaling.
- Converted Fig. 4 from forced-position `[H]` placement to a cross-column `figure*` float so that both training subfigures are displayed completely after the Table 2 layout change.
- Reformatted Table 3 with `tabularx`, line-broken headers, simplified method names, and no full-table scaling so that the text is larger and more readable.
- Reformatted Table 4 with `tabularx`, line-broken headers, and no full-table scaling so that the ablation table remains readable at normal table size.

## Remaining Warnings

- Minor overfull boxes remain in normal CJA two-column text and compact mathematical expressions, but no overfull box above 50 pt remains in the final log check.
- `pdflatex` reports included PDF version warnings for some figures (`PDF 1.7` included in a PDF 1.5 workflow). The final PDF is generated successfully.

## Final Verification Commands

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error newswarm_cja_english.tex
rg -n "LaTeX Warning: (Reference|Citation)|Package natbib Warning: Citation|undefined references|undefined citations|There were undefined|Fatal error|LaTeX Error" newswarm_cja_english.log
pdftotext newswarm_cja_english.pdf - | rg -n "[一-龥]|Generation of dynamic grids|Transonic flow|Zhiliang|John SMITH|instructions|example-image|Chinese Journal of Aeronautics, \(year\)|volum\(number\)|stealth"
rg -n "[一-龥]" newswarm_cja_english.tex
pdfinfo newswarm_cja_english.pdf
```

The `rg` checks above returned no matches in the final pass.
