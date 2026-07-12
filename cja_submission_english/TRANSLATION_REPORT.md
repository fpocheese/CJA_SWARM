# Translation Report

## Completed Work

- Migrated the manuscript to the actual CJA class `cja.cls`.
- Translated the title, abstract, keywords, all active section headings, body text, figure captions, subcaptions, table captions, and table headers.
- Preserved formulas, labels, citations, numerical values, units, figure data, and reference entries.
- Preserved the full inline bibliography, including the unused `WeiYang2022CJA` entry, because reference deletion was prohibited.

## Translation Principles

- Faithful technical meaning was prioritized over stylistic rewriting.
- Chinese claims were not strengthened. Simulation results are described with terms such as "show", "indicate", and "suggest" rather than unsupported proof language.
- Existing methods, algorithms, experiments, figures, and conclusions were not expanded.
- Commented-out Chinese draft blocks were not translated as active body content because they are not part of the compiled source.

## Main Expression Adjustments

- Long Chinese sentences were split into formal English journal prose.
- Repeated "本文" constructions were reduced.
- FOV, HVT, CTDE, HIL, MAPPO, and RTA-MAPPO were standardized.
- Earlier "stealth" wording was revised to "concealment", "evasion", or "maneuvering penetration" according to context, because the Chinese source concerns突防/规避 rather than low-observable aircraft design.
- The Chinese figure filename was replaced in the English project by an ASCII copy to improve `pdflatex` portability.
- The title was revised to better match the Chinese source: "An Intelligent Cooperative Maneuvering Penetration Method with Dynamic Task Reconfiguration for Aircraft Swarms Against Cooperative Interception".
- PDF-numbered formulas (18), (73), (87), and (101) were split from side-by-side expressions into two-line aligned equations without changing mathematical meaning.
- PDF-numbered long formulas (51), (52), (55), (59), (63), (68), and (111) were reformatted with multi-line aligned equation layouts. An additional long reward-sum formula was also split to remove a large overfull box.
- Fig. 2, Table 1, and Table 2 were converted to nonfloating `strip` blocks so that they remain cross-column while staying close to their source positions. Fig. 3 remains a cross-column figure.
- Table 3 and Table 4 were reformatted without full-table scaling. Long headers and method labels were line-broken where needed, while all numerical results and method meanings were preserved.

## Items Requiring Author Confirmation

- Author names, affiliations, email, and corresponding-author information were absent from the Chinese source. Clear placeholders are used in the CJA front matter.
- The requested long equations have been reformatted, and no overfull box above 50 pt remains in the final log check. Final visual layout should still be checked before submission because the CJA format is compact.
- `WeiYang2022CJA` is present in the original bibliography but is not cited in the active Chinese body. It was preserved because reference deletion was prohibited.

## Scientific-Content Changes

- No new scientific method, model, experiment, figure, reference, claim, limitation, or future work was added.
- No formula, variable definition, numerical result, experimental parameter, or conclusion was intentionally changed.

## Independent Review Results

- Fidelity review: no blocking omission, added claim, numerical change, or formula change was found after the final terminology fixes.
- Academic English review: repeated Chinese-style constructions were reduced, evidence strength was kept conservative, and aerospace/control terminology was standardized.
- Technical consistency review: core terms, abbreviations, figure/table references, and cited labels were checked for consistency.
- LaTeX/template review: blocking layout issues in normal-column figures and dense tables were fixed; undefined references/citations were not found in the final log.
