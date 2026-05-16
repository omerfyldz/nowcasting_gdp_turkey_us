# Project Status - Pipeline B Nowcasting Benchmark

**State**: Empirical package complete and re-audited as of 2026-05-16.

Both the US and Turkey pipelines now have the full 17-model notebook set, country-aware evaluation outputs, a root-level summary table, and reproducible figure outputs. The core cross-country comparison remains `m1`/`m2`/`m3`; the US pipeline additionally includes a `post1` robustness vintage, and the Turkey pipeline includes `post1`/`post2` robustness vintages.

---

## Completed Artifacts

| Country | Notebooks | Prediction files | Evaluation output |
|---|---:|---:|---|
| US | 17/17 | 68/68 | `data/evaluation_results_us.csv` |
| Turkey | 17/17 | 85/85 | `turkey_data/evaluation_results_tr.csv` |

Root summary:

```text
evaluation_summary.md
```

Paper-facing results analysis:

```text
docs/results_analysis.md
```

Main evaluator:

```text
data/evaluate.py
```

Figure generator:

```text
data/generate_figures.py
data/generate_results_visuals.py
figures/FIGURE_INDEX.md
figures/RESULTS_FIGURE_INDEX.md
```

The evaluator audits all prediction files before computing results. It handles Turkey's mixed filename convention non-destructively. Current repaired MIDAS/MIDAS-ML outputs use quarter-end target labels directly.

---

## Model List

The completed model set is:

```text
arma, ols, var, lasso, ridge, elasticnet, rf, xgboost, gb, dt,
mlp, lstm, deepvar, bvar, midas, midasml, dfm
```

Each model has `m1`, `m2`, and `m3` prediction files for both countries. US models additionally have `post1` prediction files. Turkey models additionally have `post1` and `post2` prediction files.

---

## Evaluation Windows

US:

- Pre-COVID: 2017-2019
- COVID: 2020-Q2 to 2021-Q4
- Post-COVID: 2022-2025
- Full: 2017-2025

Turkey:

- Pre-crisis: 2018-2019
- COVID: 2020-Q2 to 2021-Q4
- Post-COVID: 2022-2025
- Full: 2018-2025

---

## Required Deviations to Document

These are intentional computational deviations from the original full Cat3 design:

- **US BVAR** uses the Lasso-80 reduced predictor set because full Cat3 BVAR is computationally infeasible in `mfbvar`.
- **US BVAR 2025-Q4 m3 and post1** use Cat2 fallback forecasts because the reduced BVAR covariance matrix is singular at those vintages.
- **Turkey BVAR** uses the locked Cat2 predictor set plus the GDP target for the same reason.
- **Turkey MIDAS-ML** uses documented fixed-penalty `sglfit(lambda = 0.01)` because rolling `cv.sglfit` aborted the Jupyter process during the post-release rerun.
- **Turkey DFM** uses a validation-selected Cat2 monthly predictor set plus target. Cat2 was selected on 2012-2017 validation RMSFE across `m1`/`m2`/`m3`, then retrained through 2017 for 2018-2025 testing. The 32 Tier-C short-history variables were tested but excluded because the full sparse panel fails inside `nowcastDFM`.
- **R vintages** now use explicit `gen_vintage_data(metadata, data, target_date, vintage_date)` masking. This keeps target-quarter rows for MIDAS/DFM while making m1/m2/m3 information dates explicit.
- **Turkey MLP m1/m2** remains a paper-facing diagnostic caveat: the outputs are valid and finite, but RMSFE is unusually large and should be discussed as instability rather than highlighted as a central result. US MLP was stabilized in the 2026-05-14 rerun and is now competitive in later vintages.

Raw prediction files are preserved. Turkey legacy filenames such as `arma_m1.csv` are accepted by the evaluator alongside newer names such as `bvar_tr_m1.csv`.

---

## Current US Robustness Layer

The US robustness layer is complete:

- `data/evaluation_results_us.csv`: 272 rows = 17 models x 4 vintages x 4 panels.
- `turkey_data/evaluation_results_tr.csv`: 340 rows = 17 models x 5 vintages x 4 panels.
- `data/evaluation_results_us_improved.csv`: 576 rows = 24 models/combinations x 4 vintages x 6 panels.
- `docs/us_model_diagnostics.md`: DFM, MLP, and combination diagnostics.
- `docs/bvar_us_fallback_log.csv`: BVAR fallback log.

---

## Next Work

The empirical and figure-generation pipeline is complete. Remaining work is paper-writing and final interpretation:

1. Convert `evaluation_summary.md`, `docs/results_analysis.md`, the two evaluation CSVs, and selected `figures/` outputs into the final report.
2. Write the methodology caveats for BVAR, Turkey DFM, Turkey MIDAS-ML, pseudo-real-time evaluation, and MLP instability.
3. Interpret US vs Turkey performance patterns across models, vintages, and panels.
4. Interpret the validation-selected Turkey DFM separately from the original Cat3 diagnostic, noting that the final DFM is not test-selected.
