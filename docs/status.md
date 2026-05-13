# Project Status - Pipeline B Nowcasting Benchmark

**State**: Empirical package complete and re-audited as of 2026-05-13.

Both the US and Turkey pipelines now have the full 17-model notebook set, model predictions for all m1/m2/m3 vintages, country-aware evaluation outputs, a root-level summary table, and reproducible figure outputs.

---

## Completed Artifacts

| Country | Notebooks | Prediction files | Evaluation output |
|---|---:|---:|---|
| US | 17/17 | 51/51 | `data/evaluation_results_us.csv` |
| Turkey | 17/17 | 51/51 | `turkey_data/evaluation_results_tr.csv` |

Root summary:

```text
evaluation_summary.md
```

Main evaluator:

```text
data/evaluate.py
```

Figure generator:

```text
data/generate_figures.py
figures/FIGURE_INDEX.md
```

The evaluator audits all prediction files before computing results. It handles Turkey's mixed filename convention non-destructively. Current repaired MIDAS/MIDAS-ML outputs use quarter-end target labels directly.

---

## Model List

The completed model set is:

```text
arma, ols, var, lasso, ridge, elasticnet, rf, xgboost, gb, dt,
mlp, lstm, deepvar, bvar, midas, midasml, dfm
```

Each model has `m1`, `m2`, and `m3` prediction files for both countries.

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
- **US BVAR 2025-Q4 m3** uses a Cat2 fallback forecast because the reduced BVAR covariance matrix is singular at that vintage.
- **Turkey BVAR** uses the locked Cat2 predictor set plus the GDP target for the same reason.
- **Turkey MIDAS-ML** uses fixed-penalty `sglfit` because rolling `cv.sglfit` did not complete.
- **Turkey DFM** uses the complete 22-variable Cat3 monthly set plus target. The 32 Tier-C short-history variables were tested but excluded because the full sparse panel fails inside `nowcastDFM`.
- **R vintages** now use explicit `gen_vintage_data(metadata, data, target_date, vintage_date)` masking. This keeps target-quarter rows for MIDAS/DFM while making m1/m2/m3 information dates explicit.
- **MLP m1/m2** remains a paper-facing diagnostic caveat: the outputs are valid and finite, but RMSFE is unusually large in both countries and should be discussed as instability rather than highlighted as a central result.

Raw prediction files are preserved. Turkey legacy filenames such as `arma_m1.csv` are accepted by the evaluator alongside newer names such as `bvar_tr_m1.csv`.

---

## Next Work

The empirical and figure-generation pipeline is complete. Remaining work is paper-writing:

1. Convert `evaluation_summary.md`, the two evaluation CSVs, and selected `figures/` outputs into the final report.
2. Write the methodology caveats for BVAR, Turkey DFM, Turkey MIDAS-ML, pseudo-real-time evaluation, and MLP instability.
3. Interpret US vs Turkey performance patterns across models, vintages, and panels.
