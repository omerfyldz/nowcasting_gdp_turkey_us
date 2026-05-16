# Comparing GDP Nowcasting Models in the United States and Turkey

This repository contains the empirical package for a GDP nowcasting benchmark across two contrasting macroeconomic environments:

- **United States**: a data-rich advanced economy.
- **Turkey**: a shorter-history, more volatile emerging-market economy.

The project evaluates 17 nowcasting models across simulated information vintages and country-specific evaluation panels. The core cross-country comparison uses `m1`, `m2`, and `m3` for both countries; the US pipeline also includes `post1`, while the Turkey pipeline includes `post1` and `post2` robustness vintages. The outputs are intended to support a paper on how nowcasting model performance changes with data availability, macroeconomic volatility, and country context.

## Main Finding

Model rankings are not portable across countries.

In the United States, flexible/high-dimensional approaches perform best, especially **LSTM**, **MLP**, and **BVAR**. In Turkey, the strongest full-sample `m3` models are **Lasso**, **ElasticNet**, and **BVAR**, while `post1`/`post2` robustness horizons favor penalized linear models and BVAR. This suggests that model rankings are sensitive to country context, sample volatility, and vintage definition.

This is a **pseudo-real-time** exercise: information sets are simulated with publication-lag/ragged-edge masking, but the target GDP series uses final revised data rather than historical real-time GDP vintages.

## Model Set

The completed benchmark includes 17 models:

```text
arma, ols, var, lasso, ridge, elasticnet, rf, xgboost, gb, dt,
mlp, lstm, deepvar, bvar, midas, midasml, dfm
```

Each model has predictions for:

```text
m1, m2, m3
```

for both countries. The US notebooks additionally generate:

```text
post1
```

as a robustness horizon. The Turkey notebooks additionally generate:

```text
post1, post2
```

as Turkey robustness horizons.

## Repository Layout

```text
.
├── data/                    US data builders, helpers, evaluation script, policy docs
├── turkey_data/             Turkey data builders, helpers, metadata, final data
├── model_notebooks/         Active US model notebooks
├── turkey_model_notebooks/  Active Turkey model notebooks
├── predictions/             US prediction CSVs
├── turkey_predictions/      Turkey prediction CSVs
├── figures/                 Main paper-facing figures and figure index
├── docs/                    Audit/status documentation
└── archive/                 Preserved handoff notes, logs, diagnostics, original references
```

Nothing in `archive/` is required for the main evaluation run, but it is retained for provenance.

## Key Outputs

- `data/evaluation_results_us.csv`
- `turkey_data/evaluation_results_tr.csv`
- `evaluation_summary.md`
- `docs/results_analysis.md`
- `figures/FIGURE_INDEX.md`
- `figures/full_m3_rmsfe_rankings.png`
- `figures/full_m3_relative_rmsfe_us_tr.png`
- `figures/panel_relative_rmsfe_heatmaps.png`
- `figures/vintage_rmsfe_profiles.png`
- `figures/model_family_relative_rmsfe.png`
- `data/evaluation_results_us_improved.csv` and `docs/us_model_diagnostics.md`
  for the US-only robustness layer with forecast combinations and COVID-outlier
  diagnostics.

## Reproduce Final Evaluation

From the repository root:

```bash
python data/evaluate.py
python data/us_improvement.py
python data/generate_figures.py
python data/generate_results_visuals.py
```

Expected evaluation output:

- United States: 272 rows = 17 models x 4 vintages x 4 panels
- Turkey: 340 rows = 17 models x 5 vintages x 4 panels
- US improved robustness: 576 rows = 24 US models/combinations x 4 vintages x 6 panels
- 5 main paper PNG figures
- 7 results-analysis PNG figures
- 3 additional US-only robustness PNG figures

## Headline Results

Full-panel, `m3` vintage:

| Country | Best models |
|---|---|
| United States | LSTM, MLP, BVAR, Ridge, MIDAS |
| Turkey | Lasso, ElasticNet, BVAR, OLS, VAR |

Interpretation:

- The US results favor flexible/high-dimensional methods.
- The Turkey results are more balanced, with strong performance from penalized linear and reduced-dimensional econometric models.
- Machine learning superiority is not universal across countries.

## Important Caveats

- The exercise is pseudo-real-time, not a true historical-vintage evaluation.
- GDP targets use final revised values.
- US `post1` and Turkey `post1`/`post2` are robustness horizons, not replacements for the symmetric `m1`/`m2`/`m3` comparison.
- US BVAR uses a reduced Lasso-80 predictor set.
- US BVAR 2025-Q4 `m3` uses a documented Cat2 fallback because the reduced BVAR covariance matrix is singular at that vintage.
- Turkey BVAR uses locked Cat2 predictors plus the target.
- Turkey DFM uses a validation-selected Cat2 monthly predictor set plus target; Cat2 was selected on 2012-2017 validation RMSFE across `m1`/`m2`/`m3`, then retrained through 2017 for 2018-2025 testing. Tier-C variables were excluded after `nowcastDFM` failed on the sparse panel.
- Turkey MIDAS-ML uses documented fixed-penalty `sglfit(lambda = 0.01)` after `cv.sglfit` aborted the Jupyter process during the post-release rerun.
- Diebold-Mariano tests are included but should be interpreted cautiously because panel sample sizes are small.
- Turkey MLP `m1`/`m2` instability is an empirical caveat, not a headline result. US MLP is stabilized in the current rerun and is competitive in later vintages.

## Documentation

- `docs/audit_report.md`: end-to-end audit and literature alignment.
- `docs/results_analysis.md`: period-by-period, vintage-by-vintage interpretation
  of the final empirical results.
- `docs/literature_alignment_review.md`: project interpretation and deviations in light of the literature-review PDFs.
- `docs/status.md`: current project state and known deviations.
- `docs/repository_layout.md`: repository organization.
- `docs/github_push_checklist.md`: validation and push checklist.
- `data/evaluation_protocol.md`: evaluation panels and metrics.
- `docs/us_model_diagnostics.md`: US-only robustness diagnostics for combinations,
  DFM COVID sensitivity, and MLP instability.
- `docs/us_pipeline_preflight.md`: US rerun readiness checks and caveats.
- `docs/turkey_post_release_design.md`: implemented Turkey `post1`/`post2`
  robustness horizons and documented MIDAS-ML fallback.
- `data/target_specification.md`: target definitions.
- `figures/FIGURE_INDEX.md`: generated figure inventory.
- `figures/RESULTS_FIGURE_INDEX.md`: generated result-analysis figure inventory.
- `archive/ARCHIVE_INDEX.md`: preserved non-core files.

## Original Benchmark Reference

This project was inspired by the structure and model lineup of Hopp's open-source nowcasting benchmark:

Hopp, D. (2022). *Benchmarking econometric and machine learning methodologies in nowcasting*. UNCTAD Research Paper No. 83.
