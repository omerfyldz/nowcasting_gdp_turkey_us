# Comparing GDP Nowcasting Models in the United States and Turkey

This repository contains the empirical package for a GDP nowcasting benchmark across two contrasting macroeconomic environments:

- **United States**: a data-rich advanced economy.
- **Turkey**: a shorter-history, more volatile emerging-market economy.

The project evaluates 17 nowcasting models across three simulated information vintages (`m1`, `m2`, `m3`) and country-specific evaluation panels. The outputs are intended to support a paper on how nowcasting model performance changes with data availability, macroeconomic volatility, and country context.

## Main Finding

Model rankings are not portable across countries.

In the United States, flexible/high-dimensional approaches perform best, especially **LSTM** and **BVAR**. In Turkey, **MIDAS**, **VAR**, **LSTM**, and **BVAR** are strongest, suggesting that parsimonious and mixed-frequency econometric models remain highly competitive in a shorter and more volatile emerging-market sample.

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

for both countries.

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
├── images/                  README/appendix-style generated plots
├── docs/                    Audit/status documentation
└── archive/                 Preserved handoff notes, logs, diagnostics, original references
```

Nothing in `archive/` is required for the main evaluation run, but it is retained for provenance.

## Key Outputs

- `data/evaluation_results_us.csv`
- `turkey_data/evaluation_results_tr.csv`
- `evaluation_summary.md`
- `figures/FIGURE_INDEX.md`
- `figures/full_m3_rmsfe_rankings.png`
- `figures/full_m3_relative_rmsfe_us_tr.png`
- `figures/panel_relative_rmsfe_heatmaps.png`
- `figures/vintage_rmsfe_profiles.png`
- `figures/model_family_relative_rmsfe.png`

## Reproduce Final Evaluation

From the repository root:

```bash
python data/evaluate.py
python data/generate_figures.py
```

Expected evaluation output:

- United States: 204 rows = 17 models x 3 vintages x 4 panels
- Turkey: 204 rows = 17 models x 3 vintages x 4 panels
- 25 generated PNG figures

## Headline Results

Full-panel, `m3` vintage:

| Country | Best models |
|---|---|
| United States | LSTM, BVAR, MIDAS, MLP, MIDAS-ML |
| Turkey | MIDAS, VAR, LSTM, BVAR, Lasso |

Interpretation:

- The US results favor flexible/high-dimensional methods.
- The Turkey results are more balanced, with strong performance from mixed-frequency and parsimonious econometric models.
- Machine learning superiority is not universal across countries.

## Important Caveats

- The exercise is pseudo-real-time, not a true historical-vintage evaluation.
- GDP targets use final revised values.
- US BVAR uses a reduced Lasso-80 predictor set.
- US BVAR 2025-Q4 `m3` uses a documented Cat2 fallback because the reduced BVAR covariance matrix is singular at that vintage.
- Turkey BVAR uses locked Cat2 predictors plus the target.
- Turkey DFM uses complete Cat3 monthly predictors plus target; Tier-C variables were excluded after `nowcastDFM` failed on the sparse panel.
- Turkey MIDAS-ML uses fixed-penalty `sglfit`.
- Diebold-Mariano tests are included but should be interpreted cautiously because panel sample sizes are small.
- MLP `m1`/`m2` instability is an empirical caveat, not a headline result.

## Documentation

- `docs/audit_report.md`: end-to-end audit and literature alignment.
- `docs/status.md`: current project state and known deviations.
- `docs/repository_layout.md`: repository organization.
- `docs/github_push_checklist.md`: validation and push checklist.
- `data/evaluation_protocol.md`: evaluation panels and metrics.
- `data/target_specification.md`: target definitions.
- `figures/FIGURE_INDEX.md`: generated figure inventory.
- `archive/ARCHIVE_INDEX.md`: preserved non-core files.

## Original Benchmark Reference

This project was inspired by the structure and model lineup of Hopp's open-source nowcasting benchmark:

Hopp, D. (2022). *Benchmarking econometric and machine learning methodologies in nowcasting*. UNCTAD Research Paper No. 83.

Inherited README snapshots are preserved under:

```text
archive/original_reference/
```

The original `methodologies/` reference notebook folder is not included in this GitHub repository. A local copy was preserved outside the repo at:

```text
C:\Users\asus\Desktop\methodologies_nowcasting_original_reference
```
