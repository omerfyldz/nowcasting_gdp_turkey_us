# Comparing GDP Nowcasting Models in the United States and Turkey

This repository contains the empirical package for a GDP nowcasting benchmark across two contrasting macroeconomic environments:

- **United States**: a data-rich advanced economy.
- **Turkey**: a shorter-history, more volatile emerging-market economy.

The project evaluates 17 nowcasting models across simulated information vintages and country-specific evaluation panels. The core cross-country comparison uses `m1`, `m2`, and `m3` for both countries. The US pipeline additionally includes `post1`; the Turkey pipeline additionally includes `post1` and `post2` as robustness horizons.

The empirical design is **pseudo-real-time**: information sets are simulated through publication-lag and ragged-edge masking, but GDP targets use final revised data rather than historical real-time GDP vintages.

## Main Finding

Model rankings are not portable across countries.

In the United States, flexible and high-dimensional approaches perform best, especially **LSTM**, **MLP**, and **BVAR**. In Turkey, the strongest full-sample `m3` models are **Lasso**, **ElasticNet**, and **BVAR**. The results suggest that data richness, sample volatility, and vintage definition materially affect which nowcasting models perform best.

## Model Set

The completed benchmark includes 17 models:

```text
arma, ols, var, lasso, ridge, elasticnet, rf, xgboost, gb, dt,
mlp, lstm, deepvar, bvar, midas, midasml, dfm
```

Prediction vintages:

| Country | Core vintages | Robustness vintages |
|---|---|---|
| United States | `m1`, `m2`, `m3` | `post1` |
| Turkey | `m1`, `m2`, `m3` | `post1`, `post2` |

## Repository Layout

```text
.
|-- data/                    US data builders, helpers, evaluation scripts, policy docs
|-- turkey_data/             Turkey data builders, helpers, metadata, final data
|-- model_notebooks/         Active US model notebooks
|-- turkey_model_notebooks/  Active Turkey model notebooks
|-- predictions/             Final US prediction CSVs
|-- turkey_predictions/      Final Turkey prediction CSVs
|-- figures/                 Paper-facing generated figures
|-- docs/                    Audit, status, results, and methodology documentation
`-- archive/                 Preserved logs, diagnostics, handoff notes, and non-core material
```

Nothing in `archive/` is required for the main evaluation run. It is retained only for provenance and debugging history.

## Key Outputs

- `data/evaluation_results_us.csv`
- `turkey_data/evaluation_results_tr.csv`
- `data/evaluation_results_us_improved.csv`
- `evaluation_summary.md`
- `docs/results_analysis.md`
- `docs/us_model_diagnostics.md`
- `figures/FIGURE_INDEX.md`
- `figures/RESULTS_FIGURE_INDEX.md`

Main paper-facing figures include:

- `figures/full_m3_rmsfe_rankings.png`
- `figures/full_m3_relative_rmsfe_us_tr.png`
- `figures/panel_relative_rmsfe_heatmaps.png`
- `figures/vintage_rmsfe_profiles.png`
- `figures/model_family_relative_rmsfe.png`
- `figures/results_period_rankings_m3.png`
- `figures/results_post_release_rankings.png`
- `figures/results_covid_sensitivity_m3.png`

## Reproduce Final Evaluation

From the repository root:

```bash
python data/evaluate.py
python data/us_improvement.py
python data/generate_figures.py
python data/generate_results_visuals.py
```

Expected outputs:

| Output | Expected size |
|---|---:|
| `data/evaluation_results_us.csv` | 272 rows = 17 models x 4 vintages x 4 panels |
| `turkey_data/evaluation_results_tr.csv` | 340 rows = 17 models x 5 vintages x 4 panels |
| `data/evaluation_results_us_improved.csv` | 576 rows = 24 US models/combinations x 4 vintages x 6 panels |
| Main paper figures | 5 PNG files |
| Results-analysis figures | 7 PNG files |
| US robustness figures | 3 PNG files |

## Headline Results

Full-panel, `m3` vintage:

| Country | Best models |
|---|---|
| United States | LSTM, MLP, BVAR, Ridge, MIDAS |
| Turkey | Lasso, ElasticNet, BVAR, OLS, VAR |

Interpretation:

- The US results favor flexible and high-dimensional methods.
- The Turkey results favor shrinkage, reduced-dimensional econometric models, and parsimonious specifications.
- Machine-learning superiority is not universal across countries.
- COVID-period results should be interpreted separately because they strongly affect full-sample rankings.

## Important Caveats

- This is a pseudo-real-time exercise, not a true historical-vintage real-time evaluation.
- GDP targets use final revised values.
- `m1`, `m2`, and `m3` are the symmetric cross-country comparison vintages.
- US `post1` and Turkey `post1`/`post2` are robustness horizons, not replacements for the core `m1`/`m2`/`m3` comparison.
- US BVAR uses a reduced Lasso-80 predictor set.
- US BVAR 2025-Q4 `m3` and `post1` use documented Cat2 fallback forecasts because the reduced BVAR covariance matrix is singular at those vintages.
- Turkey BVAR uses locked Cat2 predictors plus the GDP target.
- Turkey DFM uses a validation-selected Cat2 monthly predictor set plus target. Cat2 was selected on 2012-2017 validation RMSFE across `m1`/`m2`/`m3`, then retrained through 2017 for 2018-2025 testing. Tier-C variables were excluded after `nowcastDFM` failed on the sparse panel.
- Turkey MIDAS-ML uses documented fixed-penalty `sglfit(lambda = 0.01)` after `cv.sglfit` aborted the Jupyter process during the post-release rerun.
- Diebold-Mariano tests are included but should be interpreted cautiously because panel sample sizes are small.
- Turkey MLP `m1`/`m2` instability is an empirical caveat, not a headline result. US MLP is stabilized in the current rerun and is competitive in later vintages.

## Documentation Map

| File | Purpose |
|---|---|
| `docs/audit_report.md` | End-to-end audit and implementation review |
| `docs/results_analysis.md` | Period-by-period and vintage-by-vintage interpretation |
| `docs/leakage_audit.md` | Leakage audit, pseudo-real-time limitations, and CV preprocessing notes |
| `docs/literature_alignment_review.md` | Literature alignment and deviations |
| `docs/status.md` | Current project state and known deviations |
| `docs/repository_layout.md` | Repository organization |
| `data/evaluation_protocol.md` | Evaluation panels, metrics, and filename conventions |
| `data/target_specification.md` | Target definitions and interpretation |
| `docs/us_model_diagnostics.md` | US combinations, DFM, MLP, and COVID robustness diagnostics |
| `docs/turkey_post_release_design.md` | Turkey `post1`/`post2` design and MIDAS-ML fallback |
| `figures/FIGURE_INDEX.md` | Main figure inventory |
| `figures/RESULTS_FIGURE_INDEX.md` | Results-analysis figure inventory |
