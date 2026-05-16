# Evaluation Protocol (Pipeline B)

## Loss Functions

For every model, country, target quarter, and vintage, the evaluator records:

```text
e_t  = pred_t - y_t
SE_t = e_t^2
AE_t = abs(e_t)
```

where `y_t` is the final revised quarterly log GDP change. Losses are not annualised before averaging.

Reported statistics:

- **RMSFE** = `sqrt(mean(SE_t))`
- **MAE** = `mean(AE_t)`
- **Relative RMSFE vs ARMA** = `RMSFE_model / RMSFE_ARMA`
- **DM vs ARMA** = simple Diebold-Mariano comparison against ARMA using squared-error loss

## Panels

US panels:

| Panel | Range | Quarters |
|---|---|---:|
| Pre-COVID | 2017-Q1 .. 2019-Q4 | 12 |
| COVID | 2020-Q2 .. 2021-Q4 | 7 |
| Post-COVID | 2022-Q1 .. 2025-Q4 | 16 |
| Full | 2017-Q1 .. 2025-Q4 | 36 |

Turkey panels:

| Panel | Range | Quarters |
|---|---|---:|
| Pre-crisis | 2018-Q1 .. 2019-Q4 | 8 |
| COVID | 2020-Q2 .. 2021-Q4 | 7 |
| Post-COVID | 2022-Q1 .. 2025-Q4 | 16 |
| Full | 2018-Q1 .. 2025-Q4 | 32 |

The current test horizon ends at 2025-Q4. Extending to 2026-Q1 requires regenerating target data, predictions, and evaluation outputs.

## Benchmark

The empirical package uses **ARMA** as the univariate benchmark. ARMA is refit in the rolling/vintage loop using the target history available at the information date. All relative RMSFE and Diebold-Mariano results are computed against ARMA for the same country, panel, and vintage.

## Vintage Construction

The R notebooks use `gen_vintage_data(metadata, data, target_date, vintage_date)`.

This keeps rows through the target quarter so MIDAS/DFM can produce a current-quarter nowcast, while masking each variable according to what would have been available at the vintage month:

- m1: `vintage_date = Q-end - 2 months`
- m2: `vintage_date = Q-end - 1 month`
- m3: `vintage_date = Q-end`
- post1: `vintage_date = Q-end + 1 month`
- post2, Turkey robustness only: `vintage_date = Q-end + 2 months`

Python notebooks use the same target-date/vintage-date separation. Flattened
supervised models call `make_supervised_vintage_frame(...)`, which always keeps
the test row at the GDP target quarter while using the vintage date only for
publication-lag masking. Sequence/VAR-style notebooks use `gen_vintage_data(...)`
directly for the same reason.

The symmetric cross-country comparison remains `m1`/`m2`/`m3`. US `post1` and
Turkey `post1`/`post2` are robustness horizons and should be discussed
separately from the symmetric benchmark.

## Model-Set Caveats

- US BVAR uses a Lasso-80 reduced predictor set, not full Cat3.
- Turkey BVAR uses locked Cat2 predictors plus the GDP target.
- US DFM uses monthly Cat3 plus target; quarterly Cat3 variables and COVID dummies are excluded to keep `nowcastDFM` feasible.
- Turkey DFM uses a validation-selected Cat2 monthly predictor set plus target. Candidate panels were compared on 2012-2017 validation RMSFE across `m1`/`m2`/`m3`; Cat2 was selected and then retrained through 2017 before 2018-2025 test evaluation. Tier-C short-history variables were tested but excluded because the full sparse panel fails inside `nowcastDFM`.
- Turkey MIDAS-ML uses documented fixed-penalty `sglfit(lambda = 0.01)` because rolling `cv.sglfit` aborted the Jupyter process during the post-release rerun.
- Turkey MLP m1/m2 outputs are finite and evaluated, but their very large RMSFE should be treated as model instability in the paper. US MLP was stabilized in the current rerun and should be interpreted from the updated evaluation tables.

## Evaluator Rules

`data/evaluate.py` fails if any required prediction file is missing, has the wrong row count, has missing required columns, has date-range mismatches, or contains missing/non-finite actuals or predictions.

Raw predictions are preserved on disk:

- US: `predictions/<model>_<m1|m2|m3|post1>.csv`
- Turkey preferred: `turkey_predictions/<model>_tr_<m1|m2|m3|post1|post2>.csv`
- Turkey legacy accepted: `turkey_predictions/<model>_<m1|m2|m3|post1|post2>.csv`

## US Improvement Layer

`python data/us_improvement.py` is a US-only robustness layer. It does not
touch Turkey outputs and does not replace the pre-specified 17-model benchmark.
It writes:

- `predictions/combo_<name>_<m1|m2|m3|post1>.csv` forecast-combination predictions,
- `data/evaluation_results_us_improved.csv` with the 17 base models plus
  combinations,
- `docs/us_model_diagnostics.md` with DFM/MLP error diagnostics,
- `figures/us_improved_m3_ranking.png`,
  `figures/us_improved_m3_ex_2020q2_ranking.png`, and
  `figures/us_improved_vintage_profiles.png`.

The improved US evaluation adds two robustness panels:

| Panel | Definition | Purpose |
|---|---|---|
| `full_ex_2020q2` | Full sample excluding 2020-Q2 | Checks dependence on the single largest COVID contraction. |
| `non_covid` | Pre-COVID plus post-COVID observations | Checks normal-period performance outside 2020-Q2..2021-Q4. |

Forecast combinations are paper-facing robustness checks, not replacements for
the 17 pre-registered model classes. They should be discussed separately to
avoid overstating ex post combination gains.
