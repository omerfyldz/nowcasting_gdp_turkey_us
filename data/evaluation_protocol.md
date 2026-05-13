# Evaluation Protocol (Pipeline B)

## Loss Functions

For every model, country, target quarter, and m1/m2/m3 vintage, the evaluator records:

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

Python notebooks implement the same idea directly with `forecast_date` and `gen_lagged_data(..., lag=0)`.

## Model-Set Caveats

- US BVAR uses a Lasso-80 reduced predictor set, not full Cat3.
- Turkey BVAR uses locked Cat2 predictors plus the GDP target.
- US DFM uses monthly Cat3 plus target; quarterly Cat3 variables and COVID dummies are excluded to keep `nowcastDFM` feasible.
- Turkey DFM uses the complete 22-variable Cat3 monthly set plus target; Tier-C short-history variables were tested but excluded because the full sparse panel fails inside `nowcastDFM`.
- Turkey MIDAS-ML uses fixed-penalty `sglfit(lambda = 0.01)` because rolling `cv.sglfit` did not complete.
- MLP m1/m2 outputs are finite and evaluated, but their very large RMSFE should be treated as model instability in the paper.

## Evaluator Rules

`data/evaluate.py` fails if any required prediction file is missing, has the wrong row count, has missing required columns, has date-range mismatches, or contains missing/non-finite actuals or predictions.

Raw predictions are preserved on disk:

- US: `predictions/<model>_<m1|m2|m3>.csv`
- Turkey preferred: `turkey_predictions/<model>_tr_<m1|m2|m3>.csv`
- Turkey legacy accepted: `turkey_predictions/<model>_<m1|m2|m3>.csv`
