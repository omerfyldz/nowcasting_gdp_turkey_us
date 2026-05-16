# US Pipeline Preflight

Date: 2026-05-14

Scope: US pipeline only. Turkey notebooks, Turkey data, and Turkey predictions are intentionally unchanged in this pass.

## Pre-Run Checks Completed

- No leftover notebook execution process was running before the preflight.
- All 17 US notebooks contain the US-only `post1` vintage.
- Flattened supervised notebooks now use `make_supervised_vintage_frame(...)`, which separates the GDP target quarter from the simulated information date.
- `model_lstm.ipynb`, `model_deepvar.ipynb`, and `model_var.ipynb` were checked separately because they do not use the flattened supervised helper. They now use explicit target-date/vintage-date masking through `gen_vintage_data(...)`.
- Python helper/evaluator/improvement scripts compile.
- Python notebook code cells parse.
- R notebook code cells parse using `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`.
- A semantic post1 check confirms that the target-quarter row remains `2020-06-01` when the information date is `2020-07-01`, and that raw GDP for the target quarter is masked before prediction.

## Issues Addressed Before Rerun

- `post1` cannot be implemented by simply setting `forecast_date = Q-end + 1 month` in every notebook. That can move the prediction row away from the GDP target quarter. The notebooks now keep `target_date` and `vintage_date` separate where needed.
- The supervised helper now resets the returned test target to `NaN` after mean filling. The target was not used as a predictor before, but this makes the artifact unambiguous.
- US BVAR fallback logging remains in place through `docs/bvar_us_fallback_log.csv`.
- MLP stabilization changes remain in place: smaller networks, stronger regularization, robust scaling, early stopping, and median ensembling.
- MIDAS weighting uses inverse validation-RMSE weights instead of the previous distance-from-maximum rule.
- US improvement logic includes forecast combinations and COVID-robust evaluation panels.

## Rerun Results

The unrestricted US notebook rerun completed successfully. RF was the long-running bottleneck, but it finished without reducing its grid or ensemble specification.

All US notebooks now write `m1`, `m2`, `m3`, and `post1` predictions.

`data/evaluate.py` produced:

- US: 17 models x 4 vintages x 4 panels = 272 rows.
- Turkey at the time of the US preflight: 17 models x 3 vintages x 4 panels = 204 rows. The later Turkey post-release extension now produces 17 models x 5 vintages x 4 panels = 340 rows.

`data/us_improvement.py` produced:

- 24 models/combinations x 4 vintages x 6 panels = 576 rows.

The prediction audit passed for all US and existing Turkey files: required columns are present, row counts are correct, date ranges match the evaluation windows, dates are unique, and actual/prediction values are finite.

## Remaining Caveats For Paper

- The exercise is pseudo-real-time, not true historical real-time vintage evaluation.
- US `post1` and Turkey `post1`/`post2` are robustness horizons and should not replace the symmetric `m1`/`m2`/`m3` comparison.
- US BVAR remains reduced-dimensional rather than a full-panel BVAR.
- US DFM deterioration appears concentrated in COVID observations, especially 2020-Q2 and 2020-Q3; it should be discussed rather than silently optimized away.
- US MLP is no longer explosively unstable after stabilization; it is competitive in `m3` and `post1`. Turkey MLP early-vintage instability remains a separate caveat.
- DM tests remain small-sample diagnostics, especially inside panels.
