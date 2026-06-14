# Turkey Post-Release Vintage Implementation

Date: 2026-05-14

This document records the Turkey `post1`/`post2` extension implemented after the US pipeline completion pass.

## Goal

Add Turkey `post1` and `post2` robustness horizons while preserving the current `m1`/`m2`/`m3` cross-country benchmark.

Turkey horizons should be:

- `m1`: GDP target quarter end minus 2 months.
- `m2`: GDP target quarter end minus 1 month.
- `m3`: GDP target quarter end.
- `post1`: GDP target quarter end plus 1 month.
- `post2`: GDP target quarter end plus 2 months.

`post1`/`post2` should be treated as robustness horizons, not as replacements for the symmetric US-Turkey comparison.

## Implemented Infrastructure Changes

- `turkey_data/turkey_helpers.py` imports and exposes `gen_vintage_data` and `make_supervised_vintage_frame` from `data/helpers.py`.
- Turkey Python notebooks keep the GDP target quarter fixed while using the vintage date only for publication-lag masking. This mirrors the US `post1` correction and avoids accidentally predicting the month after the GDP target quarter.
- Turkey R notebooks use `vintage_offsets <- c(m1 = -2, m2 = -1, m3 = 0, post1 = 1, post2 = 2)` while keeping `target_date` separate from `vintage_date`.
- Output names preserve the existing mixed Turkey convention. The evaluator accepts both `turkey_predictions/<model>_tr_<vintage>.csv` and legacy `turkey_predictions/<model>_<vintage>.csv` non-destructively.
- `data/evaluate.py` is configured for Turkey `m1`, `m2`, `m3`, `post1`, and `post2`, with audit checks for missing files, required columns, row counts, date range, duplicate dates, and non-finite values.

## Cross-Validation Policy

Use cross-validation wherever the model class has a defensible tuning parameter and the package can complete:

- Lasso, Ridge, ElasticNet: keep TimeSeriesSplit-based CV or built-in CV estimators.
- Decision Tree, Random Forest, Gradient Boosting, XGBoost: keep TimeSeriesSplit/GridSearchCV and frozen best parameters for rolling test predictions.
- MLP/LSTM/DeepVAR: tune architecture/regularization on the validation window or TimeSeriesSplit where practical; keep random-seed ensembles only after hyperparameters are selected.
- MIDAS: use validation-window specification/weight selection where feasible.
- MIDAS-ML: `cv.sglfit` was attempted during the post-release rerun but aborted the Jupyter process with a low-level ZMQ assertion. The completed outputs therefore use the documented fixed-penalty fallback `sglfit(lambda = 0.01)`.
- BVAR: compare feasible reduced predictor sets and/or shrinkage settings on the validation window; do not silently switch to a fixed reduced set without logging the selection rule.
- DFM: after the full Turkey extension, a separate validation-selection pass compared Cat2, selected10, and Cat3 panels on 2012-2017 validation RMSFE across `m1`/`m2`/`m3`. Cat2 was selected, frozen, and retrained through 2017 for 2018-2025 testing.
- ARMA and OLS remain benchmark/simple models; do not add artificial tuning unless the notebook already uses a defensible order/specification selection step.

## Model-Specific Implementation Notes

- CV/tuned Python models remain CV/tuned: Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and MLP.
- ARMA preserves a no-target-leakage rule: post-release horizons still exclude the target-quarter GDP actual from the ARMA history.
- OLS, VAR, LSTM, DeepVAR, and the flattened supervised ML notebooks use target-date/vintage-date separation.
- MIDAS-ML uses documented fixed-penalty `sglfit(lambda = 0.01)` after the failed `cv.sglfit` attempt.
- DFM uses the validation-selected Cat2 panel plus target in the final outputs. This replaces the earlier Cat3 diagnostic while avoiding test-window model selection.

## Validation Gates

Before accepting the Turkey extension:

- Every Turkey model must produce `m1`, `m2`, `m3`, `post1`, and `post2`.
- Each prediction file must have 32 rows from `2018-03-01` through `2025-12-01`.
- Required columns: `date`, `actual`, `prediction`.
- No duplicate dates and no NA/non-finite actuals or predictions.
- The extended evaluator should produce `17 models x 5 vintages x 4 panels = 340` Turkey rows.
- Keep a separate comparison table for the original symmetric `m1`/`m2`/`m3` benchmark.

## Paper Caveats To Preserve

- The exercise remains pseudo-real-time because GDP targets use final revised values.
- Turkey `post1`/`post2` answer a different question than `m1`/`m2`/`m3`: they use post-quarter information but still forecast the same GDP target quarter.
- MIDAS-ML should be reported as a constrained implementation: cross-validated `cv.sglfit` was attempted but failed at the notebook-process level, so fixed-penalty `sglfit(lambda = 0.01)` was used to complete the empirical package.
