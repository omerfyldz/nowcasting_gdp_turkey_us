# Leakage Audit

Last updated: 2026-05-16

## Scope

This audit reviews whether the finalized US/Turkey nowcasting benchmark leaks
future information into model estimation, hyperparameter selection, prediction
construction, or evaluation.

It covers:

- prediction CSV coverage and target consistency;
- country-aware evaluation logic in `data/evaluate.py`;
- Python and R vintage-construction helpers;
- feature-selection scripts;
- Lasso, Ridge, ElasticNet, and MLP tuning blocks;
- known limitations that are not classical leakage but affect interpretation.

## Bottom Line

No direct test-window leakage was found in the finalized prediction files or
country-aware evaluator. The benchmark remains a pseudo-real-time exercise:
publication lags and ragged edges are simulated, but underlying data use final
revised series rather than historical vintage releases.

The main technical issue found was validation-fold preprocessing leakage in
some feature-selection and tuning code, where scaled train+validation matrices
were passed into `TimeSeriesSplit`. This has now been corrected in the active
feature-selection scripts and affected model notebooks by moving scaling inside
`Pipeline`/fold-local preprocessing.

## Checks Performed

### Prediction and Evaluation Files

The evaluator audits every required prediction file before scoring:

- required columns: `date`, `actual`, `prediction`;
- expected row count;
- expected quarterly date sequence;
- duplicate dates;
- missing or non-finite actuals/predictions.

Current expected outputs:

- US: 17 models x 4 vintages x 4 panels = 272 rows.
- Turkey: 17 models x 5 vintages x 4 panels = 340 rows.
- US robustness layer: 24 models/combinations x 4 vintages x 6 panels = 576 rows.

Additional manual checks found that actual target values are consistent across
models up to floating-point tolerance. Apparent R/Python target mismatches were
only numerical precision differences on the order of `1e-16`.

### Vintage Construction

The active Python and R helpers use explicit target-quarter and information-date
logic:

- `target_date` keeps the GDP target quarter row available for nowcasting;
- `vintage_date` controls which indicators are visible;
- unavailable values after the simulated information date are masked to `NA`;
- the target value at the forecasted quarter is explicitly hidden before model
  prediction.

This design avoids the earlier ambiguity of artificial lag arguments and is
more defensible for `m1`, `m2`, `m3`, `post1`, and Turkey `post2`.

### Train/Test and Cross-Validation Splits

No random `KFold` or `train_test_split` usage was found in the active model
notebooks. Model tuning uses time-ordered splits:

- US test window: 2017-2025;
- Turkey test window: 2018-2025;
- hyperparameter and feature-selection windows are before those test windows.

No tuning code was found that directly uses the US 2017-2025 or Turkey
2018-2025 evaluation outcomes to select model hyperparameters.

## Fixed Issue: Scaler Inside Cross-Validation

### Problem

Several scripts/notebooks previously used this pattern:

```python
scaler = StandardScaler()
X_tune_scaled = scaler.fit_transform(X_tune)
model_cv.fit(X_tune_scaled, y_tune)
```

Even though `X_tune` excludes the final test window, this leaks validation-fold
means and variances across `TimeSeriesSplit` folds. It can make internal
cross-validation scores and selected hyperparameters slightly optimistic.

### Fix

Scaling is now inside the cross-validation estimator path.

For penalized linear models:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV(..., cv=TimeSeriesSplit(...))),
])
```

Analogous `Pipeline` logic is used for Ridge and ElasticNet tuning blocks.

For MLP tuning, fold-local preprocessing is explicit through
`ColumnTransformer` + `Pipeline`, with scalable columns transformed inside each
fold and COVID dummy columns passed through.

Files updated:

- `data/feature_selection_ensemble.py`
- `turkey_data/feature_selection_tr.py`
- `model_notebooks/model_lasso.ipynb`
- `model_notebooks/model_ridge.ipynb`
- `model_notebooks/model_elasticnet.ipynb`
- `model_notebooks/model_mlp.ipynb`
- `turkey_model_notebooks/model_lasso_tr.ipynb`
- `turkey_model_notebooks/model_ridge_tr.ipynb`
- `turkey_model_notebooks/model_elasticnet_tr.ipynb`
- `turkey_model_notebooks/model_mlp_tr.ipynb`

Important: the saved prediction CSVs and evaluation tables still reflect the
last completed notebook runs unless the affected notebooks are rerun. The code
is corrected; rerunning is required if the final empirical outputs must fully
reflect the stricter CV preprocessing.

## No Test-Window Leakage Found

The audit did not find:

- models trained on the target quarter's actual GDP value;
- random splitting of time-series rows;
- hyperparameter tuning on the final test period;
- evaluation code selecting models before computing reported scores;
- prediction files with missing, duplicated, or non-finite predictions;
- target-value discrepancies across models beyond floating-point precision.

## Remaining Limitations and Caveats

### Pseudo-Real-Time, Not True Real-Time

The project simulates information availability using release-lag metadata and
ragged-edge masking. However, the underlying macroeconomic series are final
revised data. Therefore:

- the benchmark is pseudo-real-time;
- it is not a historical-vintage real-time database exercise;
- revision uncertainty is not measured.

This is the central limitation to state in the paper.

### Final Revised GDP Targets

GDP `actual` values are final transformed observations, not first-release GDP
estimates available to forecasters at the time. This matters because GDP
revisions can change both the target path and model rankings. The project
should therefore avoid claims about operational real-time performance.

### COVID Dummy Caveat

The COVID dummy variables are deterministic date indicators. They are not
ordinary high-frequency indicators released by statistical agencies. They help
control an extreme structural break, but they encode ex-post knowledge that the
COVID period is special.

Recommended framing:

- keep COVID dummies in the main benchmark as shock controls;
- discuss them explicitly as a modeling choice;
- add a no-COVID robustness run if time permits.

### Feature Selection Uses Historical Training Windows

Feature selection is pre-test for both countries:

- US feature selection uses the pre-2008 training window;
- Turkey feature selection uses the pre-2012 training window.

This avoids final test leakage. The updated feature-selection scripts also
avoid validation-fold scaling leakage inside their CV selectors.

### Post-Hoc Model Combinations

US combination forecasts are useful robustness outputs, but they were added
after inspecting base-model behavior. They should remain a robustness extension,
not part of the primary 17-model ranking.

### Turkey DFM Scope

The final Turkey DFM is validation-selected Cat2 plus target. It is not a full
54-variable Tier-C DFM. Broader sparse DFM panels were attempted but failed
inside `nowcastDFM`, and this must be stated in methodology/limitations.

## Recommended Next Steps

1. Rerun the affected feature-selection scripts if the variable lists should
   fully reflect fold-safe scaling.
2. Rerun affected Lasso/Ridge/ElasticNet/MLP notebooks for both countries if
   the prediction CSVs must reflect the corrected tuning code.
3. Regenerate evaluation CSVs, summaries, and figures after reruns.
4. Add a no-COVID robustness layer for the main 17-model benchmark.
5. In the paper, use the term "pseudo-real-time nowcasting exercise" and
   explicitly distinguish it from true real-time vintage-data evaluation.

## External Methodological Basis

The leakage fix follows scikit-learn's guidance that preprocessing statistics
must be learned from training data only, and that `Pipeline` is the standard
way to avoid leakage during cross-validation. The nowcasting caveats follow the
standard literature's distinction between real-time data flow, publication-lag
ragged edges, and final revised macroeconomic data.
