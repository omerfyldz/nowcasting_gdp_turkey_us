# Scaling Policy (Pipeline B)

**Rule:** Scaling, when used, is fit on the training fold only inside the
rolling forecast loop. Never fit a scaler on the full sample. Never put
scaling into a `data/` build script.

## Per-model decisions

| Model | StandardScaler? | Reason |
|-------|-----------------|--------|
| OLS | No | Affine-equivariant; scaling does not change predictions. |
| ARMA | No | Univariate, scale invariant. |
| VAR | No | Equivariant under linear transformation; scaling complicates impulse responses. |
| DeepVAR | No | Same as VAR. |
| BVAR | **Yes** | Minnesota prior shrinks coefficients toward fixed values; meaningful only on standardized data. |
| DFM | **Yes** | Factor loadings depend on series variance; without scaling, high-variance series dominate factors. |
| Lasso / Ridge / ElasticNet | **Yes** | Penalty applies uniformly to all coefficients; without scaling, the penalty unfairly shrinks small-scale variables more. |
| MIDAS / MIDASML | **Yes** | Same penalty argument as Lasso family. |
| MLP | **Yes** | Gradient descent ill-conditioned without scaling. |
| LSTM | **Yes** | Same as MLP; also activation saturation at large inputs. |
| Random Forest | No | Tree splits are scale-invariant. |
| Gradient Boosting | No | Same as RF. |
| XGBoost | No | Same as RF. |
| Decision Tree | No | Same as RF. |

## Implementation pattern (inside the rolling loop)

```python
from sklearn.preprocessing import StandardScaler

for forecast_date in forecast_dates:
    train = data.loc[data.date < forecast_date, :]
    test  = data.loc[data.date == forecast_date, :]

    if SCALE:
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)   # fit on TRAIN only
        test_X_scaled  = scaler.transform(test_X)        # apply to test
        # NB: do NOT include the test row in scaler.fit_transform.
    else:
        train_X_scaled, test_X_scaled = train_X, test_X

    model.fit(train_X_scaled, train_y)
    pred = model.predict(test_X_scaled)
```

## ORDER OF OPERATIONS — READ THIS BEFORE WRITING ANY ROLLING-LOOP CODE

Inside the rolling forecast loop, the order of `gen_lagged_data` →
`mean_fill` → scaler `fit` → `transform` → `model.fit` → `model.predict`
is **not interchangeable**. Each step depends on the previous:

```
for forecast_date in forecast_dates:                      # rolling
    train_full  = data.loc[data.date < forecast_date]
    test_row    = data.loc[data.date == forecast_date]

    # 1. Apply ragged-edge mask (per-vintage publication lags)
    train_full = gen_lagged_data(metadata, train_full, last_date=forecast_date, lag=L)
    test_row   = gen_lagged_data(metadata, test_row,   last_date=forecast_date, lag=L)

    # 2. Mean-fill: COMPUTE MEANS FROM TRAINING FOLD ONLY, apply to both train and test
    train_filled, test_filled = mean_fill_dataset(train_full, train_full), \
                                mean_fill_dataset(train_full, test_row)

    # 3. Scaler: FIT ON TRAINING FOLD ONLY, apply to both
    if SCALE_FLAG:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_filled[feature_cols])
        test_X  = scaler.transform(test_filled[feature_cols])
    else:
        train_X, test_X = train_filled[feature_cols].values, test_filled[feature_cols].values

    # 4. Fit + predict
    model.fit(train_X, train_filled[target])
    pred = model.predict(test_X)
    store(pred, forecast_date)
```

**EMPHATIC RULES, violation of any of which invalidates the entire benchmark:**

1. **Mean-fill MUST be inside the rolling loop and computed from the training
   fold only.** A mean computed on the full sample (train + val + test)
   leaks future information into past training rows. The same applies to
   the scaler.
2. **`gen_lagged_data` MUST be called BEFORE mean-fill.** Mean-fill replaces
   NaNs with training-fold means; if the publication-lag mask is applied
   afterwards, the masked NaNs immediately get filled by the mean and the
   ragged-edge information is destroyed.
3. **Scaler MUST be fit AFTER mean-fill.** A scaler fit on data with NaNs
   crashes; a scaler fit before mean-fill fits on observed-only rows and
   then fails to transform the filled rows correctly because the
   mean-filled values are not on the same scale.
4. **The test row is processed through `transform` only, never `fit_transform`.**
   Forgetting this is the single most common silent leakage in nowcasting
   benchmarks.
5. **For variable selection (one-shot, before notebooks), the order does
   not matter as much — there is one global mean-fill, one global flatten,
   one global scaler — but the same training-only fit applies.**

If any of these rules feels inconvenient, you are about to write a paper
that will not survive review. Do not skip them.

## What the scaler must NOT see
- Future observations (post-forecast-date rows).
- The target variable y. (Standardising y is optional and orthogonal to the
  X-scaling decision; see Q10 / `tuning_policy.md`.)
- COVID dummies. They are 0/1 by construction and have zero variance in the
  pre-2020 training window. Scaling them produces NaN. Drop them from the
  scaler input or pass them through unchanged.
