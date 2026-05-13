# Hyperparameter Tuning Policy (Pipeline B)

## Sample split (fixed across all models)

| Slice | Range | Crisis | Use |
|-------|-------|--------|-----|
| Train | 1959-01 .. 2007-12 | 1973-75, 1980-82, 2001 | Lasso/EN/RF feature selection; HP-search inner-fold training |
| Validation | 2008-01 .. 2016-12 | **GFC 2008-09** | Hyperparameter selection (TimeSeriesSplit inside) |
| Test | 2017-Q1 .. 2025-Q4 (35 q) | **COVID 2020Q2-2021Q4** | Out-of-sample, rolling, m1/m2/m3 vintages. Capped at 2025-Q4 because FRED-QD vintage 2026-03 has no Q1 2026 realised target. |

**Reasoning (Addendum 2 of the plan):** the previous split (train 1959-2010,
val 2011-2016, test 2017-2026) had no recession in the validation window.
HPs tuned on that window cannot be evaluated for crisis robustness. The
revised split puts a major shock in BOTH validation (GFC) and test (COVID),
so HP choices that survive GFC are reasonably likely to generalise to
COVID. Cascaldi-Garcia et al. 2024 use the same partition logic.

## Per-model decisions

| Model | Tune? | Hyperparameters and grids |
|-------|-------|---------------------------|
| OLS | No | None. |
| ARMA | No | (p, q) selected by AIC inside fit. |
| VAR | No | Lag order p selected by AIC inside fit. |
| DeepVAR | No | Architecture fixed by paper; only seed varies. |
| DFM | No | Number of global factors and per-block factors set by economic prior (1 / 1 each). |
| BVAR | No | Minnesota tightness fixed at literature default (lambda1=0.1). |
| Lasso | **Yes** | alpha via `LassoCV(cv=TimeSeriesSplit(5))` on train+val combined, refit on full. |
| Ridge | **Yes** | alpha via `RidgeCV(cv=TimeSeriesSplit(5))`. |
| ElasticNet | **Yes** | alpha and l1_ratio via `ElasticNetCV(l1_ratio=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0], cv=TimeSeriesSplit(5))`. Note: keep the lower-half values (0.1-0.5) so CV can pick Ridge-leaning if appropriate -- the rigorous selector showed l1_ratio=0.25 wins on this dataset, which a Lasso-biased grid would have hidden. |
| MIDAS | **Yes** | nealmon/beta polynomial parameters; tuned on validation by RMSFE. |
| MIDASML | **Yes** | sg-LASSO penalty; CV on validation. |
| RF | **Yes** | n_estimators in {200, 500, 1000}; max_depth in {None, 8, 16}; min_samples_leaf in {1, 3, 5}. |
| Gradient Boosting | **Yes** | learning_rate in {0.01, 0.05, 0.1}; n_estimators in {200, 500}; max_depth in {3, 5}. |
| XGBoost | **Yes** | learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda; small grid. |
| Decision Tree | **Yes** | max_depth in {None, 5, 10, 20}; min_samples_leaf. |
| MLP | **Yes** | hidden_layer_sizes in {(64,), (128,), (64,64)}; alpha in {1e-4, 1e-3, 1e-2}; lr fixed via `adam`. Note: with 53 vars × 4 lags = 212 features for ~194 quarterly obs (ratio 1.09), the CV grid includes alpha=1e-2 as an upper bound for stronger L2 regularization. |
| LSTM | **Yes** | n_units in {32, 64}; n_layers in {1, 2}; dropout in {0.0, 0.2, 0.3}; epochs early-stopped on validation loss. Note: with 212 input features for ~194 training obs, dropout=0.3 is included as an upper grid option so CV can push toward stronger regularization if needed. |

## Validation procedure

```python
from sklearn.model_selection import TimeSeriesSplit

# Inside notebook, BEFORE the rolling test loop:
tscv = TimeSeriesSplit(n_splits=5)
best_hp = None
best_loss = np.inf
for hp in hp_grid:
    losses = []
    for tr_idx, va_idx in tscv.split(train_val):
        model = build_model(hp)
        model.fit(train_val[tr_idx])
        pred = model.predict(train_val[va_idx])
        losses.append(rmsfe(pred, train_val[va_idx, target]))
    if np.mean(losses) < best_loss:
        best_loss = np.mean(losses); best_hp = hp

# Then refit with best_hp on the full train+val for each rolling forecast.
```

## Rules
1. Never use random K-fold (e.g. `KFold`) on time-series data; only `TimeSeriesSplit`.
2. Never select HPs on the test window.
3. Once HPs are selected, **freeze them**. The rolling test loop refits the
   model with the same HPs each iteration; it does not re-tune.
4. Document the chosen HPs in the notebook's first cell.
