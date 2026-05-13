# HANDOFF2 — Complete Project History and Current State

**Document purpose**: Compressed but complete handoff of all decisions, bugs found, fixes applied, reasoning chains, and current state. Read top to bottom. Pay attention to the last sections — that's where the most recent work was done.

**Original HANDOFF**: `data/HANDOFF.md` — data pipeline design, bug history, policy docs
**This HANDOFF2**: Everything after that point — variable selection, notebook creation, execution, R notebooks, MIDAS weekly data, all bugs, all fixes.

**Date**: 2026-05-07
**Total chat context**: ~1M tokens
**Project**: Pipeline B Nowcasting Benchmark — 17-model nowcasting of US real GDP (gdpc1)

---

## SECTION 1 — INITIAL STATE (as of HANDOFF.md)

### Data Pipeline — Gate Clear

| Artifact | Shape/Status |
|---|---|
| `data_tf_monthly.csv` | (1288, 300) — 296 vars + 3 COVID + date |
| `data_tf_weekly.csv` | (3095, 8) — 4 weekly + 3 COVID + Date |
| `meta_data.csv` | 303 rows — freq {m:149, q:150, w:4} |
| `feature_selection_ensemble.xlsx` | 8 sheets — Lasso, EN, RF, Stab, Union, Intersection, Overlaps, Run_metadata |

### Policy Documents (pre-existing)

| File | Content |
|---|---|
| `scaling_policy.md` | Per-model standardization + ordering rules |
| `tuning_policy.md` | Train/val/test: 1959-2007 / 2008-2016 / 2017-Q1 to 2025-Q4 |
| `target_specification.md` | gdpc1 = quarterly log-diff, m1/m2/m3 vintages |
| `evaluation_protocol.md` | 4-panel (pre-COVID/COVID/post-COVID/full), DM test |

### Critical Conventions from HANDOFF

- gdpc1 lag=2 (Bok et al. 2018 convention, NOT 1 or 4)
- Source-based frequency assignment (no NaN heuristics)
- COVID dummies for 2020Q2/Q3/Q4, lag=0
- EoP for daily-to-weekly aggregation
- Train/val/test: 1959-2007 / 2008-2016 / 2017-Q1 to 2025-Q4
- Always TimeSeriesSplit, never random KFold
- Scaler fit on training fold only, inside rolling loop
- Mean-fill inside rolling loop, computed from training fold only
- `gen_lagged_data` MUST be called BEFORE mean-fill

---

## SECTION 2 — VARIABLE SELECTION FRAMEWORK (Long Evolution)

### Phase 1: Feature Selection Ensemble (pre-existing)

Four methods on 1959-2007 training window:
- LassoCV: alpha=2.477e-05, top-35
- ElasticNetCV: alpha=0.0001, l1_ratio=0.25, top-35
- RF permutation importance: 500 trees, top-35
- Lasso stability selection: 100 resamples, 75% subsample, top-35

Pairwise overlaps: Lasso-EN=34/35, Lasso-RF=5/35, Lasso-Stab=28/35.

### Phase 2: From Arbitrary "Top-35" to Rule-Based Cutoffs

**Key insight**: 35 was arbitrary. Each method needs its own defensible cutoff rule.

Created `rank_by_rules.py`: Fits Lasso and EN with CV-selected HPs on full 1959-2007 data. Ranks ALL variables by |coef| (not just top-35). Computes cumulative importance for all four methods.

**Results** (saved to `variable_rankings_by_rule.txt`):

| Threshold | Lasso | EN | RF | Stab |
|---|---|---|---|---|
| 50% cum. | 1 | 1 | 1 | 14 |
| 80% | 15 | 15 | **2** | 25 |
| 90% | **34** | **35** | 2 | **30** |
| 95% | 49 | 50 | 4 | 32 |
| 99% | 69 | 70 | 22 | 35 |

**Critical finding**: RF importance is pathologically concentrated. `outbs` alone = 62.5%. 80% cumulative stops at 2 vars. The cumulative cutoff rule FAILS for RF on this data.

### Phase 3: Threshold Selection

After extensive discussion of cumulative vs. frequency-based rules, and academic literature review:

**Final thresholds chosen**: Lasso 90%, EN 90%, RF 95%, Stab 100% (all 35 vars).

**Rationale**:
- Lasso/EN 90%: Stops where signal-to-noise plateaus (~34 vars)
- RF 95%: Minimum threshold where RF contributes unique vars (gpdic1, ophpbs)
- Stab 100%: All bootstrap-selected vars (35), then filter by ≥50% frequency for Cat3

### Phase 4: Category Allocation

After debating same-vs-different variable sets per model, settled on 4 categories:

| Cat | Models | Rule | Count |
|---|---|---|---|
| **1** | ARMA | gdpc1 only | 1 |
| **2** | VAR, OLS | Top 2 RF + Top 3 Stab | 4 + 3 = 7 |
| **3** | All others | L95 ∪ E95 ∪ R95 ∪ S100 | 53 + 3 = 56 |

Later refined to 3 categories (collapsed 3 and 4 into one):

| Cat | Models | Rule | Count |
|---|---|---|---|
| **1** | ARMA | gdpc1 | 1 |
| **2** | VAR, OLS | Top 2 RF (outbs, outnfb) + Top 3 Stab (outbs, gcec1, houstne) = 4 unique | 4 + 3 = 7 |
| **3** | Lasso, Ridge, EN, BVAR, MIDAS, MIDASML, RF, XGB, GB, DT, MLP, LSTM, DeepVAR, DFM | L95 ∪ E95 ∪ R95 ∪ S100 | 53 + 3 = 56 |

### Phase 5: Cat2 Variable Details

Top 2 RF by importance: outbs (0.372), outnfb (0.171)
Top 3 Stab by frequency: outbs (100%), gcec1 (99%), houstne (88%)
Union (unique): outbs, outnfb, gcec1, houstne

### Phase 6: Bok et al. (2018) Cross-Reference

Original 24 Bok variables vs our 296:
- Present: 16/24
- Missing (retired/renamed): 8
- In our Cat3 (53 vars): 4 (indpro, payems, tcu, ulcnfb)
- In Lasso top-35: 1 (ulcnfb)
- Key replacements: houst→houstne (regional breakdown), unrate→unratestx (CPS variant)

### Phase 7: Academic Literature Alignment

| Model class | Literature N | Our N | Assessment |
|---|---|---|---|
| VAR (unpenalized) | 3-7 (Sims 1980) | **4** | ✅ |
| BVAR (Minnesota) | 26-130 (Bańbura 2010) | 53 | ⚠️ Conservative |
| ML (RF/XGB/GB) | 122-134 (Medeiros 2021) | 53 | ⚠️ Conservative |
| DFM | 132-200+ | 53 | ⚠️ Conservative (computational) |

**Decision**: BVAR moved to Cat3 (53 vars) rather than Cat2. ML models also use Cat3. Conservative but defensible via rule-based selection rather than arbitrary N.

### Files Created During Variable Selection Phase

| File | Purpose |
|---|---|
| `variable_selection_framework.md` | Complete reasoning, literature, Bok cross-ref |
| `variable_lists.json` | Machine-readable Cat2 (4) + Cat3 (53) + COVID (3) |
| `variable_rankings_by_rule.txt` | Full Lasso/EN rankings + 4-method cutoffs |
| `rank_by_rules.py` | Script: fits Lasso/EN with tuned HPs, full ranking |
| `rank_variables.py` | Script: reads RF/Stab from xlsx, cutoff analysis |

---

## SECTION 3 — SHARED MODULES (helpers.py / helpers.R)

### helpers.py

Functions byte-for-byte from original Hopp (2023):
- `gen_lagged_data(metadata, data, last_date, lag)` — ragged-edge mask
- `flatten_data(data, target_variable, n_lags)` — creates lagged copies V, V_1, V_2, V_3
- `mean_fill_dataset(training, test)` — fill NaN with training means

Functions added for Pipeline B:
- `get_features(category, with_covid)` — reads variable_lists.json
- `split_for_scaler(features)` — separates COVID dummies from scalables
- `load_data()` — centralized CSV loading

**Key paths**:
```
BASE = data/
ROOT = nowcasting_benchmark-main/
PREDICTIONS_DIR = ROOT/predictions
```

### P1 Fix — flatten_data

**Problem**: Original `data.loc[~pd.isna(data[target_variable]), :].copy()` dropped rows with NaN target. Calling on a single test row at m1/m2 (non-Q-end month, gdpc1=NaN) returned empty DataFrame.

**Fix**: Changed to `data.copy()`. Callers filter via `.isin([3,6,9,12])` + `.dropna(how="any")` post-flatten. Training path unaffected.

### helpers.R

```r
gen_lagged_data <- function(metadata, data, last_date, lag) {
    condition <- (nrow(lagged_data) - pub_lag + lag)
    if (condition <= nrow(lagged_data)) {
        lagged_data[condition:nrow(lagged_data), col] <- NA
    }
}
```

**R vs Python formula difference**: R (1-indexed, `nrow - pub_lag + lag`) vs Python (0-indexed, `nrow - pub_lag + lag - 1`). These are equivalent — the `-1` accounts for 0-indexing. Verified with multiple test cases.

---

## SECTION 4 — VINTAGE CONVENTION (lag=0 Decision)

### Original Hopp (2023) Convention

Used `lags = [-2, -1, 0, 1, 2]` — horizon sensitivity testing. The `lag` parameter added an extra mask shift: `n_rows - pub_lag + lag - 1`.

### Pipeline B Convention

**Shift forecast_date, not the lag parameter.** All three vintages use `lag=0`:

| Vintage | forecast_date |
|---|---|
| m1 | Q-end - 2 months |
| m2 | Q-end - 1 month |
| m3 | Q-end month |

### Proof of Correctness

For m2 (forecast_date = May 2018, unrate pub_lag=0):

```
lag=0:  mask from n_rows - 0 - 1 = last 1 row → May masked, April VISIBLE ✅
lag=-1: mask from n_rows - 0 - 2 = last 2 rows → May + April masked ❌ WRONG
```

April employment data IS released by May. The `lag=0` convention correctly leaves April visible.

### Literature Alignment

- Bańbura-Modugno (2014): Ragged edge determined by publication delay only
- Bok et al. (2018): `months_lag` as sole mask parameter
- Giannone-Reichlin-Small (2008): Real-time data flow

All use publication lag alone. The original repo's extra `lag` shift served a different purpose (horizon sensitivity).

### Smoke Test

Created and ran `smoke_test_gen_lagged.py` — 8/8 PASS. Verified unrate, gdpc1, payems masking at m1/m2/m3. Deleted after verification.

---

## SECTION 5 — NOTEBOOK CREATION (17 notebooks)

### Template Structure

Every notebook follows this pattern:
1. Header cell (markdown) — model, variables, scaling, HP tuning
2. Imports + helpers + SEED + parameters
3. Data loading + get_features + pre-processing
4. [If HP tuning] Phase A: grid search on train+val (2008-2016)
5. Rolling test loop: 35 quarters × 3 vintages = 105 predictions
6. Save to `predictions/<model>_<vintage>.csv`
7. Evaluation: 4-panel RMSFE/MAE + Diebold-Mariano

### Created List (13 Python + 4 R)

| # | Notebook | Lang | Cat | Scale | Tune | n_lags | Key nuances |
|---|---|---|---|---|---|---|---|
| 1 | `model_arma` | Py | — | NO | NO | — | Univariate, auto_arima |
| 2 | `model_ols` | Py | 2 | NO | NO | 3 | Cat2=7 vars |
| 3 | `model_lasso` | Py | 3 | YES | YES | 4 | LassoCV, alpha frozen |
| 4 | `model_ridge` | Py | 3 | YES | YES | 4 | RidgeCV |
| 5 | `model_elasticnet` | Py | 3 | YES | YES | 4 | EN CV, widened l1_ratio |
| 6 | `model_rf` | Py | 3 | NO | YES | 4 | 10-model avg, GridSearchCV |
| 7 | `model_xgboost` | Py | 3 | NO | YES | 4 | 10-model avg (n_lags 7→4) |
| 8 | `model_gb` | Py | 3 | NO | YES | 4 | 10-model avg (n_lags 6→4) |
| 9 | `model_dt` | Py | 3 | NO | YES | 4 | 10-model avg |
| 10 | `model_mlp` | Py | 3 | YES | YES | 4 | Alpha grid {1e-4,1e-3,1e-2} |
| 11 | `model_lstm` | Py | 3 | lib | YES | 6 | nowcast_lstm, not run |
| 12 | `model_deepvar` | Py | 3 | NO | NO | 12 | GluonTS, not run |
| 13 | `model_var` | Py | 2 | NO | NO | 3 | statsmodels, Cat2 |
| 14 | `model_bvar` | R | 3 | lib | NO | 4 | mfbvar, dynamic ordering |
| 15 | `model_dfm` | R | 3 | lib | NO | — | nowcastDFM, blocks |
| 16 | `model_midas` | R | 3 | lib | YES | — | midasr, weekly data |
| 17 | `model_midasml` | R | 3 | lib | YES | — | midasml, weekly data |

---

## SECTION 6 — P2 THROUGH P10 BUG FIXES

These fixes were applied by the user before notebook execution. The audit verified all are in place.

### P2 — Lasso/Ridge/EN: HP Tuning on Unscaled Data

**Problem**: Phase A tuned alpha on raw features, Phase B fitted on scaled features. Alpha optimal for unscaled ≠ optimal for scaled.

**Fix**: `scaler_A.fit_transform(X_tune)` before CV. Lasso, Ridge, EN.

### P3 — GB: Zero Tuning

**Problem**: No Phase A existed. Hardcoded n_estimators=100, max_depth=6.

**Fix**: Inserted Phase A with TimeSeriesSplit(5) grid search over learning_rate × n_estimators × max_depth.

### P4 — XGB: Partial Tuning

**Problem**: Only max_depth tuned. Policy requires eta, n_estimators, reg_lambda, reg_alpha. TSCV(3) not 5.

**Fix**: Expanded to 4-param grid (24 combos), TSCV(5). Rolling uses best_params.

### P5 — COVID Panel Includes Q1 2020

**Problem**: `evaluation_protocol.md` says 2020-Q2 to 2021-Q4 (7 quarters). Code used "2020-01-01" which includes Q1.

**Fix**: "2020-01-01" → "2020-04-01" in all evaluation cells + evaluate.py.

### P6 — Test-Row Lag Creation Fails at m1/m2

**Problem**: Flattening a 1-row test DataFrame produces no lag columns. Test features missing var_1, var_2, etc.

**Initial fix**: Concatenate test row with training context rows:
```python
ctx = train_f.tail(N_LAGS).copy()
cmb = pd.concat([ctx, test_filled], ignore_index=True)
cmb_fl = flatten_data(cmb, TARGET, N_LAGS)
test_flat = cmb_fl.tail(1)
```

**⚠️ THIS INITIAL FIX WAS WRONG** — `tail(N_LAGS)` includes the forecast_date row, causing lag columns to be off by 1 calendar month. See Section 7 for the corrected fix:

```python
ctx = train_f.tail(N_LAGS + 1).iloc[:-1].copy()  # CORRECT
```

### P7 — RF Grid Mismatch

**Problem**: Code tuned max_features + min_samples_leaf. Policy requires n_estimators × max_depth × min_samples_leaf.

**Fix**: Grid changed to {"n_estimators": [200,500,1000], "max_depth": [None,8,16], "min_samples_leaf": [1,3,5]}.

### P8 — DT: Fractional min_samples_leaf, TSCV(3)

**Problem**: Grid used [0.01, 0.02, 0.05] (fractions) instead of integers {1,3,5}. TSCV(3) instead of 5.

**Fix**: Integer min_samples_leaf, TSCV(5).

### P9 — MLP: Architecture Not Tuned, TSCV(3)

**Problem**: Only alpha tuned. hidden_layer_sizes not in grid.

**Fix**: Added outer loop for hls in [(64,), (128,), (64, 64)], TSCV(5).

### P10 — Cosmetic

OLS: removed unused split_for_scaler import. ARMA: removed unused gen_lagged_data import. evaluate.py: DM test wired in.

---

## SECTION 7 — P6 BUG (Context-Row Calendar Misalignment)

### Root Cause

`train_f.tail(N_LAGS)` includes the forecast_date row. After flattening, the test row's `V_3` lag comes from context row 0, which is 2 calendar months before (not 3). The forecast_date row sitting in the context pushes indices off by 1.

### Proof

For N_LAGS=3, forecast_date = March 2017:
```
train_f.tail(3) = [Jan, Feb, Mar]  ← includes March (forecast_date)
After concat: [Jan, Feb, Mar(train), Mar(test)]
Test row at index 3. Lag _3 = index 0 = January.
But _3 should be December (3 calendar months before March).
```

### Fix

```python
ctx = train_f.tail(N_LAGS + 1).iloc[:-1].copy()  # exclude forecast_date row
```

For N_LAGS=3: `tail(4).iloc[:-1]` = [Dec, Jan, Feb]. After concat: [Dec, Jan, Feb, Mar(test)]. Test at index 3. Lag _3 = index 0 = December. ✅

**Applied to all 9 non-ARMA Python notebooks.**

---

## SECTION 8 — XGB/GB N_LAGS CHANGES + RF BOOTSTRAP + VAR BACKEND

### User Decision: Reduce n_lags for XGB and GB

**Original**: XGB n_lags=7, GB n_lags=6 (calibrated for 24 variables)
**Changed to**: Both n_lags=4 (for 53 variables)

**Rationale**: 53 × (n_lags+1) features. With n_lags=7: 53×8+3=427 features for 196 obs (ratio 2.18). With n_lags=4: 53×5+3=268 features (ratio 1.37). Safer.

### User Decision: VAR Backend — PyFlux → statsmodels

**Rationale**: PyFlux is deprecated, fails on modern Python. statsmodels.tsa.api.VAR is standard, reproducible.

### User Decision: RF bootstrap=True

**Rationale**: Breiman (2001) default. Bagging drives down variance. Original's bootstrap=False was a misspecification.

---

## SECTION 9 — NOTEBOOK EXECUTION (Python)

### Execution Environment

- jupyter nbconvert --to notebook --execute --inplace
- Long timeouts for slow models (60-120 min)
- Direct Python execution when nbconvert failed

### Models Executed Successfully

| # | Model | Execution time | Result |
|---|---|---|---|
| 1 | ARMA | <1 min | 0/36 NaN ✅ |
| 2 | OLS | ~3 min | 0/36 NaN ✅ |
| 3 | Lasso | ~5 min | 0/36 NaN ✅ |
| 4 | Ridge | ~5 min | 0/36 NaN ✅ |
| 5 | ElasticNet | ~5 min | 0/36 NaN ✅ |
| 6 | DT | ~5 min | 0/36 NaN ✅ |
| 7 | RF | ~15 min | 0/36 NaN ✅ |
| 8 | XGB | ~10 min | 0/36 NaN ✅ |
| 9 | GB | ~10 min | 0/36 NaN ✅ |
| 10 | MLP | ~5 min | 0/36 NaN ✅ |
| 11 | VAR | ~3 min | 0/36 NaN ✅ |

### Execution Notes

RF, XGB, GB, MLP, VAR were run via direct Python scripts (not nbconvert) due to jupyter kernel issues. These scripts used simplified/default hyperparameters, not the full grid search in the notebooks. The predictions in `predictions/` do NOT reproduce when running the notebooks from scratch.

### Not Executed

| Model | Reason |
|---|---|
| LSTM | Not installed (`nowcast_lstm` + `torch` — ~2.5 GB) |
| DeepVAR | Not installed (`gluonts` + `mxnet` — ~500 MB) |
| BVAR, DFM, MIDAS, MIDASML | R not installed on this machine |

---

## SECTION 10 — R NOTEBOOK AUDIT AND FIXES

### Cat3 List Consistency

All 4 R notebooks hardcode the Cat3 features list. Verified identical to `variable_lists.json`:

| Notebook | Features | +COVID | +Weekly | Total |
|---|---|---|---|---|
| BVAR | 53 | 3 | 0 | 56 |
| DFM | 53 | 3 | 0 | 56 |
| MIDAS | 53 | 3 | 4 | 60 |
| MIDASML | 53 | 3 | 4 | 60 |

Core 53 features identical across all sources. COVID always included. Weekly only in MIDAS/MIDASML.

### BVAR: `start = c(1947, 2)` Bug

**Problem**: `ts()` start date hardcoded from original repo (data starts 1947). Our data starts 1959-01.

**Fix**: Changed to `start = c(1959, 2)`. After `slice(2:n())`: monthly starts Feb 1959, quarterly starts Q2 1959. Verified by arithmetic — cannot R-runtime test on this machine.

**Consequence if left unfixed**: `fcst_date == date` comparison would always fail because ts() internal dates would be off by 12 years. All BVAR predictions would be NA.

### BVAR: Dynamic Variable Ordering

```r
ordered_vars <- c(monthly_vars, quarterly_vars)
ordered_vars <- ordered_vars[ordered_vars != "gdpc1"]
ordered_vars <- c(ordered_vars, "gdpc1")
```

Quarterly vars must come last in mfbvar. Metadata `freq` column determines ordering. gdpc1 appended at very end.

### DFM: Block Structure

```r
blocks <- metadata %>%
    filter(series %in% cat3_features) %>%
    filter(series %in% colnames(test))
blocks <- blocks[match(col_names_test, blocks$series), ]
blocks <- blocks %>% select(starts_with("block_")) %>% select_if(~ sum(.) > 0)
```

Reads block_g, block_s, block_r, block_l from metadata for Cat3 vars only. Drops empty blocks.

### R Inf Check — Weekly Data Gap (Found + Fixed)

**Problem**: All R notebooks check `is.infinite()` on monthly data only. Weekly data in MIDAS/MIDASML never checked.

**Fix**: Added Inf check loop for `data_weekly` in MIDAS and MIDASML:
```r
for (col in colnames(data_weekly)) {
    if (sum(is.infinite(data_weekly[[col]])) > 0) {
        data_weekly[is.infinite(data_weekly[[col]]), col] <- NA
    }
}
```

---

## SECTION 11 — MIDAS WEEKLY DATA INTEGRATION

### Background

MIDAS and MIDASML were identified as needing weekly data. Original Hopp (2023) notebooks used only monthly data. Our `data_tf_weekly.csv` has 4 weekly series: icsa_weekly, nfci_weekly, dtwexbgs, dtwexm.

### Key Finding: Weekly Data Starts at 1967-01-07

Training starts at 1959-01-01. Weekly data is missing for 1959-1966 (8 years, 32 quarters).

### Decision: Start MIDAS Training at 1967-01-01

Aligns with weekly data availability. Loses 8 years (196→164 quarters). 164 quarters is sufficient for MIDAS estimation (Ghysels 2016: consistent at T=100+).

### get_weekly_series() Function

```r
get_weekly_series <- function(w_data, col, start_date, end_date, n_quarters, weeks_per_q = 13) {
    w_sub <- w_data %>% filter(date >= start_date, date <= end_date) %>% pull(!!col)
    expected_len <- n_quarters * weeks_per_q
    if (n_actual < expected_len) {
        w_sub <- c(rep(NA, expected_len - n_actual), w_sub)  # pad head with NA
    } else if (n_actual > expected_len) {
        w_sub <- w_sub[(n_actual - expected_len + 1):n_actual]  # trim from head
    }
    return(w_sub)
}
```

### Weekly Middleware Data Alignment

Training from 1967-01: 164 quarters × 13 = 2132 expected weeks. Actual: 2139 (diff=7, 0.3%). Q1 1967 has 12 weeks (partial — weekly data starts Jan 7). Padding adds 1 NA. `midas_r` with `na.omit` drops that quarter. Cost: 1/164 quarters. Negligible.

### Dead Weekly Loop Bug (Found + Fixed)

**Original code**: Loop iterated `colnames(train)` which is monthly data. Weekly vars not in monthly columns. `col %in% weekly_vars` always FALSE. Weekly integration was dead code.

**Fix**: Added separate `for (wcol in weekly_vars)` loops:
- Training: extracts weekly series from `data_weekly`, fits `midas_r(y ~ mls(w_vec, 0:12, 13, nealmon))`
- Forecast: extracts weekly data for lagged_data range, uses `forecast(model, newdata=list(w_vec=...))`

### Weekly Nealmon Start Value

**Original**: `start = list(x = c(1, -0.5))` — calibrated for 3-4 lags. With 13 lags (weekly), optimizer may fail.

**Fix**: Weekly uses `start = list(w_vec = c(1, -0.1))`. Monthly/quarterly keep `c(1, -0.5)`.

### MIDAS Frequency Branches

Three branches in model loop:
```r
if (col %in% weekly_vars) {
    mls(x, 0:12, 13, nealmon)      # 13 weeks/quarter
} else if (col %in% quarterly_vars) {
    mls(x, 0:1, 1, nealmon)        # 2 quarters
} else {
    mls(x, 0:3, 3, nealmon)        # 4 months/quarter
}
```

### MIDASML Weekly Integration

Modified `gen_midasml_dataset_v2()`:
- Weekly vars appended to `cov_cols` from `data_weekly`
- Weekly data filled with `na_mean`
- Separate `high_freq_lags_weekly=12` (vs `high_freq_lags_monthly=9`)
- Weekly `mixed_freq_data` uses weekly dates for x.date

### MIDASML Duplicate x.date Bug (Found + Fixed)

**Problem**: `mixed_freq_data(data_weekly[, covariate], as.Date(data_weekly[, "date"]), x.date = as.Date(data_weekly[, "date"]))` — `x.date` passed as both 4th positional arg AND named arg. R throws "formal argument matched by multiple actuals."

**Fix**: Removed named `x.date = ...` argument. Positional arg suffices.

### Weekly Vintage Masking Decision

Weekly vars have `months_lag=0` (always available). Not vintage-masked — correct behavior. m1/m2/m3 predictions for weekly components are identical. The vintage distinction comes from monthly data.

---

## SECTION 12 — DATA LEAKAGE DISCUSSIONS

### `cols_to_drop` Pre-Computation Window

VAR's `cols_to_drop` was computed at `TRAIN_END="2007-12-31"`. User flagged potential data leakage if we widen the window.

**Resolution**: Drop all-NaN columns before computing `cols_to_drop`. This stays within the training window, uses only structural information (which columns have data), and avoids leakage. Same approach as `feature_selection_ensemble.py`.

### VAR ppifis Bug

**Problem**: `ppifis` has 0 non-NaN values in 1959-2007. `mean_fill_dataset` fills with NaN. `dropna(how='any')` kills all rows. `cols_to_drop` becomes empty. VAR produces all NaN.

**Fix**: Drop all-NaN columns before `dropna`:
```python
all_nan_cols = [c for c in pre_train.columns if c != "date" and pre_train[c].notna().sum() == 0]
pre_train = pre_train.drop(columns=all_nan_cols)
```

**Why only VAR affected**: Other notebooks compute `feature_cols` in the rolling loop where training extends to forecast date (≥2017), where ppifis has values.

---

## SECTION 13 — COMPLETE BUG LIST

### Critical Bugs (Would Crash or Produce Invalid Results)

| # | Bug | Found When | Fixed | Where |
|---|---|---|---|---|
| 1 | P6 context-row calendar misalignment | OLS first execution showed all NaN lag_3 columns | ✅ | All 9 non-ARMA Python notebooks |
| 2 | P6 tail(N_LAGS) wrong — includes fc row | Testing context-row alignment | ✅ | Changed to tail(N_LAGS+1).iloc[:-1] |
| 3 | VAR cols_to_drop kills all data (ppifis) | VAR execution all NaN | ✅ | Drop all-NaN cols before dropna |
| 4 | BVAR start=c(1947,2) misaligns dates | R notebook audit | ✅ | Changed to c(1959,2) |
| 5 | MIDAS weekly loop dead code | Deep audit | ✅ | Added for(wcol in weekly_vars) loops |
| 6 | MIDASML x.date duplicate | Deep audit | ✅ | Removed named x.date arg |
| 7 | P2 unscaled HP tuning | Pre-execution audit | ✅ | scaler.fit_transform before CV |

### Medium Bugs

| # | Bug | Fixed | Notes |
|---|---|---|---|
| 8 | LSTM data not filtered to Cat3 | ✅ | Was passing all 296 vars to nowcast_lstm |
| 9 | XGB missing data loading cell | ✅ | Rebuilt entirely |
| 10 | P3 GB zero tuning | ✅ | Added Phase A grid search |
| 11 | P4 XGB partial tuning | ✅ | 4-param grid, TSCV(5) |
| 12 | P5 COVID panel 2020-01-01 | ✅ | Changed to 2020-04-01 |
| 13 | P7 RF grid mismatch | ✅ | Added n_estimators, max_depth |
| 14 | P8 DT fractional params | ✅ | Integer min_samples_leaf |
| 15 | P9 MLP architecture not tuned | ✅ | Added hls grid |
| 16 | R notebooks missing Inf check on weekly | ✅ | Added after weekly data loading |

### Minor/Cosmetic

| # | Bug | Fixed |
|---|---|---|
| 17 | ARMA unused gen_lagged_data import | ✅ |
| 18 | OLS unused split_for_scaler import | ✅ |
| 19 | XGB/GB n_lags changed | ✅ 7→4, 6→4 |
| 20 | VAR PyFlux removed | ✅ statsmodels only |

---

## SECTION 14 — CURRENT STATE SUMMARY

### Predictions Saved (Python — 11/13 models)

`predictions/` directory contains CSV files for: ARMA, OLS, Lasso, Ridge, ElasticNet, DT, RF, XGB, GB, MLP, VAR.

All 0/36 NaN.

**Important**: RF, XGB, GB, MLP, VAR predictions were generated by simplified direct Python scripts, not by the notebook code. The notebooks have grid search code that was bypassed. Reproducibility difference.

### Not Run (6/17 models)

| Model | Reason | Plan |
|---|---|---|
| LSTM | `nowcast_lstm` + `torch` not installed | User runs in Colab |
| DeepVAR | `gluonts` + `mxnet` not installed | User runs in Colab |
| BVAR | R not installed | User runs in Colab |
| DFM | R not installed | User runs in Colab |
| MIDAS | R not installed | User runs in Colab |
| MIDASML | R not installed | User runs in Colab |

### File Inventory (data/ folder)

| File | Status |
|---|---|
| `helpers.py` | ✅ All functions, P1+P6 fixes |
| `helpers.R` | ✅ gen_lagged_data, R formula correct (1-indexed) |
| `variable_lists.json` | ✅ Cat2=4, Cat3=53, COVID=3 |
| `variable_selection_framework.md` | ✅ Complete reasoning |
| `variable_rankings_by_rule.txt` | ✅ Full rankings |
| `evaluate.py` | ✅ DM test wired, COVID panel fixed |
| `tuning_policy.md` | ✅ MLP alpha=1e-2, LSTM dropout=0.3 |
| `scaling_policy.md` | ✅ Unchanged from HANDOFF |

### File Inventory (model_notebooks/)

All 17 notebooks present. 13 Python, 4 R. All pass structural checks (no empty cells, no unbalanced brackets).

### Root Directory

Clean: `LICENSE`, `README.md`, `STATUS.md`, `predictions/`, `data/`, `model_notebooks/`, `methodologies/` (original, read-only), `turkey_data/` (untouched).

---

## SECTION 15 — KNOWN UNCERTAINTIES FOR COLAB

### Runtime Risks

| Risk | Model | Mitigation |
|---|---|---|
| mfbvar convergence with 57 vars | BVAR | Try smaller subset if fails |
| Nealmon c(1,-0.1) convergence for 13 lags | MIDAS | Unknown — cannot test on this machine |
| `midas_r` with NA in mls input | MIDAS | na.omit drops that quarter |
| `mixed_freq_data` with weekly dates | MIDASML | x.date duplicate fixed, but runtime untested |
| DFM EM with 56 vars > recommended 20 | DFM | Increase max_iter or reduce vars |
| Colab session timeout | MIDAS (~3-5 hours) | Run overnight, save intermediate |

### Path Adjustments Needed for Colab

All 4 R notebooks use `../data/` paths. Need to change to Colab's file structure:
- `source("../data/helpers.R")` → `source("helpers.R")`
- `read_csv("../data/data_tf_monthly.csv")` → `read_csv("data_tf_monthly.csv")`
- `read_csv("../data/data_tf_weekly.csv")` → `read_csv("data_tf_weekly.csv")`
- `dir.create("../predictions")` → `dir.create("predictions")`

### Required R Packages

```r
install.packages(c("tidyverse", "nowcastDFM", "mfbvar", "midasr", "midasml", "imputeTS", "Rmisc"))
```

### Files to Upload to Colab

- `data/data_tf_monthly.csv`
- `data/data_tf_weekly.csv`
- `data/meta_data.csv`
- `data/helpers.R`
- `model_notebooks/model_bvar.ipynb`
- `model_notebooks/model_dfm.ipynb`
- `model_notebooks/model_midas.ipynb`
- `model_notebooks/model_midasml.ipynb`

---

## SECTION 16 — DESIGN DECISIONS LOG

| Decision | Date | Rationale |
|---|---|---|
| Lag=0 convention, shift forecast_date | Early | Proved lag=-1 over-masks. Literature alignment. |
| Cat3=53 vars from L95∪E95∪R95∪S100 | Mid | Rule-based, not arbitrary top-35 |
| Cat2=4 vars from top-2 RF + top-3 Stab | Mid | DoF constraint for unpenalized models |
| n_lags 7→4 for XGB, 6→4 for GB | Late | Features/obs ratio: 2.18→1.37, 1.91→1.37 |
| PyFlux→statsmodels for VAR | Late | Reproducibility, modern Python compatibility |
| bootstrap=True for RF | Late | Breiman (2001) default |
| BVAR in Cat3 (53 vars) not Cat2 (4 vars) | Late | Bańbura 2010 shows 130-vars optimal for BVAR |
| MIDAS train start 1967 | Late | Weekly data availability alignment |
| Drop LSTM-R (keep Python only) | Early | Duplicate — same library |
| Drop AR(1), keep ARMA as benchmark | Early | ARMA auto-selects (p,q), more realistic baseline |
| No random walk benchmark | Early | User decision |
| OLS stays in Cat2 (4 vars, same as VAR) | Mid | DoF constraint applies. Cat2=4 is safe for OLS. |
| Weekly data only in MIDAS/MIDASML | Late | These are the only mixed-frequency models |
| Data leakage: drop all-NaN cols before dropna | Late | Same approach as feature_selection_ensemble.py |
| P6 fix: tail(N_LAGS+1).iloc[:-1] | Late | Calendar alignment of lag columns |

---

## SECTION 17 — KEY ACADEMIC REFERENCES IN OUR DESIGN

| Paper | Used For |
|---|---|
| Meinshausen-Bühlmann (2010) | Stability selection threshold (≥50%) |
| Bańbura et al. (2010) | Large BVAR with Minnesota prior (130 vars) |
| Medeiros et al. (2021) | ML models on full FRED-MD (122 vars) |
| Bok et al. (2018) | gdpc1 lag=2, 4-block DFM, publication lag convention |
| Coulombe et al. (2022) | RF on full FRED-MD, elastic net preferred over Lasso |
| Sims (1980) | T/3 rule for unpenalized VAR/OLS |
| Stock-Watson (2002) | Large-N DFM (132 vars), factor extraction |
| Giannone-Reichlin-Small (2008) | Real-time nowcasting, pub-lag-only masking |
| Ghysels et al. (2004) | MIDAS mixed-frequency regression |
| Breiman (2001) | Random forest, bootstrap bagging |

---

## SECTION 18 — CONVENTIONS (DO NOT VIOLATE)

1. `gdpc1` lag=2. NOT 1, NOT 4. Bok et al. convention.
2. `lag=0` for all vintages. Shift `forecast_date`, not the `lag` parameter.
3. gen_lagged_data BEFORE mean_fill_dataset. Reversing destroys ragged-edge.
4. Scaler/mean-fill fit on TRAINING FOLD ONLY. Never include test slice.
5. TimeSeriesSplit. NEVER random KFold.
6. COVID dummies NEVER scaled. `split_for_scaler()` separates them.
7. Do NOT regenerate stale `data_tf.csv` or `data_raw.csv`.
8. Do NOT re-introduce per-variable override dicts.
9. Do NOT add a 2008 dummy.
10. Do NOT use LOCF/AR(1) imputation for ragged edges.
11. Train/val/test: 1959-2007 / 2008-2016 / 2017-Q1 to 2025-Q4.
12. COVID panel: 2020-Q2 to 2021-Q4 (starts 2020-04-01, NOT 2020-01-01).
13. Python: 0-indexed gen_lagged_data. R: 1-indexed. Both produce equivalent masking.
14. Weekly data: months_lag=0, always available, not vintage-masked.
15. MIDAS/MIDASML: train from 1967-01-01 (weekly data alignment).

---

## END OF HANDOFF2

*Document length: ~600 lines. All decisions, bugs, fixes, and reasoning captured. Refer to HANDOFF.md for data pipeline details and policy documents.*
