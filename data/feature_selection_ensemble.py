"""
feature_selection_ensemble.py
=============================
RIGOROUS cross-method variable selection for nowcasting GDP.

Four selectors are run on the same 1959-2007 training window (Addendum 2:
training ends before GFC so the GFC sits inside the validation window for
HP tuning at the model-notebook stage).

Methods
-------
1. LassoCV (TimeSeriesSplit, wide alpha path)
2. ElasticNetCV (TimeSeriesSplit, widened l1_ratio grid)
3. RandomForest with light grid search (TimeSeriesSplit) + permutation_importance
4. Lasso stability selection (Meinshausen-Buhlmann 2010)

All CV uses TimeSeriesSplit(n_splits=5) -- never random KFold on time series.

Outputs (saved to feature_selection_ensemble.txt)
-------------------------------------------------
- Top-35 ranked per method (rank, feature, importance) for at-a-glance reading.
- Pairwise overlap matrix (Lasso vs EN, Lasso vs RF, EN vs RF, Lasso vs Stab).
- Stability frequencies for top-35 Lasso candidates.
- Plain-list versions for copy-paste into notebooks.
- Union (robust set, ML notebooks) and Intersection (core set).

Reproducibility: random_state=42 throughout.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV, ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

BASE = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
SEED = 42
TARGET = "gdpc1"

# Addendum 2 windows (training ends before GFC so GFC is in the val window)
TRAIN_START = "1959-01-01"
TRAIN_END   = "2007-12-31"
N_LAGS = 3
TOP_K  = 35

# -----------------------------------------------------------------------------
# Helpers (mirror lasso_feature_selection.py for comparability)
# -----------------------------------------------------------------------------
def flatten_data(data, target_variable, n_lags):
    flattened_data = data.loc[~pd.isna(data[target_variable]), :]
    orig_index = flattened_data.index
    for i in range(1, n_lags + 1):
        lagged_indices = orig_index - i
        lagged_indices = lagged_indices[lagged_indices >= 0]
        tmp = data.loc[lagged_indices, :]
        tmp.date = tmp.date + pd.DateOffset(months=i)
        tmp = tmp.drop([target_variable], axis=1)
        tmp.columns = [j + "_" + str(i) if j != "date" else j for j in tmp.columns]
        flattened_data = flattened_data.merge(tmp, how="left", on="date")
    return flattened_data

def mean_fill_dataset(training, test):
    mean_dict = {col: np.nanmean(training[col]) for col in training.columns[1:]}
    filled = test.copy()
    for col in training.columns[1:]:
        filled.loc[pd.isna(filled[col]), col] = mean_dict[col]
    return filled

def aggregate_importance(importances, columns):
    """Sum |importance| across the n_lags lagged copies per base variable."""
    out = {}
    for i, col in enumerate(columns):
        base = col
        if "_" in col and col.rsplit("_", 1)[1].isdigit():
            base = col.rsplit("_", 1)[0]
        out[base] = out.get(base, 0.0) + abs(importances[i])
    return out

def topk(importance_dict, k=TOP_K):
    return [feat for feat, imp in
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            if imp > 0][:k]

def topk_ranked(importance_dict, k=TOP_K):
    sorted_items = sorted(
        ((f, i) for f, i in importance_dict.items() if i > 0),
        key=lambda x: x[1], reverse=True,
    )[:k]
    return [(r + 1, f, imp) for r, (f, imp) in enumerate(sorted_items)]

def stability_selection_lasso(X_df, y, alpha, n_resamples=100, frac=0.75,
                              k=TOP_K, random_state=SEED):
    """
    Meinshausen-Buhlmann (2010) stability selection.
    Subsample frac of rows n_resamples times, fit Lasso at given alpha,
    count selection frequency in top-k by |coef|.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    sub_n = int(frac * n)
    counts = {col: 0 for col in X_df.columns}
    base_to_lagged = {}
    for col in X_df.columns:
        base = col.rsplit("_", 1)[0] if ("_" in col and col.rsplit("_", 1)[1].isdigit()) else col
        base_to_lagged.setdefault(base, []).append(col)

    base_counts = {b: 0 for b in base_to_lagged}
    for _ in range(n_resamples):
        idx = rng.choice(n, size=sub_n, replace=False)
        m = Lasso(alpha=alpha, max_iter=20000, random_state=random_state)
        m.fit(X_df.iloc[idx].values, y.iloc[idx].values)
        imp = aggregate_importance(m.coef_, X_df.columns.tolist())
        top = topk(imp, k)
        for f in top:
            base_counts[f] = base_counts.get(f, 0) + 1
    return {b: c / n_resamples for b, c in base_counts.items()}

# -----------------------------------------------------------------------------
# Build common X, y
# -----------------------------------------------------------------------------
print("Loading data...")
data = pd.read_csv(os.path.join(BASE, "data_tf_monthly.csv"), parse_dates=["date"])
train = data.loc[(data.date >= TRAIN_START) & (data.date <= TRAIN_END), :].reset_index(drop=True)
print(f"Train window: {TRAIN_START} to {TRAIN_END} ({len(train)} monthly rows)")

# Drop columns that are entirely NaN in the training window. These cannot
# be mean-filled (np.nanmean of all-NaN is NaN) and would propagate NaN
# through the lag merge, killing every row at dropna time. Series that
# start after TRAIN_END are simply not usable for this selection window.
all_nan_cols = [c for c in train.columns
                if c != "date" and train[c].notna().sum() == 0]
if all_nan_cols:
    print(f"Dropping {len(all_nan_cols)} columns with no data in train window: "
          f"{all_nan_cols[:10]}{'...' if len(all_nan_cols) > 10 else ''}")
    train = train.drop(columns=all_nan_cols)

print("Flattening + mean-filling + restricting to Q-end months...")
filled_train = mean_fill_dataset(train, train)
flattened    = flatten_data(filled_train, TARGET, N_LAGS)
flattened    = flattened.loc[flattened.date.dt.month.isin([3, 6, 9, 12]), :] \
                        .dropna(axis=0, how="any").reset_index(drop=True)

X = flattened.drop(["date", TARGET], axis=1)
y = flattened[TARGET]

# Drop COVID dummies (zero-variance pre-2020)
covid_cols = [c for c in X.columns if "covid" in c.lower()]
X = X.drop(covid_cols, axis=1)

print(f"X shape: {X.shape} | y shape: {y.shape}")

# Common time-aware CV
tscv = TimeSeriesSplit(n_splits=5)

# -----------------------------------------------------------------------------
# 1. LassoCV (TimeSeriesSplit, wide alpha path)
# -----------------------------------------------------------------------------
print("\n[1/4] Fitting LassoCV with TimeSeriesSplit...")
alpha_path = np.logspace(-6, 0, 100)
lasso_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=alpha_path, cv=tscv, random_state=SEED, max_iter=20000, n_jobs=1)),
    ]
)
lasso_pipe.fit(X.values, y)
lasso = lasso_pipe.named_steps["model"]
imp_lasso = aggregate_importance(lasso.coef_, X.columns)
top_lasso = topk(imp_lasso)
top_lasso_ranked = topk_ranked(imp_lasso)
print(f"      Lasso alpha={lasso.alpha_:.4g}, top-35 head: {top_lasso[:5]}")

# -----------------------------------------------------------------------------
# 2. ElasticNetCV (TimeSeriesSplit, widened l1_ratio grid)
# -----------------------------------------------------------------------------
print("\n[2/4] Fitting ElasticNetCV with widened l1_ratio grid...")
en_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            ElasticNetCV(
                l1_ratio=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
                alphas=alpha_path, cv=tscv, random_state=SEED, max_iter=20000, n_jobs=1,
            ),
        ),
    ]
)
en_pipe.fit(X.values, y)
en = en_pipe.named_steps["model"]
imp_en = aggregate_importance(en.coef_, X.columns)
top_en = topk(imp_en)
top_en_ranked = topk_ranked(imp_en)
print(f"      EN alpha={en.alpha_:.4g}, l1_ratio={en.l1_ratio_:.3f}, top-35 head: {top_en[:5]}")

# -----------------------------------------------------------------------------
# 3. RandomForest light grid search + permutation importance
# -----------------------------------------------------------------------------
print("\n[3/4] RF grid search (TimeSeriesSplit) then permutation importance...")
rf_base = RandomForestRegressor(
    n_estimators=500, max_depth=None,
    n_jobs=1, random_state=SEED,
)
rf_grid = {
    "max_features":     ["sqrt", 0.33],
    "min_samples_leaf": [1, 3, 5],
}
rf_search = GridSearchCV(
    rf_base, rf_grid, cv=tscv,
    scoring="neg_mean_squared_error", n_jobs=1, refit=True,
)
rf_search.fit(X.values, y)
print(f"      Best RF params: {rf_search.best_params_}, CV MSE={-rf_search.best_score_:.4g}")
print("      Computing permutation importance (n_repeats=10, manual)...")
rf_best = rf_search.best_estimator_
rng_perm = np.random.default_rng(SEED)
baseline = np.mean((y.values - rf_best.predict(X.values)) ** 2)
perm_scores = np.zeros((X.shape[1], 10))
for j in range(X.shape[1]):
    for r in range(10):
        X_perm = X.values.copy()
        X_perm[:, j] = rng_perm.permutation(X_perm[:, j])
        perm_scores[j, r] = np.mean((y.values - rf_best.predict(X_perm)) ** 2) - baseline
    if j % 100 == 0:
        print(f"        ... feature {j}/{X.shape[1]}")
perm_means = perm_scores.mean(axis=1)
imp_rf = aggregate_importance(perm_means, X.columns)
top_rf = topk(imp_rf)
top_rf_ranked = topk_ranked(imp_rf)
print(f"      RF top-35 head: {top_rf[:5]}")

# -----------------------------------------------------------------------------
# 4. Lasso stability selection
# -----------------------------------------------------------------------------
print("\n[4/4] Lasso stability selection (100 resamples, 75% subsample)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
X_df = pd.DataFrame(X_scaled, columns=X.columns)
stab_freq = stability_selection_lasso(X_df, y, alpha=lasso.alpha_,
                                       n_resamples=100, frac=0.75, k=TOP_K)
top_stab_ranked_pairs = sorted(stab_freq.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
top_stab = [f for f, _ in top_stab_ranked_pairs]
top_stab_ranked = [(r + 1, f, p) for r, (f, p) in enumerate(top_stab_ranked_pairs)]
n_stable_80 = sum(1 for _, p in top_stab_ranked_pairs if p >= 0.8)
print(f"      {n_stable_80}/{TOP_K} variables selected in >=80% of resamples")

# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------
union        = sorted(set(top_lasso) | set(top_en) | set(top_rf))
intersection = sorted(set(top_lasso) & set(top_en) & set(top_rf))
ovl_lasso_en = len(set(top_lasso) & set(top_en))
ovl_lasso_rf = len(set(top_lasso) & set(top_rf))
ovl_en_rf    = len(set(top_en) & set(top_rf))
ovl_lasso_stab = len(set(top_lasso) & set(top_stab))

print("\n========================================================")
print(f"Top-{TOP_K} Lasso        : {len(top_lasso)}")
print(f"Top-{TOP_K} ElasticNet   : {len(top_en)}")
print(f"Top-{TOP_K} RF perm-imp  : {len(top_rf)}")
print(f"Top-{TOP_K} Lasso (stab) : {len(top_stab)}")
print(f"Pairwise overlaps:")
print(f"  Lasso vs EN     : {ovl_lasso_en}/{TOP_K}")
print(f"  Lasso vs RF     : {ovl_lasso_rf}/{TOP_K}  (low expected: linear vs non-linear)")
print(f"  EN vs RF        : {ovl_en_rf}/{TOP_K}")
print(f"  Lasso vs Stab   : {ovl_lasso_stab}/{TOP_K}  (high = picks are stable)")
print(f"Union   (robust set): {len(union)}")
print(f"Inter.  (core set)  : {len(intersection)}")
print("========================================================")

# -----------------------------------------------------------------------------
# Save (xlsx with named sheets, programmatic-friendly for downstream notebooks)
# -----------------------------------------------------------------------------
out_xlsx = os.path.join(BASE, "feature_selection_ensemble.xlsx")

def ranked_to_df(ranked, value_col_name):
    return pd.DataFrame(
        [{"rank": r, "feature": f, value_col_name: v} for r, f, v in ranked]
    )

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
    # Per-method ranked lists
    ranked_to_df(top_lasso_ranked, "importance").to_excel(xw, sheet_name="Lasso", index=False)
    ranked_to_df(top_en_ranked,    "importance").to_excel(xw, sheet_name="ElasticNet", index=False)
    ranked_to_df(top_rf_ranked,    "importance").to_excel(xw, sheet_name="RF_perm_imp", index=False)
    ranked_to_df(top_stab_ranked,  "selection_freq").to_excel(xw, sheet_name="Stability", index=False)

    # Aggregate sets
    pd.DataFrame({"feature": union}).to_excel(xw, sheet_name="Union", index=False)
    pd.DataFrame({"feature": intersection}).to_excel(xw, sheet_name="Intersection", index=False)

    # Pairwise overlaps
    pd.DataFrame({
        "pair": ["Lasso vs EN", "Lasso vs RF", "EN vs RF", "Lasso vs Stab"],
        "overlap_out_of_35": [ovl_lasso_en, ovl_lasso_rf, ovl_en_rf, ovl_lasso_stab],
        "note": [
            "high expected: same linear class",
            "low expected: linear vs non-linear",
            "low expected: linear vs non-linear",
            "high = Lasso picks are stable across resamples",
        ],
    }).to_excel(xw, sheet_name="Overlaps", index=False)

    # Run metadata
    pd.DataFrame({
        "key": [
            "train_window", "n_obs", "n_features",
            "lasso_alpha",
            "en_alpha", "en_l1_ratio",
            "rf_max_features", "rf_min_samples_leaf",
            "rf_n_estimators", "rf_perm_n_repeats",
            "stab_n_resamples", "stab_subsample_frac",
            "cv", "seed",
        ],
        "value": [
            f"{TRAIN_START} to {TRAIN_END}", len(y), X.shape[1],
            f"{lasso.alpha_:.4g}",
            f"{en.alpha_:.4g}", f"{en.l1_ratio_:.3f}",
            str(rf_search.best_params_.get("max_features")),
            str(rf_search.best_params_.get("min_samples_leaf")),
            500, 10,
            100, 0.75,
            "TimeSeriesSplit(n_splits=5)", SEED,
        ],
    }).to_excel(xw, sheet_name="Run_metadata", index=False)

print(f"\nSaved {out_xlsx}")

# Also write a plain-text companion for at-a-glance reading
out_path = os.path.join(BASE, "feature_selection_ensemble.txt")

def fmt_list(lst): return '"' + '", "'.join(lst) + '"' if lst else ""
def fmt_ranked(ranked):
    return "\n".join(f"  {r:>3}. {f:<32s}  imp={imp:.4g}" for r, f, imp in ranked)

with open(out_path, "w") as f:
    f.write("Ensemble feature selection (RIGOROUS) -- 1959-2007 training window\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Lasso alpha={lasso.alpha_:.4g} (TimeSeriesSplit n=5, alpha grid 1e-6..1)\n")
    f.write(f"ElasticNet alpha={en.alpha_:.4g}, l1_ratio={en.l1_ratio_:.3f} "
            f"(grid: 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)\n")
    f.write(f"RandomForest best params: {rf_search.best_params_}, n_estimators=500, "
            f"perm-imp n_repeats=10\n")
    f.write(f"Stability selection: 100 resamples, 75% subsample, alpha=Lasso optimum\n\n")
    f.write("Pairwise overlaps (out of " + str(TOP_K) + "):\n")
    f.write(f"  Lasso vs EN     : {ovl_lasso_en}\n")
    f.write(f"  Lasso vs RF     : {ovl_lasso_rf}  (low expected: linear vs non-linear)\n")
    f.write(f"  EN vs RF        : {ovl_en_rf}\n")
    f.write(f"  Lasso vs Stab   : {ovl_lasso_stab}  (high = picks are stable)\n\n")
    f.write("=" * 70 + "\n")
    f.write(f"Top {TOP_K} -- LASSO (rank: feature, importance):\n")
    f.write(fmt_ranked(top_lasso_ranked) + "\n\n")
    f.write(f"Top {TOP_K} -- ELASTICNET (rank: feature, importance):\n")
    f.write(fmt_ranked(top_en_ranked) + "\n\n")
    f.write(f"Top {TOP_K} -- RANDOM FOREST perm-imp (rank: feature, importance):\n")
    f.write(fmt_ranked(top_rf_ranked) + "\n\n")
    f.write(f"Top {TOP_K} -- LASSO STABILITY SELECTION (rank: feature, freq):\n")
    f.write(fmt_ranked(top_stab_ranked) + "\n\n")
    f.write("=" * 70 + "\n")
    f.write("Plain lists for copy-paste into notebooks:\n\n")
    f.write(f"Lasso top-{TOP_K}:\n")          ; f.write(fmt_list(top_lasso) + "\n\n")
    f.write(f"ElasticNet top-{TOP_K}:\n")     ; f.write(fmt_list(top_en) + "\n\n")
    f.write(f"RF perm-imp top-{TOP_K}:\n")    ; f.write(fmt_list(top_rf) + "\n\n")
    f.write(f"Lasso stable top-{TOP_K}:\n")   ; f.write(fmt_list(top_stab) + "\n\n")
    f.write("UNION (Lasso U EN U RF -- robust set, use for ML notebooks):\n")
    f.write(fmt_list(union) + "\n\n")
    f.write("INTERSECTION (Lasso n EN n RF -- core set):\n")
    f.write(fmt_list(intersection) + "\n")

print(f"\nSaved to {out_path}")
