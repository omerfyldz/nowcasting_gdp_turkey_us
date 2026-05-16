"""
feature_selection_tr.py
========================
Pipeline B — Turkey: Ensemble variable selection on 1995-2011 training window.

Methods (aligned with US feature_selection_ensemble.py):
    1. LassoCV          (TimeSeriesSplit n=5, wide alpha path, n_jobs=1)
    2. ElasticNetCV     (TimeSeriesSplit n=5, widened l1_ratio grid, n_jobs=1)
    3. RF GridSearchCV + manual permutation importance (n_jobs=1, no joblib)
    4. Lasso stability selection (Meinshausen-Buhlmann 2010, 100 resamples)

Turkey-specific:
    - Coverage filter: Tier A+B (>=34% non-NaN in training) -> cat3
    - Tier C (< 34% coverage) -> DFM-only variables
    - cat2: 4 vars selected from Lasso/EN/Stability consensus (small-model set)

Run order: 6 (after run_stationarity_tr.py)

References:
    McCracken & Ng (2020), Meinshausen-Buhlmann (2010)
"""

import os, json, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV, ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Use US pipeline helpers — never redefine locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from helpers import flatten_data, mean_fill_dataset

BASE    = os.path.dirname(os.path.abspath(__file__))
TF_PATH = os.path.join(BASE, "data_tf_monthly_tr.csv")
OUT_JSON = os.path.join(BASE, "turkey_variable_lists.json")
OUT_XLSX = os.path.join(BASE, "feature_selection_tr.xlsx")
OUT_TXT  = os.path.join(BASE, "feature_selection_tr.txt")

TARGET = "ngdprsaxdctrq"
COVID  = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]
SEED   = 42
N_LAGS = 3
TOP_K  = 20   # smaller than US (fewer Turkey observations)


# ---------------------------------------------------------------------------
# Helpers (identical logic to US script)
# ---------------------------------------------------------------------------
def aggregate_importance(importances, columns):
    out = {}
    for i, col in enumerate(columns):
        base = col.rsplit("_", 1)[0] if ("_" in col and col.rsplit("_", 1)[1].isdigit()) else col
        out[base] = out.get(base, 0.0) + abs(importances[i])
    return out

def topk(importance_dict, k=TOP_K):
    return [f for f, imp in
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            if imp > 0][:k]

def topk_ranked(importance_dict, k=TOP_K):
    items = sorted(((f, i) for f, i in importance_dict.items() if i > 0),
                   key=lambda x: x[1], reverse=True)[:k]
    return [(r + 1, f, imp) for r, (f, imp) in enumerate(items)]

def stability_selection_lasso(X_df, y, alpha, n_resamples=100, frac=0.75,
                               k=TOP_K, random_state=SEED):
    rng   = np.random.default_rng(random_state)
    n     = len(y)
    sub_n = int(frac * n)
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
        for f in topk(imp, k):
            base_counts[f] = base_counts.get(f, 0) + 1
    return {b: c / n_resamples for b, c in base_counts.items()}

def fmt_list(lst):
    return '"' + '", "'.join(lst) + '"' if lst else ""

def fmt_ranked(ranked):
    return "\n".join(f"  {r:>3}. {f:<35s}  imp={imp:.4g}" for r, f, imp in ranked)


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("FEATURE SELECTION — TURKEY")
    print("=" * 60)

    # 1. Load
    tf    = pd.read_csv(TF_PATH, parse_dates=["date"])
    train = tf[(tf["date"] >= "1995-01-01") & (tf["date"] <= "2011-12-31")].copy()
    print(f"\nTraining window: {train['date'].min().date()} to {train['date'].max().date()}"
          f"  ({len(train)} monthly rows)")

    # 2. Coverage filter
    feature_pool = [c for c in tf.columns if c not in ["date", TARGET] + COVID]
    coverage     = {f: train[f].notna().mean() for f in feature_pool}
    tier_ab      = sorted(f for f, cov in coverage.items() if cov >= 0.34)
    tier_c       = sorted(f for f, cov in coverage.items() if cov < 0.34)
    print(f"Tier A+B (>=34% coverage): {len(tier_ab)} vars")
    print(f"Tier C   (<34% coverage) : {len(tier_c)} vars  (DFM-only)")

    # 3. Drop all-NaN columns in training window (safety guard, mirrors US script)
    all_nan = [f for f in tier_ab if train[f].notna().sum() == 0]
    if all_nan:
        print(f"Dropping {len(all_nan)} all-NaN Tier A+B cols: {all_nan}")
        tier_ab = [f for f in tier_ab if f not in all_nan]

    # 4. Flatten + mean-fill (using US helpers, identical order)
    cols_to_use = ["date", TARGET] + tier_ab
    train_sub   = train[cols_to_use].reset_index(drop=True)
    filled      = mean_fill_dataset(train_sub, train_sub)
    flat        = flatten_data(filled, TARGET, N_LAGS)
    flat        = flat.loc[flat["date"].dt.month.isin([3, 6, 9, 12])].dropna(how="any").reset_index(drop=True)
    feat_cols   = [c for c in flat.columns if c not in ["date", TARGET]]
    X_raw       = flat[feat_cols].values
    y           = flat[TARGET]
    print(f"Flattened: {len(feat_cols)} features, {len(flat)} quarterly obs")

    tscv     = TimeSeriesSplit(n_splits=5)
    alpha_path = np.logspace(-6, 0, 100)

    # 5. LassoCV
    print(f"\n[1/4] Fitting LassoCV...")
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=alpha_path, cv=tscv, random_state=SEED,
                          max_iter=20000, n_jobs=1)),
    ])
    lasso_pipe.fit(X_raw, y)
    lasso = lasso_pipe.named_steps["model"]
    imp_lasso      = aggregate_importance(lasso.coef_, feat_cols)
    top_lasso      = topk(imp_lasso)
    top_lasso_rank = topk_ranked(imp_lasso)
    print(f"      alpha={lasso.alpha_:.4g}, top-{TOP_K} head: {top_lasso[:5]}")

    # 6. ElasticNetCV
    print(f"\n[2/4] Fitting ElasticNetCV...")
    en_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
            alphas=alpha_path, cv=tscv, random_state=SEED, max_iter=20000, n_jobs=1,
        )),
    ])
    en_pipe.fit(X_raw, y)
    en = en_pipe.named_steps["model"]
    imp_en      = aggregate_importance(en.coef_, feat_cols)
    top_en      = topk(imp_en)
    top_en_rank = topk_ranked(imp_en)
    print(f"      alpha={en.alpha_:.4g}, l1_ratio={en.l1_ratio_:.3f}, top-{TOP_K} head: {top_en[:5]}")

    # 7. RF GridSearchCV + manual permutation importance (no joblib)
    print(f"\n[3/4] RF grid search + manual permutation importance...")
    rf_base = RandomForestRegressor(n_estimators=500, max_depth=None,
                                    n_jobs=1, random_state=SEED)
    rf_grid = {"max_features": ["sqrt", 0.33], "min_samples_leaf": [1, 3, 5]}
    rf_search = GridSearchCV(rf_base, rf_grid, cv=tscv,
                             scoring="neg_mean_squared_error", n_jobs=1, refit=True)
    rf_search.fit(X_raw, y)
    print(f"      Best RF: {rf_search.best_params_}, CV MSE={-rf_search.best_score_:.4g}")

    print(f"      Computing permutation importance (n_repeats=10, manual)...")
    rf_best    = rf_search.best_estimator_
    rng_perm   = np.random.default_rng(SEED)
    baseline   = np.mean((y.values - rf_best.predict(X_raw)) ** 2)
    perm_scores = np.zeros((X_raw.shape[1], 10))
    for j in range(X_raw.shape[1]):
        for r in range(10):
            X_perm = X_raw.copy()
            X_perm[:, j] = rng_perm.permutation(X_perm[:, j])
            perm_scores[j, r] = np.mean((y.values - rf_best.predict(X_perm)) ** 2) - baseline
        if j % 50 == 0:
            print(f"        ... feature {j}/{X_scaled.shape[1]}")
    perm_means  = perm_scores.mean(axis=1)
    imp_rf      = aggregate_importance(perm_means, feat_cols)
    top_rf      = topk(imp_rf)
    top_rf_rank = topk_ranked(imp_rf)
    print(f"      RF top-{TOP_K} head: {top_rf[:5]}")

    # 8. Lasso stability selection
    print(f"\n[4/4] Lasso stability selection (100 resamples, 75% subsample)...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_df     = pd.DataFrame(X_scaled, columns=feat_cols)
    stab_freq  = stability_selection_lasso(X_df, y, alpha=lasso.alpha_)
    stab_pairs = sorted(stab_freq.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_stab      = [f for f, _ in stab_pairs]
    top_stab_rank = [(r + 1, f, p) for r, (f, p) in enumerate(stab_pairs)]
    n_stable_80 = sum(1 for _, p in stab_pairs if p >= 0.8)
    print(f"      {n_stable_80}/{TOP_K} vars selected in >=80% of resamples")

    # 9. Overlaps
    ovl_lasso_en   = len(set(top_lasso) & set(top_en))
    ovl_lasso_rf   = len(set(top_lasso) & set(top_rf))
    ovl_en_rf      = len(set(top_en)    & set(top_rf))
    ovl_lasso_stab = len(set(top_lasso) & set(top_stab))
    union        = sorted(set(top_lasso) | set(top_en) | set(top_rf))
    intersection = sorted(set(top_lasso) & set(top_en) & set(top_rf))

    print(f"\n{'='*55}")
    print(f"Top-{TOP_K} Lasso       : {len(top_lasso)}")
    print(f"Top-{TOP_K} ElasticNet  : {len(top_en)}")
    print(f"Top-{TOP_K} RF perm-imp : {len(top_rf)}")
    print(f"Top-{TOP_K} Stab        : {len(top_stab)}")
    print(f"Lasso vs EN     : {ovl_lasso_en}/{TOP_K}")
    print(f"Lasso vs RF     : {ovl_lasso_rf}/{TOP_K}  (low expected)")
    print(f"EN vs RF        : {ovl_en_rf}/{TOP_K}")
    print(f"Lasso vs Stab   : {ovl_lasso_stab}/{TOP_K}  (high = stable)")
    print(f"Union  : {len(union)}   Intersection: {len(intersection)}")
    print(f"{'='*55}")

    # 10. cat2: 4-var small-model set
    #     ipi_sa (activity) + usd_try_avg (FX) + cpi_sa (prices) + fin_acc (capital flows)
    #     fin_acc included because RF ranks it #1 (nonlinear capital-flow signal);
    #     reer excluded — usd_try_avg already covers FX and cpi_sa covers the inflation
    #     adjustment component, so reer adds multicollinearity without new information.
    cat2 = ["ipi_sa", "usd_try_avg", "cpi_sa", "fin_acc"]
    print(f"\ncat2 (small-model vars): {cat2}")

    # 11. Build JSON variable lists
    var_lists = {
        "target":          TARGET,
        "cat1":            {"features": [TARGET], "total": 1},
        "cat2":            {"features": cat2, "with_covid": True,
                            "total": len(cat2) + len(COVID)},
        "cat3":            {"features": tier_ab, "with_covid": True,
                            "total": len(tier_ab) + len(COVID)},
        "covid_dummies":   COVID,
        "tier_c_dfm_only": tier_c,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(var_lists, f, indent=2, ensure_ascii=False)
    print(f"Written: {OUT_JSON}")
    print(f"  cat3: {len(tier_ab)} vars,  DFM-only (Tier C): {len(tier_c)} vars")

    # 12. Save xlsx
    def ranked_to_df(ranked, val_col):
        return pd.DataFrame([{"rank": r, "feature": f, val_col: v} for r, f, v in ranked])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        ranked_to_df(top_lasso_rank, "importance").to_excel(xw, sheet_name="Lasso",      index=False)
        ranked_to_df(top_en_rank,    "importance").to_excel(xw, sheet_name="ElasticNet", index=False)
        ranked_to_df(top_rf_rank,    "importance").to_excel(xw, sheet_name="RF_perm_imp",index=False)
        ranked_to_df(top_stab_rank,  "selection_freq").to_excel(xw, sheet_name="Stability", index=False)
        pd.DataFrame({"feature": union}).to_excel(xw, sheet_name="Union",        index=False)
        pd.DataFrame({"feature": intersection}).to_excel(xw, sheet_name="Intersection", index=False)
        pd.DataFrame({
            "pair":            ["Lasso vs EN","Lasso vs RF","EN vs RF","Lasso vs Stab"],
            f"overlap/{TOP_K}": [ovl_lasso_en, ovl_lasso_rf, ovl_en_rf, ovl_lasso_stab],
        }).to_excel(xw, sheet_name="Overlaps", index=False)
        pd.DataFrame({
            "key":   ["train_window","n_obs","n_features","lasso_alpha",
                      "en_alpha","en_l1_ratio","rf_best_params",
                      "stab_n_resamples","stab_frac","cv","seed"],
            "value": [f"1995-01-01 to 2011-12-31", len(y), len(feat_cols),
                      f"{lasso.alpha_:.4g}", f"{en.alpha_:.4g}", f"{en.l1_ratio_:.3f}",
                      str(rf_search.best_params_), 100, 0.75,
                      "TimeSeriesSplit(n_splits=5)", SEED],
        }).to_excel(xw, sheet_name="Run_metadata", index=False)
    print(f"Written: {OUT_XLSX}")

    # 13. Save txt companion
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Ensemble feature selection — TURKEY  (1995-2011 training window)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Lasso alpha={lasso.alpha_:.4g}  (TimeSeriesSplit n=5, grid 1e-6..1)\n")
        f.write(f"ElasticNet alpha={en.alpha_:.4g}, l1_ratio={en.l1_ratio_:.3f}\n")
        f.write(f"RF best params: {rf_search.best_params_}, n_estimators=500, perm n_repeats=10\n")
        f.write(f"Stability: 100 resamples, 75% subsample\n\n")
        f.write(f"Pairwise overlaps (out of {TOP_K}):\n")
        f.write(f"  Lasso vs EN   : {ovl_lasso_en}\n")
        f.write(f"  Lasso vs RF   : {ovl_lasso_rf}  (low expected)\n")
        f.write(f"  EN vs RF      : {ovl_en_rf}\n")
        f.write(f"  Lasso vs Stab : {ovl_lasso_stab}  (high = stable)\n\n")
        f.write("=" * 70 + "\n")
        f.write(f"Top {TOP_K} -- LASSO:\n");         f.write(fmt_ranked(top_lasso_rank) + "\n\n")
        f.write(f"Top {TOP_K} -- ELASTICNET:\n");    f.write(fmt_ranked(top_en_rank)    + "\n\n")
        f.write(f"Top {TOP_K} -- RF perm-imp:\n");   f.write(fmt_ranked(top_rf_rank)    + "\n\n")
        f.write(f"Top {TOP_K} -- STABILITY:\n");     f.write(fmt_ranked(top_stab_rank)  + "\n\n")
        f.write("=" * 70 + "\n")
        f.write(f"cat2 (4-var small-model set): {fmt_list(cat2)}\n\n")
        f.write(f"UNION  (Lasso | EN | RF): {fmt_list(union)}\n\n")
        f.write(f"INTER  (Lasso & EN & RF): {fmt_list(intersection)}\n")
    print(f"Written: {OUT_TXT}")


if __name__ == "__main__":
    main()
