"""
rank_by_rules.py
================
Rule-based variable ranking for ALL FOUR selection methods.
No arbitrary "top-35" cutoff. Each method's stop point is determined by
its own defensible rule.

Lasso:    cumulative absolute coefficient importance
ElasticNet: cumulative absolute coefficient importance
RF:       cumulative permutation importance
Stability: selection frequency (Meinshausen-Buhlmann 2010)

Fits Lasso and EN with the CV-selected hyperparameters from
feature_selection_ensemble.xlsx Run_metadata, then ranks ALL variables
(not just top-35). RF and Stability rankings are read from the xlsx.

Output: variable_rankings_by_rule.txt
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

BASE = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
SEED = 42
TARGET = "gdpc1"
TRAIN_START = "1959-01-01"
TRAIN_END   = "2007-12-31"
N_LAGS = 3

# Tuned HPs from feature_selection_ensemble.xlsx Run_metadata
LASSO_ALPHA = 2.477e-05
EN_ALPHA    = 0.0001
EN_L1_RATIO = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (identical to feature_selection_ensemble.py)
# ─────────────────────────────────────────────────────────────────────────────
def flatten_data(data, target_variable, n_lags):
    flattened_data = data.loc[~pd.isna(data[target_variable]), :].copy()
    orig_index = flattened_data.index
    for i in range(1, n_lags + 1):
        lagged_indices = orig_index - i
        lagged_indices = lagged_indices[lagged_indices >= 0]
        tmp = data.loc[lagged_indices, :].copy()
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
    """Sum |importance| across n_lags lagged copies per base variable."""
    out = {}
    for i, col in enumerate(columns):
        base = col
        if "_" in col and col.rsplit("_", 1)[1].isdigit():
            base = col.rsplit("_", 1)[0]
        out[base] = out.get(base, 0.0) + abs(importances[i])
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and preprocess data (same as feature_selection_ensemble.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(os.path.join(BASE, "data_tf_monthly.csv"), parse_dates=["date"])
train = data.loc[(data.date >= TRAIN_START) & (data.date <= TRAIN_END), :].reset_index(drop=True)
print("Train window: {} to {} ({} monthly rows)".format(TRAIN_START, TRAIN_END, len(train)))

# Drop columns entirely NaN in training window
all_nan_cols = [c for c in train.columns if c != "date" and train[c].notna().sum() == 0]
if all_nan_cols:
    print("Dropping {} columns with no data in train window".format(len(all_nan_cols)))
    train = train.drop(columns=all_nan_cols)

# Preprocess
filled_train = mean_fill_dataset(train, train)
flattened = flatten_data(filled_train, TARGET, N_LAGS)
flattened = flattened.loc[flattened.date.dt.month.isin([3, 6, 9, 12]), :] \
                      .dropna(axis=0, how="any").reset_index(drop=True)

X = flattened.drop(["date", TARGET], axis=1)
y = flattened[TARGET]

# Drop COVID dummies (zero variance pre-2020)
covid_cols = [c for c in X.columns if "covid" in c.lower()]
X = X.drop(covid_cols, axis=1)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("X shape: {} | y shape: {}".format(X_scaled.shape, y.shape))
print("{} training quarters after preprocessing".format(len(y)))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit Lasso with CV-selected alpha → full coefficient ranking
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/2] Fitting Lasso (alpha={:.4g})...".format(LASSO_ALPHA))
lasso = Lasso(alpha=LASSO_ALPHA, max_iter=20000, random_state=SEED)
lasso.fit(X_scaled, y)
imp_lasso = aggregate_importance(lasso.coef_, X.columns.tolist())
n_nonzero = sum(1 for v in imp_lasso.values() if v > 0)
print("  Non-zero coefficients: {} base variables".format(n_nonzero))

# Rank by |coef|
lasso_ranked = sorted(imp_lasso.items(), key=lambda x: -x[1])
lasso_total = sum(v for _, v in lasso_ranked)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fit ElasticNet with CV-selected alpha/l1_ratio → full coefficient ranking
# ─────────────────────────────────────────────────────────────────────────────
print("[2/2] Fitting ElasticNet (alpha={:.4g}, l1_ratio={:.3f})...".format(
    EN_ALPHA, EN_L1_RATIO))
en = ElasticNet(alpha=EN_ALPHA, l1_ratio=EN_L1_RATIO, max_iter=20000, random_state=SEED)
en.fit(X_scaled, y)
imp_en = aggregate_importance(en.coef_, X.columns.tolist())
n_nonzero_en = sum(1 for v in imp_en.values() if v > 0)
print("  Non-zero coefficients: {} base variables".format(n_nonzero_en))

en_ranked = sorted(imp_en.items(), key=lambda x: -x[1])
en_total = sum(v for _, v in en_ranked)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Read RF and Stability from existing xlsx
# ─────────────────────────────────────────────────────────────────────────────
rf_df = pd.read_excel(os.path.join(BASE, "feature_selection_ensemble.xlsx"),
                       sheet_name="RF_perm_imp")
stab_df = pd.read_excel(os.path.join(BASE, "feature_selection_ensemble.xlsx"),
                         sheet_name="Stability")

rf_ranked = [(str(r["feature"]).lower(), r["importance"])
             for _, r in rf_df.iterrows()]
rf_total = sum(v for _, v in rf_ranked)

stab_ranked = [(str(r["feature"]).lower(), r["selection_freq"])
               for _, r in stab_df.iterrows()]
stab_total = sum(v for _, v in stab_ranked)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cumulative importance helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_cumulative(ranked_list):
    """Return list of (name, importance, cumulative_pct)."""
    total = sum(v for _, v in ranked_list)
    cum = 0.0
    result = []
    for name, imp in ranked_list:
        cum += imp
        result.append((name, imp, cum / total * 100))
    return result, total

def find_cutoff(cumulative_list, threshold_pct):
    """Return the index (1-based) where cumulative pct first >= threshold."""
    for i, (name, imp, cum_pct) in enumerate(cumulative_list):
        if cum_pct >= threshold_pct:
            return i + 1, name, cum_pct
    return len(cumulative_list), cumulative_list[-1][0], cumulative_list[-1][2]

lasso_cum, lasso_tot = compute_cumulative(lasso_ranked)
en_cum, en_tot = compute_cumulative(en_ranked)
rf_cum, rf_tot = compute_cumulative(rf_ranked)
stab_cum, stab_tot = compute_cumulative(stab_ranked)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build rule-based sets at different thresholds
# ─────────────────────────────────────────────────────────────────────────────
def set_at_threshold(cumulative_list, threshold_pct):
    """Return set of variable names up to threshold_pct cumulative."""
    result = set()
    cum = 0.0
    total = sum(v for _, v in cumulative_list)  # use original unsorted? no, already sorted
    # Actually cumulative_list is already sorted. Let's just use the list.
    # Recompute total from the full list
    full_cum = 0.0
    cutoff_set = set()
    for name, imp in cumulative_list:  # use the original ranked list
        full_cum += imp
        cutoff_set.add(name)
        if full_cum / total >= threshold_pct / 100:
            break
    return cutoff_set

# Use the original ranked tuples for set building
lasso_ranked_tuples = lasso_ranked  # (name, imp) sorted desc
en_ranked_tuples = en_ranked
rf_ranked_tuples = rf_ranked
stab_ranked_tuples = stab_ranked

# ─────────────────────────────────────────────────────────────────────────────
# 7. Write output
# ─────────────────────────────────────────────────────────────────────────────
OUT = os.path.join(BASE, "variable_rankings_by_rule.txt")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("=" * 82 + "\n")
    f.write("VARIABLE RANKINGS BY RULE (NO ARBITRARY TOP-35)\n")
    f.write("=" * 82 + "\n")
    f.write("Training: {} to {} ({} quarterly obs)\n".format(TRAIN_START, TRAIN_END, len(y)))
    f.write("n_lags: {}\n\n".format(N_LAGS))

    # ── Cumulative cutoff summary ──
    f.write("-" * 82 + "\n")
    f.write("CUMULATIVE IMPORTANCE CUTOFFS ACROSS ALL FOUR METHODS\n")
    f.write("-" * 82 + "\n")
    f.write("  Thresh   Lasso(#)       EN(#)          RF(#)          Stab(#)\n")
    f.write("  ------   --------       ----           ----           -------\n")
    for thresh in [50, 60, 70, 80, 90, 95, 99]:
        ln, lname, lcum = find_cutoff(lasso_cum, thresh)
        enn, ename, encum = find_cutoff(en_cum, thresh)
        rn, rname, rcum = find_cutoff(rf_cum, thresh)
        sn, sname, scum = find_cutoff(stab_cum, thresh)
        f.write("  {:>4}%    {:>3} ({:<15s}) {:>3} ({:<15s}) {:>3} ({:<15s}) {:>3} ({:<15s})\n".format(
            thresh,
            ln, lname[:15],
            enn, ename[:15],
            rn, rname[:15],
            sn, sname[:15],
        ))
    f.write("\n")

    # ── Overlap at different thresholds ──
    f.write("-" * 82 + "\n")
    f.write("OVERLAP BETWEEN RULE-BASED SETS\n")
    f.write("-" * 82 + "\n")
    f.write("  Thresh   Lasso    EN       RF       Stab     LnEn    LnEnR   LnEnRnS\n")
    f.write("  ------   -----    --       --       ----     ----    -----   -------\n")
    for thresh in [50, 70, 80, 90, 95, 99]:
        Ls = set_at_threshold(lasso_ranked, thresh)
        Es = set_at_threshold(en_ranked, thresh)
        Rs = set_at_threshold(rf_ranked, thresh)
        Ss = set_at_threshold(stab_ranked, thresh)
        f.write("  {:>4}%    {:>4}    {:>4}     {:>4}     {:>4}     {:>4}    {:>4}     {:>4}\n".format(
            thresh, len(Ls), len(Es), len(Rs), len(Ss),
            len(Ls & Es), len(Ls & Es & Rs),
            len(Ls & Es & Rs & Ss),
        ))
    f.write("\n")

    # ── Key thresholds for paper ──
    f.write("=" * 82 + "\n")
    f.write("RECOMMENDED THRESHOLDS BY METHOD\n")
    f.write("=" * 82 + "\n\n")

    f.write("Stability (Meinshausen-Buhlmann 2010):\n")
    f.write("  Rule: variables selected in >= 50% of bootstrap Lasso resamples.\n")
    sn, _, scum = find_cutoff(stab_cum, 50)
    f.write("  Result: {} vars at 50%, {} vars at 80%\n\n".format(
        sn, find_cutoff(stab_cum, 80)[0]))

    f.write("RF Permutation Importance:\n")
    f.write("  Rule: cumulative permutation importance threshold.\n")
    for t in [50, 80, 90, 95, 99]:
        n, name, cum = find_cutoff(rf_cum, t)
        f.write("  {}%: {} vars (stops at {})\n".format(t, n, name))
    f.write("  NOTE: RF importance is extremely concentrated.\n")
    f.write("  outbs alone = {:.1f}% of total. 80% reached at 2 vars.\n\n".format(
        rf_cum[0][2]))

    f.write("Lasso (|coef| at CV-selected alpha={:.4g}):\n".format(LASSO_ALPHA))
    f.write("  Rule: cumulative absolute coefficient importance.\n")
    for t in [50, 80, 90, 95, 99]:
        n, name, cum = find_cutoff(lasso_cum, t)
        f.write("  {}%: {} vars (stops at {})\n".format(t, n, name))
    f.write("\n")

    f.write("ElasticNet (|coef| at alpha={:.4g}, l1={:.3f}):\n".format(EN_ALPHA, EN_L1_RATIO))
    f.write("  Rule: cumulative absolute coefficient importance.\n")
    for t in [50, 80, 90, 95, 99]:
        n, name, cum = find_cutoff(en_cum, t)
        f.write("  {}%: {} vars (stops at {})\n".format(t, n, name))
    f.write("\n")

    # ── Full Lasso ranking ──
    f.write("-" * 82 + "\n")
    f.write("LASSO FULL RANKING (|coef|, alpha={:.4g})\n".format(LASSO_ALPHA))
    f.write("-" * 82 + "\n")
    f.write("Total variables with non-zero coef: {}\n".format(n_nonzero))
    f.write("Total absolute coef sum: {:.4f}\n\n".format(lasso_tot))
    f.write("  Rank  Feature                          |coef|       Cum%\n")
    f.write("  ----  -------                          ------       ----\n")
    for i, (name, imp, cum_pct) in enumerate(lasso_cum):
        marker = ""
        for t in [50, 80, 90, 95]:
            if i > 0 and lasso_cum[i-1][2] < t and cum_pct >= t:
                marker = " <-- {}%".format(t)
        f.write("  {:>4}  {:<30s}  {:.6f}   {:>6.1f}%{}\n".format(
            i+1, name, imp, cum_pct, marker))
        # Show ~50 vars max
        if i >= 49:
            remaining = len(lasso_cum) - 50
            if remaining > 0:
                f.write("  ...  {} more variables with non-zero coef ...\n".format(remaining))
            break
    f.write("\n")

    # ── Full EN ranking ──
    f.write("-" * 82 + "\n")
    f.write("ELASTICNET FULL RANKING (|coef|, alpha={:.4g}, l1={:.3f})\n".format(EN_ALPHA, EN_L1_RATIO))
    f.write("-" * 82 + "\n")
    f.write("Total variables with non-zero coef: {}\n".format(n_nonzero_en))
    f.write("Total absolute coef sum: {:.4f}\n\n".format(en_tot))
    f.write("  Rank  Feature                          |coef|       Cum%\n")
    f.write("  ----  -------                          ------       ----\n")
    for i, (name, imp, cum_pct) in enumerate(en_cum):
        marker = ""
        for t in [50, 80, 90, 95]:
            if i > 0 and en_cum[i-1][2] < t and cum_pct >= t:
                marker = " <-- {}%".format(t)
        f.write("  {:>4}  {:<30s}  {:.6f}   {:>6.1f}%{}\n".format(
            i+1, name, imp, cum_pct, marker))
        if i >= 49:
            remaining = len(en_cum) - 50
            if remaining > 0:
                f.write("  ...  {} more variables with non-zero coef ...\n".format(remaining))
            break
    f.write("\n")

    # ── Comparison: what do we get at 95% cumulative? ──
    f.write("=" * 82 + "\n")
    f.write("95% CUMULATIVE SETS — PAIRWISE OVERLAP\n")
    f.write("=" * 82 + "\n\n")
    L95 = set_at_threshold(lasso_ranked, 95)
    E95 = set_at_threshold(en_ranked, 95)
    R95 = set_at_threshold(rf_ranked, 95)
    S95 = set_at_threshold(stab_ranked, 95)
    f.write("Lasso 95%:       {} vars\n".format(len(L95)))
    f.write("EN 95%:          {} vars\n".format(len(E95)))
    f.write("RF 95%:          {} vars\n".format(len(R95)))
    f.write("Stab 95%:        {} vars\n\n".format(len(S95)))
    f.write("Lasso n EN:      {} vars\n".format(len(L95 & E95)))
    f.write("Lasso n RF:      {} vars\n".format(len(L95 & R95)))
    f.write("Lasso n Stab:    {} vars\n".format(len(L95 & S95)))
    f.write("Lasso n EN n RF: {} vars\n".format(len(L95 & E95 & R95)))
    tri95 = L95 & E95 & R95 & S95
    f.write("ALL 4 at 95%:    {} vars\n".format(len(tri95)))
    if tri95:
        for v in sorted(tri95):
            f.write("  {}\n".format(v))
    f.write("\n")

    # ── 99% cumulative ──
    f.write("=" * 82 + "\n")
    f.write("99% CUMULATIVE SETS — PAIRWISE OVERLAP\n")
    f.write("=" * 82 + "\n\n")
    L99 = set_at_threshold(lasso_ranked, 99)
    E99 = set_at_threshold(en_ranked, 99)
    R99 = set_at_threshold(rf_ranked, 99)
    S99 = set_at_threshold(stab_ranked, 99)
    f.write("Lasso 99%:       {} vars\n".format(len(L99)))
    f.write("EN 99%:          {} vars\n".format(len(E99)))
    f.write("RF 99%:          {} vars\n".format(len(R99)))
    f.write("Stab 99%:        {} vars\n\n".format(len(S99)))
    f.write("Lasso n EN:      {} vars\n".format(len(L99 & E99)))
    f.write("Lasso n RF:      {} vars\n".format(len(L99 & R99)))
    f.write("Lasso n EN n RF: {} vars\n".format(len(L99 & E99 & R99)))
    tri99 = L99 & E99 & R99 & S99
    f.write("ALL 4 at 99%:    {} vars\n".format(len(tri99)))
    if tri99:
        for v in sorted(tri99):
            f.write("  {}\n".format(v))

    # ── Stability >= 50% set ──
    f.write("\n")
    f.write("=" * 82 + "\n")
    f.write("STABILITY >= 50% SET (Meinshausen-Buhlmann)\n")
    f.write("=" * 82 + "\n\n")
    stable50 = sorted(k for k, v in stab_ranked if v >= 0.50)
    f.write("{} vars\n\n".format(len(stable50)))
    for v in stable50:
        v_freq = next(fr for n, fr in stab_ranked if n == v)
        l_rank = next((i+1 for i,(n,_) in enumerate(lasso_ranked) if n == v), "-")
        e_rank = next((i+1 for i,(n,_) in enumerate(en_ranked) if n == v), "-")
        r_rank = next((i+1 for i,(n,_) in enumerate(rf_ranked) if n == v), "-")
        f.write("  {:<30s}  Stab={:.0%}  L#{:>4s}  E#{:>4s}  RF#{:>4s}\n".format(
            v, v_freq, str(l_rank), str(e_rank), str(r_rank)))

    f.write("\n" + "=" * 82 + "\n")
    f.write("END\n")

print("Wrote {}".format(OUT))
print("Done. Lasso: {} non-zero vars, EN: {} non-zero vars".format(n_nonzero, n_nonzero_en))
