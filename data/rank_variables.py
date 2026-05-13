"""
rank_variables.py
=================
Reads RF permutation importance from feature_selection_ensemble.xlsx,
ranks all variables by importance, computes cumulative importance,
and identifies cutoff thresholds.

Output: variable_rankings.txt in the same directory.

Note: RF permutation importance was computed on the full 1959-2007
training window via feature_selection_ensemble.py. Only top-35
importances are stored. Variables not in top-35 are listed with
importance < minimum recorded value.

Reference: Strobl et al. (2007) "Bias in random forest variable
importance measures" — permutation importance is preferred over MDI
for avoiding bias toward high-cardinality features.
"""
import pandas as pd
import numpy as np
import os

BASE = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
IN_XLSX = os.path.join(BASE, "feature_selection_ensemble.xlsx")
OUT_TXT = os.path.join(BASE, "variable_rankings.txt")

# ---------------------------------------------------------------------------
# 1. Load existing RF permutation importance (top-35)
# ---------------------------------------------------------------------------
rf_df = pd.read_excel(IN_XLSX, sheet_name="RF_perm_imp")

rf_ranked = []
for _, row in rf_df.iterrows():
    rf_ranked.append({
        "rank": int(row["rank"]),
        "feature": str(row["feature"]).lower(),
        "importance": float(row["importance"]),
    })

total_imp = sum(r["importance"] for r in rf_ranked)
min_imp = min(r["importance"] for r in rf_ranked)

# ---------------------------------------------------------------------------
# 2. Load Union set
# ---------------------------------------------------------------------------
union_df = pd.read_excel(IN_XLSX, sheet_name="Union")
union_set = {str(f).lower() for f in union_df["feature"]}

rf_features = {r["feature"] for r in rf_ranked}
union_in_rf = union_set & rf_features          # Union vars with RF importance
union_not_in_rf = union_set - rf_features      # Union vars without RF importance

# ---------------------------------------------------------------------------
# 3. Also load Lasso and Stability for cross-reference
# ---------------------------------------------------------------------------
lasso_df = pd.read_excel(IN_XLSX, sheet_name="Lasso")
stab_df = pd.read_excel(IN_XLSX, sheet_name="Stability")
en_df = pd.read_excel(IN_XLSX, sheet_name="ElasticNet")

lasso_rank = dict(zip(lasso_df["feature"].str.lower(), lasso_df["rank"]))
en_rank = dict(zip(en_df["feature"].str.lower(), en_df["rank"]))
stab_freq = dict(zip(stab_df["feature"].str.lower(), stab_df["selection_freq"]))

# ---------------------------------------------------------------------------
# 4. Compute cumulative importance and cutoffs
# ---------------------------------------------------------------------------
cutoffs = {}
cumulative = 0.0
for i, r in enumerate(rf_ranked):
    cumulative += r["importance"]
    pct = cumulative / total_imp * 100
    r["cumulative_pct"] = round(pct, 1)
    for thresh in [50, 60, 70, 80, 90, 95, 99]:
        if thresh not in cutoffs and pct >= thresh:
            cutoffs[thresh] = i + 1

# ---------------------------------------------------------------------------
# 5. Write output
# ---------------------------------------------------------------------------
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 78 + "\n")
    f.write("VARIABLE RANKINGS — RF Permutation Importance\n")
    f.write("=" * 78 + "\n")
    f.write("Source    : feature_selection_ensemble.xlsx / RF_perm_imp\n")
    f.write("Method    : Random Forest (500 trees, permutation importance, n_repeats=10)\n")
    f.write("Window    : 1959-01 to 2007-12 (pre-GFC training)\n")
    f.write("Total RF  : {} vars ranked (top-35 stored, rest below threshold)\n".format(len(rf_ranked)))
    f.write("Total imp : {:.4f}\n\n".format(total_imp))

    # Cutoff summary box
    f.write("-" * 78 + "\n")
    f.write("CUMULATIVE IMPORTANCE CUTOFFS\n")
    f.write("-" * 78 + "\n")
    f.write("  Threshold    Vars needed    Cumulative %\n")
    f.write("  ---------    -----------    ------------\n")
    for thresh in [50, 60, 70, 80, 90, 95, 99]:
        n = cutoffs.get(thresh, ">35")
        if isinstance(n, int) and n <= len(rf_ranked):
            cum_at = rf_ranked[n - 1]["cumulative_pct"]
            f.write("  {:>4}%         {:<13}  {:.1f}%\n".format(thresh, n, cum_at))
        else:
            f.write("  {:>4}%         {:<13}  >{:.1f}%\n".format(thresh, str(n),
                    rf_ranked[-1]["cumulative_pct"] if n == ">35" else 0))
    f.write("\n")

    # Top-35 full ranking
    f.write("-" * 78 + "\n")
    f.write("TOP-35 RF PERMUTATION IMPORTANCE (ranked)\n")
    f.write("-" * 78 + "\n")
    f.write("  Rank  Feature                         Importance     Cum %\n")
    f.write("  ----  ------------------------------  ----------     -----\n")
    for r in rf_ranked:
        # Mark intersections with other methods
        flags = []
        if r["feature"] in lasso_rank:
            flags.append("L{}".format(lasso_rank[r["feature"]]))
        if r["feature"] in en_rank:
            flags.append("E{}".format(en_rank[r["feature"]]))
        if r["feature"] in stab_freq:
            flags.append("S{:.0%}".format(stab_freq[r["feature"]]))

        flag_str = " [" + " ".join(flags) + "]" if flags else ""

        # Mark cutoff thresholds
        marker = ""
        for thresh in cutoffs:
            if cutoffs[thresh] == r["rank"]:
                marker = " <-- {}% cumulative threshold".format(thresh)

        f.write("  {:>4}  {:<30}  {:.6f}    {:>5.1f}%{}\n".format(
            r["rank"], r["feature"], r["importance"], r["cumulative_pct"],
            marker
        ))
        if marker:
            f.write("  {}\n".format("-" * 50))

    # Union set variables NOT in RF top-35
    if union_not_in_rf:
        f.write("\n")
        f.write("-" * 78 + "\n")
        f.write("UNION SET VARIABLES BELOW RF TOP-35\n")
        f.write("(importance < {:.4g} — the minimum in top-35)\n".format(min_imp))
        f.write("-" * 78 + "\n")
        f.write("These are in the Union set (selected by Lasso or EN but not RF).\n")
        f.write("Count: {} vars\n\n".format(len(union_not_in_rf)))

        sorted_below = sorted(union_not_in_rf)
        # Show cross-reference
        for feat in sorted_below:
            flags = []
            if feat in lasso_rank:
                flags.append("L#{}".format(lasso_rank[feat]))
            if feat in en_rank:
                flags.append("E#{}".format(en_rank[feat]))
            if feat in stab_freq:
                flags.append("S{:.0%}".format(stab_freq[feat]))
            f.write("  {:<32s} {}\n".format(feat, " ".join(flags) if flags else "(no cross-ref)"))

    # Category recommendations
    f.write("\n")
    f.write("=" * 78 + "\n")
    f.write("CATEGORY ALLOCATION RECOMMENDATIONS\n")
    f.write("=" * 78 + "\n\n")

    f.write("Category 1 — Univariate (ARMA):  1 var\n")
    f.write("  gdpc1 (target only)\n\n")

    f.write("Category 2 — Unpenalized (VAR):  4 vars + 3 COVID = 7 total\n")
    f.write("  Triple intersection (Lasso + Stab + RF):\n")
    triple = sorted({"outbs", "outnfb", "hwiuratiox", "ulcnfb"} & set(r["feature"] for r in rf_ranked))
    for v in triple:
        rf_r = next(r for r in rf_ranked if r["feature"] == v)
        f.write("    {:20s} L#{:<3} S{:.0%} RF#{:<3} imp={:.4g}\n".format(
            v, lasso_rank.get(v, "-"), stab_freq.get(v, 0),
            rf_r["rank"], rf_r["importance"]))
    f.write("  + covid_2020q2, covid_2020q3, covid_2020q4\n\n")

    # Category 3 — Stability >= 50%
    stable50 = sorted(k for k, v in stab_freq.items() if v >= 0.50)
    f.write("Category 3 — Penalized (OLS, Lasso, Ridge, EN, BVAR, MIDAS, MIDASML):  "
            "{} vars + 3 COVID = {} total\n".format(len(stable50), len(stable50) + 3))
    f.write("  Rule: Stability >= 50% (Meinshausen-Buhlmann 2010)\n\n")
    for feat in stable50:
        rf_r = next((r for r in rf_ranked if r["feature"] == feat), None)
        s_f = stab_freq[feat]
        l_r = lasso_rank.get(feat, "-")
        rf_rank_str = "RF#{} imp={:.4g}".format(rf_r["rank"], rf_r["importance"]) if rf_r else "RF: below top-35"
        f.write("    {:30s} L#{}   S{:.0%}   {}\n".format(feat, l_r, s_f, rf_rank_str))
    f.write("  + covid_2020q2, covid_2020q3, covid_2020q4\n\n")

    # Category 4 — Union
    union_sorted = sorted(union_set)
    f.write("Category 4 — ML + DFM (RF, XGB, GB, DT, MLP, LSTM, DeepVAR, DFM):  "
            "{} vars + 3 COVID = {} total\n".format(len(union_sorted), len(union_sorted) + 3))
    f.write("  Rule: Union (Lasso U EN U RF) — all variables flagged by >=1 method\n\n")
    f.write("  Full Union set:\n")
    for i, feat in enumerate(union_sorted):
        rf_r = next((r for r in rf_ranked if r["feature"] == feat), None)
        if rf_r:
            f.write("    {:30s} RF#{} imp={:.4g}\n".format(feat, rf_r["rank"], rf_r["importance"]))
        else:
            has_l = "L#{}".format(lasso_rank[feat]) if feat in lasso_rank else ""
            has_e = "E#{}".format(en_rank[feat]) if feat in en_rank else ""
            f.write("    {:30s} {} {}\n".format(feat, has_l, has_e))
    f.write("  + covid_2020q2, covid_2020q3, covid_2020q4\n\n")

    f.write("=" * 78 + "\n")
    f.write("END OF RANKINGS\n")

print("Wrote {}".format(OUT_TXT))
print("Top-{} RF vars ranked. Cumulative cutoffs:".format(len(rf_ranked)))
for thresh in sorted(cutoffs.keys()):
    print("  {}% cumulative at var #{}".format(thresh, cutoffs[thresh]))
print("Stability >= 50%: {} vars".format(len(stable50)))
print("Union set: {} vars".format(len(union_sorted)))
