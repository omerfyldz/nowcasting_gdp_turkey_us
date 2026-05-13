"""
build_final_tf_data_tr.py
==========================
Pipeline B — Turkey: Apply McCracken-Ng-style transformation codes, seasonal
adjustment for NSA variables, COVID dummies, and produce the final transformed
CSVs that model notebooks consume.

Processing order per variable:
    1. Seasonal adjustment (STL) for NSA variables with sufficient history
    2. Apply tcode (1=none, 2=diff, 5=log-diff)
    3. Sanitize inf / extreme values

GDP special handling: quarterly series on monthly grid → log-diff after
dropna to get Q-to-Q growth rate at Q-end months only.

References:
    McCracken & Ng (2020) — tcode definitions
    US Pipeline B — apply_tcode logic, COVID dummy structure

Run order: 4 (after determine_tcodes_tr.py)

Inputs:
    turkey_data/data_raw_monthly_tr.xlsx
    turkey_data/tcode_assignments_tr.csv
    turkey_data/turkey_data_dictionary.xlsx  (for seasonal adj. flags)

Outputs:
    turkey_data/data_tf_monthly_tr.csv
    turkey_data/data_tf_weekly_tr.csv
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "data_raw_monthly_tr.xlsx")
WEEKLY_RAW = os.path.join(BASE, "turkey_weekly_series.xlsx")
TCODE_PATH = os.path.join(BASE, "tcode_assignments_tr.csv")
DICT_PATH = os.path.join(BASE, "turkey_data_dictionary.xlsx")
OUT_MONTHLY = os.path.join(BASE, "data_tf_monthly_tr.csv")
OUT_WEEKLY = os.path.join(BASE, "data_tf_weekly_tr.csv")

COVID_DUMMIES = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]


# ═══════════════════════════════════════════════════════════════════════════════
# Seasonal Adjustment
# ═══════════════════════════════════════════════════════════════════════════════

def seasonal_adjust_stl(series, period=12, min_obs=24):
    """
    STL seasonal adjustment. Robust to outliers.
    Returns SA series (or original if insufficient observations).
    """
    s = series.dropna()
    if len(s) < min_obs:
        return series  # not enough data
    try:
        stl = STL(s, period=period, robust=True)
        result = stl.fit()
        out = series.copy()
        idx = series[series.notna()].index
        out.loc[idx] = result.trend + result.resid
        return out
    except Exception:
        return series


# ═══════════════════════════════════════════════════════════════════════════════
# Tcode Application
# ═══════════════════════════════════════════════════════════════════════════════

def apply_tcode(series, tcode, warnings_list=None):
    """
    Apply McCracken-Ng transformation code.
    Mirrors US pipeline logic exactly.

    Approach: drop NaNs first (so quarterly variables on a monthly grid
    compute Q-to-Q diffs, not month-to-month NaN-spanning diffs), then
    drop non-positives for log tcodes, compute the transformation on the
    dense valid sequence, and finally reindex to the original time grid.
    Inf from zero-denominator pct_change is sanitised to NaN.
    """
    if pd.isna(tcode):
        return series
    tcode = int(tcode)
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(np.nan, index=series.index, name=series.name)

    if tcode in (4, 5, 6):
        n_bad = int((s <= 0).sum())
        if n_bad > 0:
            if warnings_list is not None:
                warnings_list.append(
                    f"{series.name}: tcode={tcode} dropped {n_bad} non-positive obs"
                )
            s = s[s > 0]
            if len(s) == 0:
                return pd.Series(np.nan, index=series.index, name=series.name)

    if tcode == 1:
        out = s
    elif tcode == 2:
        out = s.diff()
    elif tcode == 3:
        out = s.diff().diff()
    elif tcode == 4:
        out = np.log(s)
    elif tcode == 5:
        out = np.log(s).diff()
    elif tcode == 6:
        out = np.log(s).diff().diff()
    elif tcode == 7:
        out = s.pct_change()
    else:
        out = s

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reindex(series.index)


def sanitize(series):
    """Replace inf/-inf with NaN."""
    return series.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("BUILD FINAL TF DATA — TURKEY")
    print("=" * 60)

    # ── 1. Load inputs ────────────────────────────────────────────────────────
    print("\n1. Loading inputs ...")
    raw = pd.read_excel(RAW_PATH, parse_dates=["Date"])
    tcodes = pd.read_csv(TCODE_PATH)
    tr_dict = pd.read_excel(DICT_PATH)
    # Rename Turkish characters to ASCII in Variable column
    TR_TO_ASCII = {"\u0131": "i", "\u0130": "I", "\u015f": "s", "\u015e": "S",
                   "\u011f": "g", "\u011e": "G", "\u00e7": "c", "\u00c7": "C",
                   "\u00f6": "o", "\u00d6": "O", "\u00fc": "u", "\u00dc": "U"}
    for tr_c, ascii_c in TR_TO_ASCII.items():
        tr_dict["Variable"] = tr_dict["Variable"].str.replace(tr_c, ascii_c, regex=False)

    tcode_map = {}
    for _, r in tcodes.iterrows():
        if pd.notna(r["tcode"]):
            tcode_map[r["series"]] = int(r["tcode"])

    # Build a fuzzy lookup: data column -> nearest dictionary entry
    col_to_dict_var = {}
    dict_vars = [str(r["Variable"]).lower().strip() for _, r in tr_dict.iterrows()]
    for col_lower in [c.lower().strip() for c in raw.columns if c != "Date"]:
        # Exact match
        if col_lower in dict_vars:
            col_to_dict_var[col_lower] = col_lower
            continue
        # Substring / fuzzy match
        for dv in dict_vars:
            if col_lower in dv or dv in col_lower:
                col_to_dict_var[col_lower] = dv
                break
        # Default: use column name itself
        if col_lower not in col_to_dict_var:
            col_to_dict_var[col_lower] = col_lower
    # Map: data_column_name_lower -> is_sa (bool)
    sa_flags = {}
    var_to_sa = {}
    for _, r in tr_dict.iterrows():
        var = str(r["Variable"]).lower().strip()
        sa_str = str(r.get("Seasonal Adj.", "")).lower().strip()
        sa_flags[var] = (sa_str == "yes")

    # Also identify NSA variables that genuinely need seasonal adjustment.
    # Literature (Eurostat 2015, Stock & Watson 2016): SA is for real-economy
    # variables with natural seasonal patterns (production, labor, tourism, trade,
    # retail, prices, tax, electricity). Financial variables (rates, FX, stocks,
    # reserves, money supply) and survey data do NOT have seasonal components.
    #
    # NSA variables needing SA: Real Activity, Real Sector, Labor, Price Index,
    #   Tourism, Housing, Fiscal, and External-trade (export/import volumes have
    #   seasonal shipping patterns).
    # NSA variables NOT needing SA: Financial, Financial Sector, Monetary,
    #   Survey (already done), External-Financial, External-Stability.
    sa_needed_cats = {
        "real activity", "real sector", "labor market", "price index",
        "tourism", "housing", "real activity / housing",
        "fiscal sector / government revenue", "fiscal sector",
    }
    # External Sector is split: trade volumes need SA, financial flows don't
    external_trade_vars = {
        "exp_vol_i", "imp_vol_i", "exp_ind_auto", "exp_ind_dg",
        "imp_ind_auto", "imp_ind_dg", "auto_exp_vol_i", "dg_exp_vol_i",
        "auto_imp_vol_i", "dg_imp_vol_i",
    }

    nsa_real_vars = set()
    for _, r in tr_dict.iterrows():
        var = str(r["Variable"]).lower().strip()
        sa_str = str(r.get("Seasonal Adj.", "")).lower().strip()
        cat = str(r.get("Economic Category", "")).lower().strip()
        if sa_str == "no":
            if any(k in cat for k in sa_needed_cats):
                nsa_real_vars.add(var)
            elif var in external_trade_vars:
                nsa_real_vars.add(var)
            # Card transactions, card payments (Real Sector/Activity, No SA flag)
            if "card" in var and sa_str == "no":
                nsa_real_vars.add(var)
            # Electricity (Real Activity, No SA flag)
            if var in {"elec_prod", "elec_cons"}:
                nsa_real_vars.add(var)
            # Tax (Fiscal Sector, No SA flag)
            if var == "tax":
                nsa_real_vars.add(var)

    print(f"   Raw: {raw.shape}")
    print(f"   Tcodes: {len(tcode_map)} variables")
    print(f"   Dictionary: {len(sa_flags)} entries")

    # ── 2. Transform each variable ─────────────────────────────────────────────
    print("\n2. Transforming variables ...")
    tf_data = pd.DataFrame()
    tf_data["date"] = raw["Date"]

    n_sa_applied = 0
    n_transformed = 0
    warnings_list = []

    for col in raw.columns:
        if col == "Date":
            continue

        col_lower = col.lower().strip()
        series = raw[col].copy()

        # Get tcode
        tcode = tcode_map.get(col_lower)
        if tcode is None:
            print(f"   SKIP  {col}: no tcode assigned")
            continue

        # Seasonal adjustment: only for NSA real-economy variables
        dict_var = col_to_dict_var.get(col_lower, col_lower)
        needs_sa = (dict_var in nsa_real_vars)
        if needs_sa and tcode != 1:
            series_before = series.copy()
            series = seasonal_adjust_stl(series)
            if not series.equals(series_before):
                n_sa_applied += 1

        # Apply tcode (identical to US pipeline: dropna().diff().reindex())
        tf_series = apply_tcode(series, tcode, warnings_list)

        # Sanitize
        tf_series = sanitize(tf_series)

        tf_data[col_lower] = tf_series
        n_transformed += 1

    if warnings_list:
        print(f"   DATA HYGIENE WARNINGS ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"    - {w}")

    print(f"   Seasonal adjustment applied: {n_sa_applied} variables")
    print(f"   Transformed: {n_transformed} variables")

    # ── 3. Add COVID dummies ───────────────────────────────────────────────────
    print("\n3. Adding COVID dummies ...")
    for q, months in [("q2", [4, 5, 6]), ("q3", [7, 8, 9]), ("q4", [10, 11, 12])]:
        col_name = f"covid_2020{q}"
        tf_data[col_name] = (
            (tf_data["date"].dt.year == 2020) &
            (tf_data["date"].dt.month.isin(months))
        ).astype(int)
        ones = tf_data[col_name].sum()
        print(f"   {col_name}: {int(ones)} ones")

    # ── 4. Sanitize all columns ────────────────────────────────────────────────
    print("\n4. Sanitizing ...")
    for col in tf_data.columns:
        if col == "date":
            continue
        tf_data[col] = sanitize(tf_data[col])

    inf_count = np.isinf(tf_data.select_dtypes(include=[np.number]).values).sum()
    print(f"   Inf values remaining: {inf_count}")

    # ── 5. Save monthly ────────────────────────────────────────────────────────
    print("\n5. Saving monthly ...")
    tf_data.to_csv(OUT_MONTHLY, index=False)
    print(f"   Shape: {tf_data.shape}")
    print(f"   Date range: {tf_data['date'].min().date()} to {tf_data['date'].max().date()}")
    print(f"   Columns: {len(tf_data.columns) - 1} transformed + 3 COVID + date")
    print(f"   Written: {OUT_MONTHLY}")

    # ── 6. Weekly data ─────────────────────────────────────────────────────────
    print("\n6. Weekly data ...")
    if os.path.exists(WEEKLY_RAW):
        w = pd.read_excel(WEEKLY_RAW, parse_dates=["Date"])
        w_out = pd.DataFrame()
        w_out["Date"] = w["Date"]

        for col in ["consu_i_weekly", "deposit_i_weekly"]:
            if col in w.columns:
                s = w[col].copy()
                tc = tcode_map.get(col.lower().replace("_weekly", ""), 5)
                tf = apply_tcode(s, tc)
                tf = sanitize(tf)
                w_out[col.lower()] = tf

        for q, months in [("q2", [4, 5, 6]), ("q3", [7, 8, 9]), ("q4", [10, 11, 12])]:
            col_name = f"covid_2020{q}"
            w_out[col_name] = (
                (w_out["Date"].dt.year == 2020) &
                (w_out["Date"].dt.month.isin(months))
            ).astype(int)

        w_out.to_csv(OUT_WEEKLY, index=False)
        print(f"   Shape: {w_out.shape}")
        print(f"   Written: {OUT_WEEKLY}")
    else:
        print("   Weekly file not found — skipping")

    # ── 7. GDP verification ────────────────────────────────────────────────────
    print("\n7. GDP verification ...")
    gdp_vals = tf_data[["date", "ngdprsaxdctrq"]].dropna()
    print(f"   gdpc1 non-NaN quarters: {len(gdp_vals)}")
    if len(gdp_vals) >= 3:
        print(f"   First 3 quarterly growth rates: "
              f"{gdp_vals['ngdprsaxdctrq'].head(3).round(6).tolist()}")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
