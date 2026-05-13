"""
determine_tcodes_tr.py
=======================
Pipeline B — Turkey: Auto-determine McCracken-Ng-style transformation codes
for each variable using Zivot-Andrews (structural-break-robust) first,
with ADF fallback.

ZA priority order (Perron 1989 logic):
    Stage 1: ZA on levels (trend model) → if stationary, tcode=1
    Stage 2: ADF on levels → if stationary, tcode=1
    Stage 3: ADF on first differences → if stationary, tcode=2
    Stage 4: ADF on log-differences → if stationary, tcode=5
    Default: tcode=5

Zivot-Andrews (1992) jointly estimates the break date and tests for a unit
root around a broken trend. This prevents over-differencing series that are
trend-stationary with a structural break (e.g., Turkish GDP after 2001).

References:
    Perron, P. (1989). The Great Crash, the Oil Price Shock, and the Unit Root.
    Zivot, E. & Andrews, D.W. (1992). Further Evidence on the Great Crash...
    McCracken, M.W. & Ng, S. (2020). FRED-MD: A Monthly Database...

Run order: 3 (after build_metadata_tr.py)

Input:
    turkey_data/data_raw_monthly_tr.xlsx

Output:
    turkey_data/tcode_assignments_tr.csv  (series, tcode, method, break_date)
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, zivot_andrews

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "data_raw_monthly_tr.xlsx")
OUTPUT = os.path.join(BASE, "tcode_assignments_tr.csv")

MIN_OBS = 40  # minimum observations for reliable testing


def schwert_maxlag(n):
    """Schwert (1989) criterion for max lag length."""
    return int(12 * (n / 100) ** 0.25)


def test_za_stationary(series):
    """
    Test if series is trend-stationary around an endogenous structural break
    using Zivot-Andrews (1992).

    Returns (is_stationary: bool, pvalue: float, break_date: str or None)
    """
    series = series.dropna()
    if len(series) < MIN_OBS:
        return False, 1.0, None

    maxlag = schwert_maxlag(len(series))

    try:
        stat, pval, _, _, break_idx = zivot_andrews(
            series.values, maxlag=maxlag, regression="ct", trend="ct"
        )
        if pval < 0.05:
            break_date = str(series.index[break_idx].date()) if break_idx < len(series) else None
            return True, pval, break_date
        return False, pval, break_date
    except Exception:
        pass

    # Fallback: ZA with constant only
    try:
        stat, pval, _, _, break_idx = zivot_andrews(
            series.values, maxlag=maxlag, regression="ct", trend="c"
        )
        if pval < 0.05:
            break_date = str(series.index[break_idx].date()) if break_idx < len(series) else None
            return True, pval, break_date
        return False, pval, break_date
    except Exception:
        return False, 1.0, None


def test_adf_stationary(series, alpha=0.05):
    """Standard ADF test. Returns (is_stationary, pvalue)."""
    series = series.dropna()
    if len(series) < MIN_OBS:
        return False, 1.0

    maxlag = schwert_maxlag(len(series))
    try:
        _, pval, _, _, _, _ = adfuller(series.values, maxlag=maxlag,
                                        regression="ct", autolag="AIC")
        return pval < alpha, pval
    except Exception:
        try:
            _, pval, _, _, _, _ = adfuller(series.values, maxlag=maxlag,
                                            regression="c", autolag="AIC")
            return pval < alpha, pval
        except Exception:
            return False, 1.0


def determine_tcode(series, var_name):
    """
    Hierarchical stationarity testing for a single variable.
    Returns (tcode: int, method: str, break_date: str or None).
    """
    n_obs = series.dropna().shape[0]
    if n_obs < MIN_OBS:
        return None, f"insufficient_obs({n_obs})", None

    # ---- Stage 1: ZA on levels ----
    is_stat, pval, break_date = test_za_stationary(series)
    if is_stat:
        return 1, f"ZA_levels(p={pval:.4f})", break_date

    # ---- Stage 2: ADF on levels ----
    is_stat, pval = test_adf_stationary(series)
    if is_stat:
        return 1, f"ADF_levels(p={pval:.4f})", None

    # ---- Stage 3: ADF on first differences ----
    diff = series.diff().dropna()
    if len(diff) >= MIN_OBS:
        is_stat, pval = test_adf_stationary(diff)
        if is_stat:
            return 2, f"ADF_diff(p={pval:.4f})", None

    # ---- Stage 4: ADF on log-differences ----
    positive = series[series > 0]
    if len(positive) >= MIN_OBS:
        log_series = np.log(positive)
        log_diff = log_series.diff().dropna()
        if len(log_diff) >= MIN_OBS:
            is_stat, pval = test_adf_stationary(log_diff)
            if is_stat:
                return 5, f"ADF_logdiff(p={pval:.4f})", None

    # ---- Default ----
    return 5, "default(log-diff)", None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DETERMINE TCODES — TURKEY (ZA + ADF)")
    print("=" * 60)

    # 1. Load raw data
    print("\n1. Loading raw data ...")
    raw = pd.read_excel(RAW_PATH, parse_dates=["Date"])
    raw = raw.set_index("Date")
    cols = [c for c in raw.columns]
    print(f"   {len(cols)} variables")

    # 2. Determine tcodes
    print("\n2. Testing stationarity (ZA first, ADF fallback) ...")
    results = []
    za_stationary_count = 0
    excluded_count = 0

    for col in cols:
        series = raw[col].copy()
        series.index = pd.to_datetime(series.index)
        tcode, method, break_date = determine_tcode(series, col)

        if tcode is None:
            excluded_count += 1
            results.append({
                "series": col.lower(),
                "tcode": None,
                "method": method,
                "break_date": break_date,
                "n_obs": series.dropna().shape[0],
            })
            continue

        if "ZA_levels" in method:
            za_stationary_count += 1

        results.append({
            "series": col.lower(),
            "tcode": tcode,
            "method": method,
            "break_date": break_date,
            "n_obs": series.dropna().shape[0],
        })

    tcode_df = pd.DataFrame(results)

    # 3. Summary
    print(f"\n3. Summary:")
    print(f"   Total variables: {len(cols)}")
    print(f"   Excluded (too few obs): {excluded_count}")
    print(f"   ZA-stationary (no differencing needed): {za_stationary_count}")

    valid = tcode_df[tcode_df["tcode"].notna()]
    tcode_counts = valid["tcode"].value_counts().sort_index()
    for tc, count in tcode_counts.items():
        print(f"   tcode={int(tc)}: {count} variables")

    # 4. List ZA-detected breaks
    za_vars = tcode_df[tcode_df["method"].str.contains("ZA", na=False)]
    if len(za_vars) > 0:
        print(f"\n   ZA-detected structural breaks:")
        for _, row in za_vars.iterrows():
            bd = row["break_date"] if pd.notna(row["break_date"]) else "unknown"
            print(f"      {row['series']}: break at {bd} ({row['method']})")

    # 5. Save
    tcode_df.to_csv(OUTPUT, index=False)
    print(f"\n   Written: {OUTPUT}")
    return tcode_df


if __name__ == "__main__":
    main()
