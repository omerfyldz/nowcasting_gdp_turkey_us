"""
run_stationarity_tr.py
=======================
Pipeline B — Turkey: Verify all transformed variables are stationary using
ZA + ADF + KPSS + PP test battery. Runs on the transformed CSV produced by
Step 4. Reports classification per variable.

This is verification only — tcodes are already applied. Role is same as
US run_stationarity.py: catch variables that remain non-stationary after
transformation, flag for review.

Run order: 5 (after build_final_tf_data_tr.py)

Input:  turkey_data/data_tf_monthly_tr.csv
Output: turkey_data/stationarity_report_tr.csv
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron

BASE = os.path.dirname(os.path.abspath(__file__))
TF_PATH = os.path.join(BASE, "data_tf_monthly_tr.csv")
OUTPUT = os.path.join(BASE, "stationarity_report_tr.csv")
COVID = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]
MIN_OBS = 20


def test_adf(s):
    try:
        ml = int(12 * (len(s) / 100) ** 0.25)
        _, p, _, _, _, _ = adfuller(s.values, maxlag=ml, regression="ct", autolag="AIC")
        return p < 0.05, p
    except:
        return False, 1.0


def test_kpss(s):
    try:
        ml = int(12 * (len(s) / 100) ** 0.25)
        _, p, _, _ = kpss(s.values, nlags=ml, regression="ct")
        return p > 0.05, p  # KPSS null = stationary, reject if p < 0.05
    except:
        return False, 0.0


def test_pp(s):
    try:
        pp = PhillipsPerron(s, trend='c')
        p = pp.pvalue
        return p < 0.05, p
    except:
        return False, 1.0


def test_za(s):
    try:
        ml = max(int(12 * (len(s) / 100) ** 0.25), 4)
        stat, pval, _, _, _ = zivot_andrews(s.values, maxlag=ml, regression="ct")
        z_p = pval[0] if hasattr(pval, '__len__') else float(pval) if not isinstance(pval, tuple) else pval[0]
        return z_p < 0.05, float(z_p)
    except:
        return False, 1.0


def main():
    print("=" * 60)
    print("STATIONARITY VERIFICATION — TURKEY")
    print("=" * 60)

    tf = pd.read_csv(TF_PATH, parse_dates=["date"])
    cols = [c for c in tf.columns if c not in ["date"] + COVID]

    print(f"\n   Testing {len(cols)} variables ...")
    results = []
    stationary = inconclusive = non_stationary = 0

    for col in cols:
        s = tf[col].dropna()
        if len(s) < MIN_OBS:
            results.append({"series": col, "classification": "INSUFFICIENT_DATA",
                            "n_obs": len(s), "ADF": None, "KPSS": None, "PP": None, "ZA": None})
            continue

        passed = 0
        a_ok, a_p = test_adf(s)
        if a_ok: passed += 1
        k_ok, k_p = test_kpss(s)
        if k_ok: passed += 1
        p_ok, p_p = test_pp(s)
        if p_ok: passed += 1
        z_ok, z_p = test_za(s) if len(s) > 30 else (None, None)
        if z_ok: passed += 1

        if passed >= 3:
            cls = "STATIONARY"; stationary += 1
        elif passed >= 2:
            cls = "INCONCLUSIVE"; inconclusive += 1
        else:
            cls = "NON_STATIONARY"; non_stationary += 1

        results.append({
            "series": col, "classification": cls, "tests_passed": passed,
            "n_obs": len(s), "ADF_p": round(a_p, 4), "KPSS_p": round(k_p, 4),
            "PP_p": round(p_p, 4),
            "ZA_p": round(z_p, 4) if z_p is not None else None,
        })

    res = pd.DataFrame(results).sort_values("classification")
    print(f"   STATIONARY:     {stationary}")
    print(f"   INCONCLUSIVE:   {inconclusive}")
    print(f"   NON-STATIONARY: {non_stationary}")

    if non_stationary > 0:
        ns = res[res["classification"] == "NON_STATIONARY"]
        print(f"   Non-stationary: {ns['series'].tolist()}")

    res.to_csv(OUTPUT, index=False)
    print(f"\n   Written: {OUTPUT}")

    # Quick target check
    gdp = tf[["date", "ngdprsaxdctrq"]].dropna()
    print(f"\n   GDP: {len(gdp)} quarterly obs after transformation")
    print(f"   First 3: {gdp['ngdprsaxdctrq'].head(3).round(6).tolist()}")


if __name__ == "__main__":
    main()
