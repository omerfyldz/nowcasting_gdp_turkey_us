"""
build_raw_data_tr.py
====================
Pipeline B — Turkey: Read raw Turkish monthly series, parse European number
format, drop broken variables, place quarterly GDP on monthly grid at Q-end
months, run hard-fail range checks, output clean merged grid.

Run order: 1 (first script in the Turkey pipeline)

Input:
    turkey_data/tr_monthly_series.xlsx

Output:
    turkey_data/data_raw_monthly_tr.xlsx
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "tr_monthly_series.xlsx")
OUTPUT = os.path.join(BASE, "data_raw_monthly_tr.xlsx")

# ── Columns to DROP (broken data) ──────────────────────────────────────────────
# CPI has only 1 non-NaN observation — unusable. CPI_SA (255 obs) is the
# usable CPI series.
DROP_COLS = ["CPI"]

# ── Quarterly GDP: place at Q-end months (March, June, September, December) ────
# NGDPRSAXDCTRQ arrives as quarterly with values at Q-beginning months.
# We verify alignment and, if needed, shift to Q-end.
Q_END_MONTHS = [3, 6, 9, 12]
GDP_COL = "NGDPRSAXDCTRQ"


# ═══════════════════════════════════════════════════════════════════════════════
# European Number Format Parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tr_number(val):
    """
    Parse a Turkish/European-formatted number string to float.

    Handles:
      - "14.594,01"    → 14594.01   (period=thousands, comma=decimal)
      - "0,36"         → 0.36
      - "4.295e-05"    → 4.295e-05  (scientific, dot as decimal)
      - 3.6            → 3.6        (already float)
      - NaN            → NaN

    Strategy: try Python's native float() first (catches scientific notation
    and already-numeric values). If that fails, assume European format:
    strip all periods, replace comma with dot.
    """
    if pd.isna(val):
        return val
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if not isinstance(val, str):
        return float(val)

    val = val.strip()
    if val in ("", "-", "."):
        return np.nan

    # Stage 1: native parse (handles "4.295e-05", "37.0", pure integers)
    try:
        return float(val)
    except ValueError:
        pass

    # Stage 2: European format — strip periods (thousands), comma→dot
    val = val.replace(".", "")
    val = val.replace(",", ".")
    try:
        return float(val)
    except ValueError:
        print(f"  WARNING: could not parse '{val}' — returning NaN")
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Main Build
# ═══════════════════════════════════════════════════════════════════════════════

def build_raw_monthly():
    print("=" * 60)
    print("BUILD RAW MONTHLY DATA — TURKEY")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\n1. Loading tr_monthly_series.xlsx ...")
    df = pd.read_excel(INPUT)
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # ── 2. Parse European number format ─────────────────────────────────────────
    print("\n2. Parsing European number format ...")
    n_parsed = 0
    for col in df.columns:
        if col == "Date":
            continue
        if df[col].dtype == object:
            before_nan = df[col].isna().sum()
            df[col] = df[col].apply(parse_tr_number)
            after_nan = df[col].isna().sum()
            n_values = df[col].notna().sum()
            print(f"   {col}: object->float, {n_values} non-NaN "
                  f"(NaN: {before_nan}->{after_nan})")
            n_parsed += 1
        else:
            # Already numeric — verify it's float
            df[col] = df[col].astype(float)

    if n_parsed == 0:
        print("   No object columns found — all columns already numeric")
    else:
        print(f"   Parsed {n_parsed} column(s)")

    # ── 2b. Rename Turkish characters to ASCII ────────────────────────────────────
    # Turkish dotless i (ı) and other characters cause confusion in downstream
    # code (KeyError: "altin" != "altın"). Rename them at the source.
    print("\n2b. Renaming Turkish characters to ASCII ...")
    TR_TO_ASCII = {
        "\u0131": "i",    # dotless i → i
        "\u0130": "I",    # dotted I → I
        "\u015f": "s",    # s-cedilla → s
        "\u015e": "S",    # S-cedilla → S
        "\u011f": "g",    # g-breve → g
        "\u011e": "G",    # G-breve → G
        "\u00e7": "c",    # c-cedilla → c
        "\u00c7": "C",    # C-cedilla → C
        "\u00f6": "o",    # o-umlaut → o
        "\u00d6": "O",    # O-umlaut → O
        "\u00fc": "u",    # u-umlaut → u
        "\u00dc": "U",    # U-umlaut → U
    }
    renamed = 0
    new_columns = {}
    for col in df.columns:
        new_col = col
        for tr_char, ascii_char in TR_TO_ASCII.items():
            new_col = new_col.replace(tr_char, ascii_char)
        if new_col != col:
            renamed += 1
            print(f"   renamed column {renamed}: -> {new_col}")
        new_columns[col] = new_col
    df.rename(columns=new_columns, inplace=True)
    # Update GDP_COL reference in case it was renamed
    gdp_col = GDP_COL
    for old, new in new_columns.items():
        if old == gdp_col:
            gdp_col = new
    print(f"   Renamed {renamed} column(s)")

    # ── 3. Drop broken variables ────────────────────────────────────────────────
    print("\n3. Dropping broken variables ...")
    for col in DROP_COLS:
        if col in df.columns:
            n_vals = df[col].notna().sum()
            df.drop(columns=[col], inplace=True)
            print(f"   Dropped '{col}' ({n_vals} obs — broken)")
    print(f"   Shape after drops: {df.shape}")

    # ── 4. Quarterly GDP → Q-end months ─────────────────────────────────────────
    print("\n4. Quarterly GDP placement ...")
    gdp_vals = df[df[GDP_COL].notna()][["Date", GDP_COL]].copy()
    print(f"   {GDP_COL}: {len(gdp_vals)} non-NaN observations")
    print(f"   First 5 GDP dates: {gdp_vals['Date'].dt.date.head(5).tolist()}")
    print(f"   Last  3 GDP dates: {gdp_vals['Date'].dt.date.tail(3).tolist()}")

    # Check: all GDP dates fall in Q-end months
    gdp_months = gdp_vals["Date"].dt.month.unique()
    expected = set(Q_END_MONTHS)
    if not set(gdp_months).issubset(expected):
        non_qend = set(gdp_months) - expected
        print(f"   *** WARNING: GDP has values in non-Q-end months: {non_qend}")
        print(f"   *** GDP will be shifted to Q-end months")
        # Shift: for each non-Q-end GDP value, move to nearest future Q-end
        # This handles the case where GDP is at Q-beginning months
        new_gdp = pd.Series(np.nan, index=df.index, name=GDP_COL)
        for _, row in gdp_vals.iterrows():
            d = row["Date"]
            # Find the Q-end month: round up to next 3, 6, 9, or 12
            month = d.month
            q_end_month = ((month - 1) // 3 + 1) * 3
            q_end_date = pd.Timestamp(year=d.year, month=q_end_month, day=1)
            idx = df[df["Date"] == q_end_date].index
            if len(idx) > 0:
                new_gdp.loc[idx[0]] = row[GDP_COL]
        df[GDP_COL] = new_gdp
        print(f"   GDP shifted to Q-end months. New count: {df[GDP_COL].notna().sum()}")
    else:
        print(f"   GDP already at Q-end months: {gdp_months.tolist()} — no shift needed")

    # ── 5. Hard-fail range checks ───────────────────────────────────────────────
    print("\n5. Hard-fail range checks ...")
    checks = []

    # Macro fundamentals
    if "unemp_rate" in df.columns:
        checks.append(("unemp_rate", 0, 50))
    if "emp_rate" in df.columns:
        checks.append(("emp_rate", 0, 100))
    if "CPI_SA" in df.columns:
        checks.append(("CPI_SA", 0, 1e9))
    if "PPI" in df.columns:
        checks.append(("PPI", 0, 1e9))
    if "IPI_SA" in df.columns:
        checks.append(("IPI_SA", 0, 1e6))
    if "NGDPRSAXDCTRQ" in df.columns:
        checks.append(("NGDPRSAXDCTRQ", 0, 1e9))
    if "REAL_GDP_I" in df.columns:
        checks.append(("REAL_GDP_I", 0, 1e6))
    if "USD_TRY_AVG" in df.columns:
        checks.append(("USD_TRY_AVG", 0, 1e3))
    if "1week-repo" in df.columns:
        checks.append(("1week-repo", 0, 200))
    if "BIST100" in df.columns:
        checks.append(("BIST100", 0, 1e7))

    # Monetary / credit
    if "M3" in df.columns:
        checks.append(("M3", 0, 1e13))
    if "LOANS" in df.columns:
        checks.append(("LOANS", 0, 1e13))
    if "tourist" in df.columns:
        checks.append(("tourist", 0, 1e8))

    # Reserves
    for res_col in ["doviz_rezerv_var", "resmi_rezerv_var"]:
        if res_col in df.columns:
            checks.append((res_col, 0, 1e9))
    # Turkish character column name — separate handling
    for c in df.columns:
        if "rezerv_var" in c and c not in ["doviz_rezerv_var", "resmi_rezerv_var"]:
            checks.append((c, 0, 1e9))

    passed = 0
    failed = 0
    for col, lo, hi in checks:
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"   SKIP  {col}: no non-NaN observations")
            continue
        vmin, vmax = vals.min(), vals.max()
        ok = (vmin >= lo) and (vmax <= hi)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        # Safe print: some TR column names have non-ASCII chars
        try:
            print(f"   {status}  {col}: min={vmin:.4f}, max={vmax:.4f} "
                  f"(expected [{lo}, {hi}])")
        except UnicodeEncodeError:
            print(f"   {status}  (col): min={vmin:.4f}, max={vmax:.4f} "
                  f"(expected [{lo}, {hi}])")

    print(f"\n   {passed} passed, {failed} failed")

    if failed > 0:
        raise AssertionError(
            f"{failed} range checks FAILED. "
            "Data may have scale corruption. Halting pipeline."
        )

    # ── 6. Save ─────────────────────────────────────────────────────────────────
    print("\n6. Saving ...")
    df.to_excel(OUTPUT, index=False)
    print(f"   Written: {OUTPUT}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # ── 7. Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Variables: {len(df.columns) - 1}")
    print(f"  Rows: {len(df)}")
    print(f"  European number parsing: {n_parsed} column(s)")
    print(f"  Dropped: {len(DROP_COLS)} broken column(s): {DROP_COLS}")
    print(f"  Range checks: {passed} passed, {failed} failed")
    return df


if __name__ == "__main__":
    build_raw_monthly()
