"""
build_metadata_tr.py
====================
Pipeline B — Turkey: Build meta_data_tr.csv with frequency, publication lags,
and economic block assignments for every variable.

Uses the dictionary as the authoritative source for variables that have entries.
For variables in the monthly file without dictionary entries, infers frequency
from the data and assigns conservative defaults.

References:
    - McCracken-Ng (2020) FRED-MD variable classification
    - Bok et al. (2018) Bańbura-style nowcasting framework
    - Bańbura-Modugno (2014) block structure for DFM factor extraction

Run order: 2 (after build_raw_data_tr.py)

Input:
    turkey_data/turkey_data_dictionary.xlsx
    turkey_data/data_raw_monthly_tr.xlsx   (from Step 1)

Output:
    turkey_data/meta_data_tr.csv
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(BASE, "turkey_data_dictionary.xlsx")
RAW_PATH = os.path.join(BASE, "data_raw_monthly_tr.xlsx")
OUTPUT = os.path.join(BASE, "meta_data_tr.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Publication lag: text -> months
# ═══════════════════════════════════════════════════════════════════════════════

def release_lag_to_months(lag_str):
    """
    Convert dictionary Release Lag text to integer months_lag.

    Conservative rule: floor to the slower side when ambiguous.
    GDP at 60 days -> 2 months (Bok convention).
    IPI at 40 days -> 2 months (conservative, could be 1).
    CPI at 3 days -> 0 months (available within month).
    """
    if pd.isna(lag_str):
        return 1  # default: available next month

    lag_str = str(lag_str).lower().strip()

    # Extract numbers
    import re
    nums = re.findall(r'\d+', lag_str)
    days = int(nums[0]) if nums else 30

    # Convert days to months
    if days <= 15:
        return 0
    elif days <= 45:
        return 1
    else:
        return 2


# ═══════════════════════════════════════════════════════════════════════════════
# Economic Category -> Bok block mapping
# ═══════════════════════════════════════════════════════════════════════════════

def map_category_to_blocks(category):
    """
    Map Turkish dictionary Economic Category to Bok (2018) block flags.
    Returns (block_s, block_r, block_l) as (int, int, int).
    block_g is always 1 for all variables.

    Block assignment mirrors US pipeline logic (build_metadata.py):
      - Financial / Monetary / Liquidity → Global only (block_r=0, block_l=0)
        Financial prices (rates, FX, stocks, reserves) have no seasonal
        cycle and do not belong in the real-activity block.
      - External Sector → Real (trade volumes have real-economy seasonality)
      - Real Activity / Real Sector / Housing / Tourism → Real
      - Labor Market → Labor
      - Survey / Sentiment → Soft
    """
    if pd.isna(category):
        return (0, 1, 0)  # default: real activity

    cat = str(category).lower().strip()

    # Global only: financial, monetary, liquidity (no real-economy block)
    global_keywords = ["financial", "monetary", "liquidity"]
    if any(k in cat for k in global_keywords):
        return (0, 0, 0)

    # Soft / Survey
    if any(k in cat for k in ["survey", "soft", "sentiment", "confidence"]):
        return (1, 0, 0)

    # Labor
    if "labor" in cat:
        return (0, 0, 1)

    # Real activity / output / prices / external trade / fiscal / tourism / housing
    real_keywords = [
        "real activity", "real sector", "output", "production",
        "price", "external", "fiscal", "tourism", "housing",
    ]
    if any(k in cat for k in real_keywords):
        return (0, 1, 0)

    # Default: real
    return (0, 1, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Frequency assignment
# ═══════════════════════════════════════════════════════════════════════════════

def infer_frequency(series, col_name):
    """Infer frequency from data pattern when dictionary has no entry."""
    non_nan = series.dropna()
    if len(non_nan) < 2:
        return "m"  # default

    # Check for quarterly pattern: values only at months 3, 6, 9, 12
    months_with_data = set(series.dropna().index.month)
    if months_with_data.issubset({3, 6, 9, 12}) and len(months_with_data) <= 4:
        return "q"

    return "m"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Build
# ═══════════════════════════════════════════════════════════════════════════════

def build_metadata():
    print("=" * 60)
    print("BUILD METADATA — TURKEY")
    print("=" * 60)

    # 1. Load dictionary
    print("\n1. Loading dictionary ...")
    tr_dict = pd.read_excel(DICT_PATH)
    # Rename Turkish characters to ASCII in Variable column
    TR_TO_ASCII = {"\u0131": "i", "\u0130": "I", "\u015f": "s", "\u015e": "S",
                   "\u011f": "g", "\u011e": "G", "\u00e7": "c", "\u00c7": "C",
                   "\u00f6": "o", "\u00d6": "O", "\u00fc": "u", "\u00dc": "U"}
    for tr_c, ascii_c in TR_TO_ASCII.items():
        tr_dict["Variable"] = tr_dict["Variable"].str.replace(tr_c, ascii_c, regex=False)
        tr_dict["Actual Name"] = tr_dict["Actual Name"].astype(str).str.replace(tr_c, ascii_c, regex=False)
    print(f"   Dictionary: {len(tr_dict)} entries")

    # 2. Load raw data to get actual column inventory
    print("\n2. Loading raw data for column inventory ...")
    raw = pd.read_excel(RAW_PATH)
    data_cols = [c for c in raw.columns if c != "Date"]
    print(f"   Raw data: {len(data_cols)} variables (excl. Date)")

    # 3. Build dictionary lookup (Variable -> row)
    dict_lookup = {}
    for _, row in tr_dict.iterrows():
        var = str(row["Variable"]).strip()
        dict_lookup[var.lower()] = row

    # 4. Build metadata rows
    print("\n3. Building metadata ...")
    rows = []
    orphan_count = 0

    for col in data_cols:
        col_lower = col.lower().strip()

        # Try to find in dictionary (exact match first, then fuzzy)
        dict_row = dict_lookup.get(col_lower)

        if dict_row is None:
            # Try fuzzy matching on underscore-delimited tokens
            for dict_var, drow in dict_lookup.items():
                # Match if column name equals dict var, or if stripping
                # trailing _suffixes matches (e.g. unemp_num -> unempl_num)
                if col_lower == dict_var:
                    dict_row = drow
                    break
                # Check if base name matches after stripping known suffixes
                col_base = col_lower
                for suffix in ["_sa", "_nsa", "_i", "_r", "_var", "_avg", "_sum"]:
                    if col_base.endswith(suffix):
                        col_base = col_base[:-len(suffix)]
                        break
                if col_base == dict_var or dict_var == col_base:
                    dict_row = drow
                    break

        if dict_row is not None:
            # Authoritative: use dictionary values
            name = str(dict_row.get("Actual Name", col))
            if pd.isna(dict_row.get("Actual Name")) or str(dict_row["Actual Name"]).strip() == "nan":
                name = col

            freq_raw = str(dict_row.get("Frequency", "")).strip().lower()
            if "quarterly" in freq_raw:
                freq = "q"
            elif "weekly" in freq_raw:
                freq = "w"
            else:
                freq = "m"

            lag = release_lag_to_months(dict_row.get("Release Lag"))

            category = dict_row.get("Economic Category")
            bs, br, bl = map_category_to_blocks(category)

        else:
            # Orphan: no dictionary entry — infer from data
            orphan_count += 1
            name = col
            series_vals = raw[col].copy()
            series_vals.index = pd.to_datetime(raw["Date"])
            freq = infer_frequency(series_vals, col)

            # Conservative defaults for orphans
            lag = 0 if freq == "w" else 1
            # auto_prod, total_prod -> real activity
            # USD_TRY_AVG -> financial (real block)
            # house_sales_sa etc. -> real activity / housing
            bs, br, bl = (0, 1, 0)

        rows.append({
            "series": col_lower,
            "name": name,
            "freq": freq,
            "block_g": 1,
            "block_s": bs,
            "block_r": br,
            "block_l": bl,
            "months_lag": lag,
        })

    meta = pd.DataFrame(rows)

    # 5. Override GDP lag to 2 (Bok convention)
    gdp_mask = meta["series"].str.contains("ngdprsaxdctrq", case=False)
    if gdp_mask.any():
        meta.loc[gdp_mask, "months_lag"] = 2
        print(f"   GDP (NGDPRSAXDCTRQ): lag set to 2 (Bok convention)")

    # 6. Heuristic overrides for common naming patterns
    for idx, row in meta.iterrows():
        s = row["series"]
        # Labor market variables (employment/unemployment)
        if any(kw in s for kw in ["empl_", "unempl_", "unemp_", "emp_rate", "emp_num"]):
            meta.at[idx, "block_l"] = 1
        # Survey/confidence variables
        if any(kw in s for kw in ["conf", "consu_i", "deposit_i", "bus_"]):
            meta.at[idx, "block_s"] = 1
        # Financial / monetary — force to global only (no real-economy block)
        fin_kw = ["usd_try", "loan", "fin_acc", "bist", "reer", "reserv",
                  "m3", "credit", "1week", "bond", "bill"]
        if any(kw in s for kw in fin_kw):
            if row["block_r"] == 1:
                meta.at[idx, "block_r"] = 0
    print("   Applied heuristic block overrides")

    # 6. Verify consistency: every data column has a metadata row
    meta_set = set(meta["series"])
    data_set = {c.lower() for c in data_cols}
    missing = data_set - meta_set
    if missing:
        print(f"   WARNING: {len(missing)} data columns missing from metadata: {missing}")
    else:
        print(f"   All {len(data_cols)} data columns have metadata rows")

    # 7. Summary statistics
    print(f"\n   Orphans (no dictionary entry): {orphan_count}")
    print(f"   Frequency:     {meta['freq'].value_counts().to_dict()}")
    print(f"   Lag:           {meta['months_lag'].value_counts().sort_index().to_dict()}")
    print(f"   block_s (soft): {meta['block_s'].sum()}")
    print(f"   block_r (real): {meta['block_r'].sum()}")
    print(f"   block_l (labor): {meta['block_l'].sum()}")

    # 8. Save
    meta.to_csv(OUTPUT, index=False)
    print(f"\n   Written: {OUTPUT}")
    print(f"   Rows: {len(meta)}")

    return meta


if __name__ == "__main__":
    build_metadata()
