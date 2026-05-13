import pandas as pd
import numpy as np
from collections import Counter

BASE     = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
MD_PATH  = BASE + "/fred-md.xlsx"
QD_PATH  = BASE + "/fred-qd.xlsx"
UM_PATH  = BASE + "/us_master_monthly.xlsx"
OUT_PATH = BASE + "/data_raw_monthl.xlsx"

USER_MONTHLY_KEEP = {
    'BOPGSTB':              2,
    'CFNAI':                1,
    'CSCICP03USM665S':      2,
    'DSPIC96':              5,
    'DTWEXBGS_monthly_avg': 5,
    'DTWEXM_monthly_avg':   5,
    'GACDFSA066MSFRBPHI':   1,
    'GACDISA066MSFRBNY':    1,
    'HSN1F':                4,
    'NFCI_monthly_avg':     1,
    'PCE':                  5,
    'PCEC96':               5,
    'PI':                   5,
    'PPIFIS':               6,
    'RSAFS':                5,
    'TCU':                  2,
    'TLOFCONS':             5,
    'TTLCONS':              5,
    'USPMI_clean':          1,
    'US_services_PMI_clean':1,
}

USER_QUARTERLY_KEEP = {
    'EXPGS': 5,
    'IMPGS': 5,
}

# ── STEP 1: FRED-MD ──────────────────────────────────────────────────────────
print("Step 1: Loading FRED-MD...")
md_raw = pd.read_excel(MD_PATH, sheet_name='Worksheet', header=0)
# Row 0 = tcode row (sasdate cell contains "Transform:")
md_tcodes = {col: float(md_raw.iloc[0][col]) for col in md_raw.columns[1:]}
# Rows 1+ = data
md_data = md_raw.iloc[1:].copy()
md_data['sasdate'] = pd.to_datetime(md_data['sasdate'])
md_data = md_data.rename(columns={'sasdate': 'date'}).set_index('date')
for col in md_data.columns:
    md_data[col] = pd.to_numeric(md_data[col], errors='coerce')
print(f"  {len(md_data.columns)} variables | {len(md_data)} monthly rows | {md_data.index.min().date()} to {md_data.index.max().date()}")

# ── STEP 2: FRED-QD ──────────────────────────────────────────────────────────
print("Step 2: Loading FRED-QD...")
qd_raw = pd.read_excel(QD_PATH, sheet_name='Worksheet', header=0)
# Row 0 = factors (skip), Row 1 = tcodes
qd_tcodes_all = {col: float(qd_raw.iloc[1][col]) for col in qd_raw.columns[1:]}
# Rows 2+ = data
qd_data = qd_raw.iloc[2:].copy()
qd_data['sasdate'] = pd.to_datetime(qd_data['sasdate'])
qd_data = qd_data.rename(columns={'sasdate': 'date'}).set_index('date')
for col in qd_data.columns:
    qd_data[col] = pd.to_numeric(qd_data[col], errors='coerce')

# Keep only QD cols not already in MD and not overridden by user monthly
md_col_set = set(md_data.columns)
user_monthly_set = set(USER_MONTHLY_KEEP.keys())
qd_only_cols = [c for c in qd_data.columns if c not in md_col_set and c not in user_monthly_set]
qd_only_data   = qd_data[qd_only_cols]
qd_only_tcodes = {col: qd_tcodes_all[col] for col in qd_only_cols}
print(f"  {len(qd_only_cols)} QD-unique variables | {len(qd_data)} quarterly rows | {qd_data.index.min().date()} to {qd_data.index.max().date()}")

# ── STEP 3: User data ────────────────────────────────────────────────────────
print("Step 3: Loading user data (us_master_monthly.xlsx)...")
um_raw = pd.read_excel(UM_PATH)
um_raw['Date'] = pd.to_datetime(um_raw['Date'])
um_raw = um_raw.rename(columns={'Date': 'date'}).set_index('date')

missing = [c for c in list(USER_MONTHLY_KEEP) + list(USER_QUARTERLY_KEEP) if c not in um_raw.columns]
if missing:
    print(f"  WARNING — missing from user file: {missing}")

um_monthly  = um_raw[[c for c in USER_MONTHLY_KEEP  if c in um_raw.columns]]
um_quarterly = um_raw[[c for c in USER_QUARTERLY_KEEP if c in um_raw.columns]]
print(f"  {len(um_monthly.columns)} user monthly variables")
print(f"  {len(um_quarterly.columns)} user quarterly variables")

# ── STEP 4: Monthly date index ───────────────────────────────────────────────
print("Step 4: Building monthly date index 1919-01 to 2026-04...")
date_index = pd.date_range(start='1919-01-01', end='2026-04-01', freq='MS')
combined = pd.DataFrame(index=date_index)
combined.index.name = 'date'

# ── STEP 5: Merge ────────────────────────────────────────────────────────────
print("Step 5: Merging all sources...")
combined = combined.join(md_data,       how='left')
combined = combined.join(qd_only_data,  how='left')
combined = combined.join(um_monthly,    how='left')
combined = combined.join(um_quarterly,  how='left')
print(f"  Combined: {combined.shape[0]} rows × {combined.shape[1]} columns")

# ── STEP 6: Tcode dictionary ─────────────────────────────────────────────────
print("Step 6: Assembling tcode dictionary...")
all_tcodes = {}
for col in md_data.columns:
    all_tcodes[col] = md_tcodes[col]
for col in qd_only_cols:
    if col in combined.columns:
        all_tcodes[col] = qd_only_tcodes[col]
for col, tc in USER_MONTHLY_KEEP.items():
    if col in combined.columns:
        all_tcodes[col] = float(tc)
for col, tc in USER_QUARTERLY_KEEP.items():
    if col in combined.columns:
        all_tcodes[col] = float(tc)

missing_tc = [c for c in combined.columns if c not in all_tcodes]
if missing_tc:
    print(f"  WARNING — no tcode for: {missing_tc}")
else:
    print(f"  All {len(all_tcodes)} variables have tcodes.")

# ── STEP 7: Assemble final DataFrame ─────────────────────────────────────────
print("Step 7: Assembling final DataFrame...")
combined_reset = combined.reset_index()
combined_reset['date'] = combined_reset['date'].dt.strftime('%Y-%m-%d')

tcode_vals = ['tcode'] + [all_tcodes.get(col, np.nan) for col in combined.columns]
tcode_row  = pd.DataFrame([tcode_vals], columns=['date'] + list(combined.columns))

final_df = pd.concat([tcode_row, combined_reset], ignore_index=True)
print(f"  Final shape: {final_df.shape}  (1 tcode row + {len(combined)} data rows)")

# ── STEP 8: Write Excel ──────────────────────────────────────────────────────
print(f"Step 8: Writing to {OUT_PATH} ...")
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name='raw_data', index=False)
print("  File written.")

# ── STEP 9: VALIDATION ───────────────────────────────────────────────────────
print("\n" + "="*62)
print("STEP 9 — VALIDATION REPORT")
print("="*62)

data = combined.copy()
PASS = []
FAIL = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    (PASS if condition else FAIL).append(name)
    mark = "OK" if condition else "!!"
    print(f"  {mark} {name}: {detail}")

# Shape
check("Row count",    len(data) == 1288,          f"{len(data)} rows (expected 1288)")
check("Column count", len(data.columns) == 296,   f"{len(data.columns)} vars (expected 296)")

# No duplicates
dupes = [c for c in data.columns if list(data.columns).count(c) > 1]
check("No duplicate columns", len(dupes) == 0, f"{dupes if dupes else 'none'}")

# All tcodes valid
valid_tc = {1.0, 2.0, 4.0, 5.0, 6.0, 7.0}
bad_tc = [(c, all_tcodes[c]) for c in all_tcodes if all_tcodes[c] not in valid_tc]
check("All tcodes valid", len(bad_tc) == 0, f"invalid: {bad_tc if bad_tc else 'none'}")

# Quarterly variables in correct months only
for qvar in ['GDPC1', 'EXPGS', 'IMPGS']:
    s = data[qvar].dropna()
    months = sorted(s.index.month.unique().tolist())
    check(f"{qvar} quarterly months", months == [3,6,9,12], f"months={months}")

# TCU monthly (all 12 months)
tcu_months = sorted(data['TCU'].dropna().index.month.unique().tolist())
check("TCU monthly (all 12 months)", len(tcu_months) == 12, f"months={tcu_months}")

# Value range sanity checks (raw levels)
range_checks = {
    'UNRATE':   (0, 30),
    'FEDFUNDS': (0, 25),
    'CPIAUCSL': (1, 400),
    'GDPC1':    (1000, 40000000),
    'TCU':      (50, 100),
    'CFNAI':    (-20, 10),
}
for col, (lo, hi) in range_checks.items():
    s = data[col].dropna()
    smin, smax = float(s.min()), float(s.max())
    ok = smin >= lo and smax <= hi
    check(f"{col} range [{lo},{hi}]", ok, f"min={smin:.3f} max={smax:.3f}")
    # HARD FAIL: scale corruption in source files (e.g. 2.04 -> 204) is exactly
    # what these range checks catch. Letting them slip through silently lost a
    # week of work upstream. Halt the pipeline immediately if any check fails.
    assert ok, (
        f"SCALE SANITY CHECK FAILED for {col}: min={smin:.3f}, max={smax:.3f}, "
        f"expected within [{lo}, {hi}]. This usually means the source xlsx "
        f"has been silently rescaled. Halt and investigate before proceeding."
    )

# No variables with 0 observations
non_null = data.notna().sum()
check("No zero-obs variables", (non_null == 0).sum() == 0,
      f"{(non_null==0).sum()} empty vars")

# Coverage summary
print(f"\n  Coverage summary:")
print(f"    Min obs: {non_null.min()} ({non_null.idxmin()})")
print(f"    Max obs: {non_null.max()} ({non_null.idxmax()})")
print(f"    Median:  {int(non_null.median())} obs")
print(f"    <200 obs: {(non_null < 200).sum()} variables")

# Tcode distribution
tc_dist = Counter([all_tcodes[c] for c in combined.columns if c in all_tcodes])
labels = {1:'level', 2:'first diff', 4:'log level', 5:'log first diff',
          6:'log second diff', 7:'delta pct chg'}
print(f"\n  Tcode distribution:")
for tc, cnt in sorted(tc_dist.items()):
    print(f"    tcode={int(tc)} ({labels.get(tc,'?')}): {cnt} vars")

# Source breakdown
print(f"\n  Source breakdown:")
print(f"    FRED-MD monthly:         {len(md_data.columns)}")
print(f"    FRED-QD unique quarterly:{len(qd_only_cols)}")
print(f"    User unique monthly:     {len(um_monthly.columns)}")
print(f"    User unique quarterly:   {len(um_quarterly.columns)}")
print(f"    TOTAL:                   {len(combined.columns)}")

print(f"\n  Results: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print(f"  FAILED CHECKS: {FAIL}")
else:
    print("  All checks passed. Dataset is clean.")
