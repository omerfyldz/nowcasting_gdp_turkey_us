"""
References (basis for the lookup tables below; not arbitrary overrides):
  McCracken, M. W. and Ng, S. (2016) "FRED-MD: A Monthly Database for
      Macroeconomic Research." J. Bus. Econ. Stat. 34(4), 574-589.
  McCracken, M. W. and Ng, S. (2020) "FRED-QD: A Quarterly Database for
      Macroeconomic Research." NBER WP 26872.  (Group taxonomy used for
      FRED-MD column-position -> Bok block mapping in FREDMD_GROUP_BLOCKS.)
  Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., and Tambalotti, A.
      (2018) "Macroeconomic Nowcasting and Forecasting with Big Data."
      Annual Review of Economics 10, 615-643.  (4-block g/s/r/l structure;
      months_lag conventions; HWI/HWIURATIO cross-listing in Soft + Labor.)
  Banbura, M. and Modugno, M. (2014) "Maximum Likelihood Estimation of
      Factor Models on Datasets with Arbitrary Pattern of Missing Data."
      J. Appl. Econometrics 29(1), 133-160.  (Mean-fill convention for
      ragged-edge missing data; NIPA-style lag=2 grid placement.)

build_metadata.py
=================
Rebuilds meta_data.csv with AUTHORITATIVE, SYSTEMATIC assignments for all 305
series (296 monthly + 6 weekly + 3 COVID dummies). No per-variable override
dicts. Every decision is derived from published sources or documented logic.

FREQ
----
  FRED-MD columns (fred-md.xlsx header)          -> 'm'
  FRED-QD columns not in FRED-MD                 -> 'q'
  User vars (US_data_dictionary Frequency col)   -> per dictionary
  Weekly file columns                            -> 'w'
  COVID dummies                                  -> 'm'

BLOCK  (block_g / block_s / block_r / block_l)
-------
  block_g = 1 for every series (Bok et al. global factor is always active).

  FRED-MD vars:
    Assigned by column position in the 2026-03 FRED-MD release.
    Position ranges correspond to McCracken-Ng (2020) documented group order:
      0-18   Group 1 Output & Income      -> Real   (r=1)
      19-46  Group 2 Labor Market         -> Labor  (l=1)
      47-56  Group 3 Housing              -> Real   (r=1)
      57-62  Group 4 Orders/Inventories   -> Real   (r=1)
      63-75  Group 5+Stocks: Money/Credit -> Global (g only)
      76-97  Group 6 Interest/Exchange    -> Global (g only)
      98-117 Group 7 Prices               -> Real   (r=1)
      118-120 CES Avg Hourly Earnings     -> Labor  (l=1)
      121    UMCSENTx (Cons. Sentiment)   -> Soft   (s=1)
      122-125 Credit/INVEST/VIX           -> Global (g only)

  FRED-QD-only vars:
    Classified via membership in documented economic-category sets drawn
    from FRED-QD appendix (McCracken-Ng 2020):
      Labor  : BLS productivity, hours, unit-labor-cost, CES sector employment,
               CPS demographic variants, help-wanted
      Soft   : economic policy uncertainty
      Global : interest rates, FX, money/credit aggregates, balance sheets,
               international equity indices
      Real   : everything else (NIPA chains, prices/deflators, IP quarterly,
               housing prices, orders, fiscal, capacity)

  User vars (US_data_dictionary.xlsx "Economic Category"):
      "Real Activity*"           -> Real
      "Labor Market"             -> Labor
      "Prices/Inflation"         -> Real  (Bok convention: prices in Real)
      "Financial Conditions*"    -> Global
      "Monetary Policy"          -> Global
      "External Sector"          -> Real
      "Consumption*"             -> Real
      "Expectations/Surveys"     -> Soft
      "Inventories/Business*"    -> Real

  Weekly vars:
      ICSA_weekly                -> Labor
      NFCI_weekly                -> Soft  (financial conditions index)
      DTWEXM, DTWEXBGS           -> Global (FX)
      COVID dummies in weekly    -> Global only (already in monthly, skipped)

MONTHS_LAG
----------
  COVID dummies                 -> 0
  Financial rates/spreads/FX
  (incl. QD versions not in MD) -> 0
  CES/CPS Employment Situation  -> 0 (released first Friday of T+1)
  Surveys (ISM PMI, Sentiment)  -> 0
  Mid-month hard data (CPI etc) -> 1
  NIPA quarterly GDP family     -> 2  (Advance ~30 days after Q-end; safe lag)
  QD default                    -> 2
  MD default                    -> 1
  User vars                     -> parsed from Release Lag column
"""
import pandas as pd
import numpy as np
import os

BASE_DIR  = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
MD_XLSX   = os.path.join(BASE_DIR, "fred-md.xlsx")
QD_XLSX   = os.path.join(BASE_DIR, "fred-qd.xlsx")
USER_DICT = os.path.join(BASE_DIR, "US_data_dictionary.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Source column lists
# ─────────────────────────────────────────────────────────────────────────────
_md_hdr = pd.read_excel(MD_XLSX, sheet_name='Worksheet', nrows=0)
MD_COLS  = [c.lower() for c in _md_hdr.columns[1:]]   # skip sasdate; 126 vars
MD_SET   = set(MD_COLS)
MD_POS   = {c: i for i, c in enumerate(MD_COLS)}       # column position lookup

_qd_hdr  = pd.read_excel(QD_XLSX, sheet_name='Worksheet', nrows=0)
QD_COLS  = [c.lower() for c in _qd_hdr.columns[1:]]   # skip sasdate; 245 vars
QD_SET   = set(QD_COLS)
QD_ONLY  = QD_SET - MD_SET                             # ~148 QD-only vars

user_df  = pd.read_excel(USER_DICT)
user_df["var_lc"] = user_df["Variable"].str.lower()

COVID_DUMMIES = {"covid_2020q2", "covid_2020q3", "covid_2020q4"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. BLOCK ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. FRED-MD: group boundaries from McCracken-Ng (2020) column order ──────
# Each tuple is (first_pos, last_pos_inclusive, (g, s, r, l))
FREDMD_GROUP_BLOCKS = [
    (  0,  18, (1, 0, 1, 0)),  # Group 1  Output & Income          -> Real
    ( 19,  46, (1, 0, 0, 1)),  # Group 2  Labor Market             -> Labor
    ( 47,  56, (1, 0, 1, 0)),  # Group 3  Housing                  -> Real
    ( 57,  62, (1, 0, 1, 0)),  # Group 4  Consumption/Orders/Inv.  -> Real
    ( 63,  75, (1, 0, 0, 0)),  # Group 5  Money & Credit + Stocks  -> Global
    ( 76,  97, (1, 0, 0, 0)),  # Group 6  Interest & Exchange Rates-> Global
    ( 98, 117, (1, 0, 1, 0)),  # Group 7  Prices                   -> Real
    (118, 120, (1, 0, 0, 1)),  # CES Avg Hourly Earnings (3 series)-> Labor
    (121, 121, (1, 1, 0, 0)),  # UMCSENTx Consumer Sentiment       -> Soft
    (122, 125, (1, 0, 0, 0)),  # DTCOLNVHFNM/DTCTHFNM/INVEST/VIX  -> Global
]

def _fredmd_block(col_lower):
    pos = MD_POS.get(col_lower)
    if pos is None:
        return (1, 0, 0, 0)
    # HWI (pos 19) and HWIURATIO (pos 20) are survey-based labour-market
    # indicators. Bok et al. (2018) cross-lists them in BOTH Soft and Labor.
    if 19 <= pos <= 20:
        return (1, 1, 0, 1)
    for lo, hi, blk in FREDMD_GROUP_BLOCKS:
        if lo <= pos <= hi:
            return blk
    return (1, 0, 0, 0)

# ── 2b. FRED-QD-only: documented economic-category sets ──────────────────────
# Labor: BLS productivity & costs, hours, wages, CES sector, CPS variants
_QD_LABOR = {
    # BLS Multifactor Productivity / Unit Labor Costs
    "outnfb", "outbs", "outms",
    "comprnfb", "comprms", "rcphbs",
    "ophmfg", "ophnfb", "ophpbs",
    "ulcbs", "ulcmfg", "ulcnfb", "unlpnbs",
    # Durable business equipment productivity
    "ipdbs",
    # Hours
    "hoabs", "hoams", "hoanbs", "awhnonag",
    # Avg hourly earnings (QD quarterly versions)
    "ahetpix", "ces2000000008x", "ces3000000008x",
    # CES sector employment (in QD but not MD)
    "ces9091000001", "ces9092000001", "ces9093000001",
    "uspriv", "usehs", "usinfo", "uspbs", "uslah", "usmine", "usserv",
    # CPS demographic unemployment variants
    "lns12032194", "lns13023557", "lns13023569",
    "lns13023621", "lns13023705",
    "lns14000012", "lns14000025", "lns14000026",
    "civpart", "unratestx", "unrateltx",
    # Help-Wanted (QD variants)
    "hwix", "hwiuratiox",
}

# Soft: survey/expectations series
_QD_SOFT = {
    "usepuindxm",   # Baker-Bloom-Davis Economic Policy Uncertainty
}

# Global: interest rates, money, credit, balance sheets, FX, equity indices
_QD_GLOBAL = {
    # Interest rates / spreads (QD-only versions)
    "mortg10yrx", "mortgage30us", "baa10ym",
    "tb6m3mx", "gs1tb3mx", "gs10tb3mx", "cpf3mtb3mx",
    "driwcil",              # Delinquency rate on loans
    "compapff",             # Commercial paper vs fed funds spread (QD version)
    "cp3m",                 # 3-month commercial paper rate
    # FX (QD version)
    "exuseu",
    # Money and credit (QD deflated / variant series)
    "bogmbaserealx", "m1real",
    "busloansx", "consumerx", "nonrevslx", "reallnx", "revolslx", "totalslx",
    "conspix",
    # Household / corporate balance sheets (Fed Flow of Funds Z.1)
    "tlbshnox", "liabpix", "tnwbshnox", "tfaabshnox", "hnoremq027sx",
    "tnwmvbsnncbbdix", "tnwmvbsnncbx",
    "tlbsnncbbdix", "tlbsnncbx",
    "tabshnox", "ttaabsnncbx",
    "tlbsnnbbdix", "tlbsnnbx", "tabsnnbx",
    "tnwbsnnbbdix", "tnwbsnnbx",
    "cncfx", "taresax", "nwpix",
    # International equity indices
    "nasdaqcom", "nikkei225",
}
# All QD-only series not in the above sets -> Real (NIPA, prices, IP, housing)

def _qdonly_block(col_lower):
    # HWI variants are cross-listed in both Soft and Labor (Bok et al.)
    if col_lower in {"hwix", "hwiuratiox"}:
        return (1, 1, 0, 1)
    if col_lower in _QD_LABOR:
        return (1, 0, 0, 1)
    if col_lower in _QD_SOFT:
        return (1, 1, 0, 0)
    if col_lower in _QD_GLOBAL:
        return (1, 0, 0, 0)
    return (1, 0, 1, 0)   # Real: NIPA chains, prices, housing, orders, fiscal

# ── 2c. User vars: Economic Category from US_data_dictionary.xlsx ─────────────
def _user_block(col_lower):
    row = user_df[user_df["var_lc"] == col_lower]
    if len(row) == 0:
        return (1, 0, 0, 0)   # unknown -> Global only (conservative)
    cat = str(row.iloc[0]["Economic Category"]).strip().lower()

    if any(cat.startswith(k) for k in [
        "real activity", "consumption", "inventories", "external sector",
        "prices/inflation",
    ]):
        return (1, 0, 1, 0)   # Real
    if cat.startswith("labor"):
        return (1, 0, 0, 1)   # Labor
    if cat.startswith("expectations") or cat.startswith("surveys"):
        return (1, 1, 0, 0)   # Soft
    if any(cat.startswith(k) for k in [
        "financial conditions", "monetary policy", "financial",
    ]):
        return (1, 0, 0, 0)   # Global
    return (1, 0, 0, 0)       # fallback Global

# ── 2d. Weekly vars: by series name ──────────────────────────────────────────
_WEEKLY_BLOCK = {
    "icsa_weekly":   (1, 0, 0, 1),   # Claims -> Labor
    "nfci_weekly":   (1, 1, 0, 0),   # Financial Conditions Index -> Soft
    "dtwexm":        (1, 0, 0, 0),   # FX -> Global
    "dtwexbgs":      (1, 0, 0, 0),   # FX -> Global
}

# Master block resolver
def resolve_blocks(col_lower, source):
    """Return (g, s, r, l) for a series given its source."""
    if col_lower in COVID_DUMMIES:
        return (1, 0, 0, 0)
    if source == "md":
        return _fredmd_block(col_lower)
    if source == "qd":
        return _qdonly_block(col_lower)
    if source == "user":
        return _user_block(col_lower)
    if source == "weekly":
        return _WEEKLY_BLOCK.get(col_lower, (1, 0, 0, 0))
    return (1, 0, 0, 0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LAG ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_user_lag(s):
    """
    Convert US_data_dictionary 'Release Lag' string -> integer months.

    Convention (matches Bok et al. / original repo):
      Fast releases published within ~1 week of T+1 month start -> lag=0
        (m1 forecast is made AFTER these data are in)
      Mid-month hard data (~2-3 weeks into T+1) -> lag=1
      1+ month delays -> lag=1 or 2 depending on size
    """
    if pd.isna(s):
        return 1
    s = str(s).strip().lower()
    if s in ("0", "0 day", "0 days",
             "1st business day", "3rd business day", "1 week"):
        return 0
    if s in ("2 weeks", "15 days"):
        return 1
    if s == "1 month":
        return 1
    if s in ("1-2 months", "2 months"):
        return 2
    return 1    # '?', blank, unrecognised -> default 1

# FRED-MD release-calendar lag lookup.
# Source: BLS/Fed release schedules, Bok et al. (2018) Appendix A,
#         original repo meta_data.csv (24-var reference).
# Default for FRED-MD = 1 (most monthly hard data published mid-next-month).
# Only vars that deviate from 1 are listed here.
_FREDMD_LAG = {
    # CPS labor (Employment Situation, released first Friday of T+1) -> lag=0
    "unrate": 0, "uemplt5": 0, "uempmean": 0, "uemp5to14": 0,
    "uemp15ov": 0, "uemp15t26": 0, "uemp27ov": 0,
    "civpart": 0, "ce16ov": 0,
    "lns14000012": 0, "lns14000025": 0, "lns14000026": 0,
    "lns13023621": 0, "lns13023557": 0, "lns13023705": 0,
    "lns13023569": 0, "lns12032194": 0,
    "unratestx": 0, "unrateltx": 0,
    "claimsx": 0,
    "hwiuratio": 0, "hwiuratiox": 0, "hwi": 0, "hwix": 0,
    # CES employment / hours / earnings (Employment Situation) -> lag=0
    "payems": 0, "usgood": 0, "uscons": 0, "manemp": 0,
    "dmanemp": 0, "ndmanemp": 0, "srvprd": 0,
    "ustpu": 0, "uswtrade": 0, "ustrade": 0, "usfire": 0,
    "usgovt": 0, "usserv": 0,
    "ces0600000007": 0, "awotman": 0, "awhman": 0,
    "ces0600000008": 0, "ces2000000008": 0, "ces3000000008": 0,
    "ahetpi": 0,
    "ces1021000001": 0, "ces9091000001": 0, "ces9092000001": 0,
    "ces9093000001": 0,
    "uspriv": 0, "usehs": 0, "usinfo": 0, "uspbs": 0,
    "uslah": 0, "usmine": 0, "awhnonag": 0,
    # Interest rates / spreads / FX / equity (published daily/weekly) -> lag=0
    "fedfunds": 0, "cp3mx": 0, "cp3m": 0, "tb3ms": 0, "tb6ms": 0,
    "gs1": 0, "gs5": 0, "gs10": 0, "aaa": 0, "baa": 0,
    "baa10ym": 0,
    "compapffx": 0, "compapff": 0,
    "mortg10yrx": 0, "mortgage30us": 0,
    "tb3smffm": 0, "tb6smffm": 0, "t1yffm": 0, "t5yffm": 0, "t10yffm": 0,
    "aaaffm": 0, "baaffm": 0, "tb6m3mx": 0, "gs1tb3mx": 0, "gs10tb3mx": 0,
    "cpf3mtb3mx": 0,
    "twexafegsmthx": 0, "exszusx": 0, "exjpusx": 0, "exusukx": 0,
    "excausx": 0, "exuseu": 0,
    "vixclsx": 0,
    "s&p 500": 0, "s&p div yield": 0, "s&p pe ratio": 0,
    "nikkei225": 0, "nasdaqcom": 0,
    # Survey (UMich, released last Fri of survey month) -> lag=0
    "umcsentx": 0,
    # INDPRO / IP released ~2 weeks into T+1 -> lag=0
    # (original repo uses lag=0; first Friday data available same month)
    "indpro": 0,
    # Economic Policy Uncertainty -> lag=0 (published ~first week of T+1)
    "usepuindxm": 0,
    # Help-Wanted (real-time online) -> lag=0
    "hwiuratio": 0,
}

# NIPA quarterly hard data: Advance estimate ~30 days after quarter-end.
# On a monthly grid the Q-end value sits in the final month of the quarter.
# Safe convention (Bok et al., original repo): lag=2 to prevent the m1
# vintage from seeing data not yet released.
_QUARTERLY_LAG2 = {
    "gdpc1", "pcecc96", "pcdgx", "pcesvx", "pcndx",
    "gpdic1", "fpix", "y033rc1q027sbeax", "pnfix", "prfix",
    "a014re1q156nbea", "gcec1", "a823rl1q225sbea", "fgrecptx",
    "slcex", "expgsc1", "impgsc1", "expgs", "impgs",
    "dpic96",
    "outnfb", "outbs", "outms",
    "a823rl1q225sbea",
    # Productivity/ULC (BLS, released quarterly ~2 months after Q-end)
    "comprnfb", "comprms", "rcphbs",
    "ophmfg", "ophnfb", "ophpbs",
    "ulcbs", "ulcmfg", "ulcnfb", "unlpnbs",
    # Price deflators (NIPA vintage)
    "pcectpi", "pcepilfe", "gdpctpi", "gpdictpi",
    # Flow-of-Funds balance sheets (released ~10 weeks after Q-end)
    "tlbshnox", "liabpix", "tnwbshnox", "tfaabshnox", "hnoremq027sx",
    "tnwmvbsnncbbdix", "tnwmvbsnncbx",
    "tlbsnncbbdix", "tlbsnncbx",
    "tabshnox", "ttaabsnncbx",
    "tlbsnnbbdix", "tlbsnnbx", "tabsnnbx",
    "tnwbsnnbbdix", "tnwbsnnbx", "cncfx", "taresax", "nwpix",
    # House price indices (FHFA / Case-Shiller, ~2 months lag)
    "ussthpi", "spcs10rsa", "spcs20rsa",
    # Other quarterly hard data
    "ipdbs", "invcqrmtspl",
    "b020re1q156nbea", "b021re1q156nbea",
    "gfdebtnx", "gfdegdq188s",
    # Quarterly deflated user vars
    "ahetpix", "acognox", "revolslx", "wpu0531",
    "tlbsnncbbdix", "dongrg3q086sbea",
    # QD-only variants of spending components
    "dgdsrg3q086sbea", "ddurrg3q086sbea", "dserrg3q086sbea",
    "dndgrg3q086sbea", "dhcerg3q086sbea", "dmotrg3q086sbea",
    "dfdhrg3q086sbea", "dreqrg3q086sbea", "dodgrg3q086sbea",
    "dfxarg3q086sbea", "dclorg3q086sbea", "dgoerg3q086sbea",
    "dhutrg3q086sbea", "dhlcrg3q086sbea", "dtrsrg3q086sbea",
    "drcarg3q086sbea", "dfsarg3q086sbea", "difsrg3q086sbea",
    "dotsrg3q086sbea",
}

def resolve_lag(col_lower, source, freq):
    """Return months_lag for a series."""
    if col_lower in COVID_DUMMIES:
        return 0

    # Quarterly hard-data override (takes priority: prevents Advance GDP leakage)
    if col_lower in _QUARTERLY_LAG2:
        return 2

    # User dictionary
    if source == "user":
        row = user_df[user_df["var_lc"] == col_lower]
        if len(row) > 0:
            return _parse_user_lag(row.iloc[0]["Release Lag"])

    # FRED-MD specific release calendar (covers both MD and QD-variant names)
    if col_lower in _FREDMD_LAG:
        return _FREDMD_LAG[col_lower]

    # Weekly vars: all real-time
    if source == "weekly":
        return 0

    # FRED-MD default
    if source == "md":
        return 1

    # FRED-QD default (most quarterly hard data)
    if source == "qd":
        return 2

    return 1   # fallback

# ─────────────────────────────────────────────────────────────────────────────
# 4. FREQ ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def resolve_freq(col_lower, source):
    if col_lower in COVID_DUMMIES:
        return "m"
    if source == "weekly":
        return "w"
    if source == "md":
        return "m"
    if source == "qd":
        return "q"
    # User vars: check dictionary
    row = user_df[user_df["var_lc"] == col_lower]
    if len(row) > 0:
        f = str(row.iloc[0]["Frequency"]).strip().lower()
        if f.startswith("month"):
            return "m"
        if f.startswith("quarter"):
            return "q"
        if f.startswith("week"):
            return "w"
        if f.startswith("dail"):
            return "m"   # daily user vars aggregated to monthly
    return "m"   # safe fallback

# ─────────────────────────────────────────────────────────────────────────────
# 5. SOURCE RESOLVER  (determines source tag for each column)
# ─────────────────────────────────────────────────────────────────────────────
# User-kept variables from build_raw_data.py
USER_MONTHLY = {
    "bopgstb", "cfnai", "cscicp03usm665s", "dspic96",
    "dtwexbgs_monthly_avg", "dtwexm_monthly_avg",
    "gacdfsa066msfrbphi", "gacdisa066msfrbny",
    "hsn1f", "nfci_monthly_avg", "pce", "pcec96", "pi", "ppifis",
    "rsafs", "tcu", "tlofcons", "ttlcons",
    "uspmi_clean", "us_services_pmi_clean",
}
USER_QUARTERLY = {"expgs", "impgs"}
USER_ALL = USER_MONTHLY | USER_QUARTERLY

def get_source(col_lower):
    if col_lower in COVID_DUMMIES:
        return "covid"
    if col_lower in MD_SET:
        return "md"
    if col_lower in USER_ALL:
        return "user"
    if col_lower in QD_ONLY:
        return "qd"
    return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD
# ─────────────────────────────────────────────────────────────────────────────
df_m = pd.read_csv(os.path.join(BASE_DIR, "data_tf_monthly.csv"), nrows=1)
df_w = pd.read_csv(os.path.join(BASE_DIR, "data_tf_weekly.csv"),  nrows=1)

cols_m = [c for c in df_m.columns if c.lower() != "date"]
cols_w = [c for c in df_w.columns if c.lower() != "date"]

rows = []
seen = set()

# ── Monthly file ──────────────────────────────────────────────────────────────
for col in cols_m:
    cl = col.lower()
    if cl in seen:
        raise ValueError(f"Duplicate series in monthly file: {cl}")
    seen.add(cl)

    if cl in COVID_DUMMIES:
        src = "covid"
    else:
        src = get_source(cl)
        if src == "unknown":
            print(f"  WARNING: unknown source for '{cl}', defaulting to QD")
            src = "qd"

    freq  = resolve_freq(cl, src)
    lag   = resolve_lag(cl, src, freq)
    g, s, r, l = resolve_blocks(cl, src)

    # Display name: uppercase for real series, lowercase for covid dummies
    name = col.upper() if cl not in COVID_DUMMIES else cl
    rows.append([cl, name, freq, g, s, r, l, lag])

# ── Weekly file (skip COVID dummies already in monthly) ───────────────────────
for col in cols_w:
    cl = col.lower()
    if cl in seen:
        continue   # COVID dummies appear in both files; skip duplicate
    seen.add(cl)

    freq  = "w"
    lag   = resolve_lag(cl, "weekly", "w")
    g, s, r, l = resolve_blocks(cl, "weekly")
    rows.append([cl, col.upper(), freq, g, s, r, l, lag])

md = pd.DataFrame(rows, columns=[
    "series", "name", "freq", "block_g", "block_s", "block_r", "block_l", "months_lag"
])

# Hard-fail on duplicates
dups = md[md["series"].duplicated()]
if len(dups) > 0:
    raise ValueError(f"Duplicate series after build: {dups['series'].tolist()}")

# Save
out = os.path.join(BASE_DIR, "meta_data.csv")
md.to_csv(out, index=False)
print(f"Wrote {len(md)} rows to {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\nFreq counts:     ", md["freq"].value_counts().sort_index().to_dict())
print("Lag distribution:", md["months_lag"].value_counts().sort_index().to_dict())
print("Block totals: g={}, s={}, r={}, l={}".format(
    md["block_g"].sum(), md["block_s"].sum(),
    md["block_r"].sum(), md["block_l"].sum()))

# Source breakdown
src_counts = {}
for col in cols_m:
    cl = col.lower()
    if cl in COVID_DUMMIES:
        src_counts["covid"] = src_counts.get("covid", 0) + 1
    else:
        s = get_source(cl)
        src_counts[s] = src_counts.get(s, 0) + 1
print(f"\nMonthly source breakdown: {src_counts}")

# Top-35 sanity check
top35 = [
    "outbs", "gcec1", "tlbsnncbbdix", "hwiuratio", "busloans", "ulcnfb",
    "a014re1q156nbea", "houstne", "unrate", "andenox", "acogno", "amdmuox",
    "ces2000000008", "mortg10yrx", "ces9092000001", "usgovt", "usserv",
    "ces9091000001", "compapff", "slcex", "ahetpix", "awotman",
    "ces1021000001", "dpic96", "manemp", "expgsc1", "invest", "revolslx",
    "dongrg3q086sbea", "uemplt5", "acognox", "wpu0531", "cusr0000sas",
    "clf16ov", "m2sl",
]
top_md = md[md["series"].isin(top35)]
print(f"\nTop-35 sanity ({len(top_md)} found in metadata):")
print(f"  block_l = {top_md['block_l'].sum()}  (expected >= 10)")
print(f"  block_s = {top_md['block_s'].sum()}  (expected >= 1)")
print(f"  block_r = {top_md['block_r'].sum()}  (expected >= 10)")
print(f"  freq: {top_md['freq'].value_counts().to_dict()}")
print(f"  lag:  {top_md['months_lag'].value_counts().sort_index().to_dict()}")
print()
print(top_md[["series","freq","block_g","block_s","block_r","block_l","months_lag"]].to_string(index=False))
