import warnings
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, DFGLS

warnings.filterwarnings("ignore")

base_dir = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
MONTHLY_TF = os.path.join(base_dir, "data_tf_monthly.csv")
WEEKLY_TF  = os.path.join(base_dir, "data_tf_weekly.csv")
REPORT     = os.path.join(base_dir, "stationarity_report.xlsx")

ALPHA  = 0.05
MIN_N  = 30

def load_tf_csv(path):
    data = pd.read_csv(path)
    date_col = data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    
    # We drop COVID dummies for stationarity testing because dummies are inherently structural breaks
    # and will fail stationarity tests.
    covid_cols = [c for c in data.columns if 'covid' in c.lower()]
    data = data.drop(covid_cols, axis=1)
    
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    return data

def test_one(s: pd.Series):
    s = s.dropna()
    n = len(s)
    out = {'n_obs': n}
    if n < MIN_N or s.std() == 0:
        out.update({'adf_p': np.nan, 'kpss_p': np.nan,
                    'pp_p': np.nan, 'dfgls_stat': np.nan, 'decision': 'TOO_SHORT_OR_CONSTANT'})
        return out

    try:
        adf_p = adfuller(s, autolag='AIC', regression='c')[1]
    except Exception:
        adf_p = np.nan

    try:
        kpss_p = kpss(s, regression='c', nlags='auto')[1]
    except Exception:
        kpss_p = np.nan

    try:
        pp_p = PhillipsPerron(s, trend='c').pvalue
    except Exception:
        pp_p = np.nan

    try:
        dfgls_stat = DFGLS(s, trend='c').stat
    except Exception:
        dfgls_stat = np.nan

    out['adf_p']     = adf_p
    out['kpss_p']    = kpss_p
    out['pp_p']      = pp_p
    out['dfgls_stat'] = dfgls_stat

    adf_rej   = (not np.isnan(adf_p))  and adf_p  < ALPHA
    kpss_rej  = (not np.isnan(kpss_p)) and kpss_p < ALPHA
    pp_rej    = (not np.isnan(pp_p))   and pp_p   < ALPHA

    if adf_rej and (not kpss_rej) and pp_rej:
        decision = 'STATIONARY'
    elif (not adf_rej) and kpss_rej and (not pp_rej):
        decision = 'NON_STATIONARY'
    else:
        decision = 'INCONCLUSIVE'

    out['decision'] = decision
    return out

def run_battery(data, label):
    print(f"\n{'='*64}\n{label}: running tests on {len(data.columns)} variables\n{'='*64}")
    rows = []
    columns = data.columns
    for i, col in enumerate(columns, 1):
        if i % 30 == 0:
            print(f"  ... {i}/{len(columns)}")
        r = test_one(data[col])
        r['variable'] = col
        rows.append(r)
    df = pd.DataFrame(rows)[['variable','n_obs','adf_p','kpss_p','pp_p','dfgls_stat','decision']]
    return df

md_data = load_tf_csv(MONTHLY_TF)
wk_data = load_tf_csv(WEEKLY_TF)

md_report = run_battery(md_data, "MONTHLY")
wk_report = run_battery(wk_data, "WEEKLY")

def summarize(df, label):
    print(f"\n--- {label} SUMMARY ---")
    counts = df['decision'].value_counts()
    print(counts.to_string())
    print(f"  Total: {len(df)}")

    bad = df[df['decision'] != 'STATIONARY'].sort_values('decision')
    if len(bad):
        print(f"\n  Variables NOT confidently stationary ({len(bad)}):")
        print(bad[['variable','n_obs','adf_p','kpss_p','pp_p','decision']]
              .to_string(index=False, max_rows=200))

summarize(md_report, "MONTHLY")
summarize(wk_report, "WEEKLY")

with pd.ExcelWriter(REPORT, engine='openpyxl') as w:
    md_report.to_excel(w, sheet_name='monthly', index=False)
    wk_report.to_excel(w, sheet_name='weekly',  index=False)

print(f"\nReport saved: {REPORT}")
