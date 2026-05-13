import pandas as pd
import numpy as np

INPUT_PATH = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data/daily_weekly_series.xlsx"
OUTPUT_PATH = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data/data_weekly_aligned.xlsx"

print(f"Loading data from {INPUT_PATH}...")
df_raw = pd.read_excel(INPUT_PATH)
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.set_index('Date')

# Identify variable types based on analysis.
# NOTE: DTWEXM_weekly and DTWEXBGS_weekly (Fed 5-day averages) are DROPPED.
# We keep only the End-of-Period (EoP) aggregations DTWEXM and DTWEXBGS
# produced below via daily->weekly resampling (last valid obs per week).
# Verification: EoP series are built from 'DTWEXBGS' and 'DTWEXM' daily
# columns in the source file, which exist and are confirmed non-empty.
weekly_native_cols = ['ICSA_weekly', 'NFCI_weekly']   # Fed-avg FX dropped
daily_cols = ['DTWEXBGS', 'DTWEXM']

# Step 1: Process native weekly variables
# Extract rows where these weekly variables are not null
print("Aligning native weekly variables...")
weekly_series = []
for col in weekly_native_cols:
    if col in df_raw.columns:
        s = df_raw[col].dropna()
        # Ensure they are aligned to the nearest Saturday of their respective week
        # ICSA is already Saturday. NFCI is Friday, so Friday + 1 = Saturday
        # `pd.offsets.Week(weekday=5)` forces the date to the nearest Saturday
        s.index = s.index + pd.offsets.Week(weekday=5) - pd.offsets.Week()
        
        # In case shifting causes duplicate dates (e.g. if a series had two points in the same week)
        s = s[~s.index.duplicated(keep='last')]
        weekly_series.append(s)

df_weekly_aligned = pd.concat(weekly_series, axis=1)

# Step 2: Process daily variables (End-of-Period aggregation)
# We resample the daily series into weeks ending on Saturday ('W-SAT')
# and take the last valid observation of that week (which handles missing weekends/holidays)
print("Aggregating daily variables to weekly using End-of-Period...")
daily_series = []
for col in daily_cols:
    if col in df_raw.columns:
        s = df_raw[col].dropna()
        # Resample to Week ending on Saturday, taking the last available observation
        s_weekly = s.resample('W-SAT').last()
        daily_series.append(s_weekly)

df_daily_agg = pd.concat(daily_series, axis=1)

# Step 3: Merge everything together
print("Merging datasets into unified weekly dataframe...")
final_df = df_weekly_aligned.join(df_daily_agg, how='outer').sort_index()

# Filter out rows that are entirely empty (just in case)
final_df = final_df.dropna(how='all')

print(f"Final dataset shape: {final_df.shape}")
print(f"Date range: {final_df.index.min().date()} to {final_df.index.max().date()}")

# Ensure all index dates are strictly Saturdays
saturday_check = (final_df.index.weekday == 5).all()
if saturday_check:
    print("SUCCESS: All dates in the final dataset are strictly Saturdays.")
else:
    print("WARNING: Some dates are not Saturdays.")

# Step 4: Inject t_code row at the top
print("Injecting t_code row at the top of the dataset...")
tcodes = {
    'ICSA_weekly': 5.0,
    'NFCI_weekly': 1.0,
    'DTWEXBGS': 5.0,
    'DTWEXM': 5.0,
}

# Reset index to make 'Date' a column
final_df_reset = final_df.reset_index()
# Format the date column as string for consistency with the tcode row
final_df_reset['Date'] = final_df_reset['Date'].dt.strftime('%Y-%m-%d')

# Create the tcode row
tcode_vals = ['tcode'] + [tcodes.get(col, np.nan) for col in final_df.columns]
tcode_row = pd.DataFrame([tcode_vals], columns=['Date'] + list(final_df.columns))

# Concatenate tcode row with data
final_with_tcode = pd.concat([tcode_row, final_df_reset], ignore_index=True)

print(f"Writing output with tcodes to {OUTPUT_PATH}...")
final_with_tcode.to_excel(OUTPUT_PATH, sheet_name='raw_data', index=False)
print("Done.")
