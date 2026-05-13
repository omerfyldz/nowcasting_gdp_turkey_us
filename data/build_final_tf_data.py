import pandas as pd
import numpy as np
import os

TCODE_OVERRIDES = {
    'NWPIx': 5,   # Net Worth (Pension), trending stock variable
    'HWIx':  5,   # Help-Wanted Index, structural redefinition
}
def apply_tcode(series, tcode, warnings_list=None):
    """
    Applies the McCracken-Ng transformation codes to a pandas Series.
    1 = no transformation
    2 = first difference
    3 = second difference
    4 = log
    5 = first difference of log
    6 = second difference of log
    7 = Delta (x_t/x_{t-1} - 1)

    Approach: drop NaNs and non-positives (for log tcodes), compute the
    transformation on the dense valid sequence, then reindex to the
    original time grid. This treats `.diff()` as "change since last
    observation", which is the correct quantity for quarterly series
    placed on a monthly grid (we want Q-to-Q diffs, not month-to-month
    NaN-spanning diffs that would be NaN everywhere). Inf from
    `pct_change` with a zero denominator is sanitised to NaN.
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

def add_covid_dummies_monthly(df):
    """
    Adds COVID_2020Q2, COVID_2020Q3, COVID_2020Q4 dummy variables to a monthly dataframe.
    The dummy is 1 for the months within that quarter, 0 otherwise.
    Q2: April, May, June
    Q3: July, August, September
    Q4: October, November, December
    """
    dates = pd.to_datetime(df.index)
    
    q2_mask = (dates.year == 2020) & (dates.month.isin([4, 5, 6]))
    q3_mask = (dates.year == 2020) & (dates.month.isin([7, 8, 9]))
    q4_mask = (dates.year == 2020) & (dates.month.isin([10, 11, 12]))
    
    df['COVID_2020Q2'] = q2_mask.astype(int)
    df['COVID_2020Q3'] = q3_mask.astype(int)
    df['COVID_2020Q4'] = q4_mask.astype(int)
    return df

def add_covid_dummies_weekly(df):
    """
    Adds COVID_2020Q2, COVID_2020Q3, COVID_2020Q4 dummy variables to a weekly dataframe.
    Q2: April 1, 2020 - June 30, 2020
    Q3: July 1, 2020 - Sept 30, 2020
    Q4: Oct 1, 2020 - Dec 31, 2020
    """
    dates = pd.to_datetime(df.index)
    
    q2_mask = (dates >= '2020-04-01') & (dates <= '2020-06-30')
    q3_mask = (dates >= '2020-07-01') & (dates <= '2020-09-30')
    q4_mask = (dates >= '2020-10-01') & (dates <= '2020-12-31')
    
    df['COVID_2020Q2'] = q2_mask.astype(int)
    df['COVID_2020Q3'] = q3_mask.astype(int)
    df['COVID_2020Q4'] = q4_mask.astype(int)
    return df

def process_file(input_path, output_path, freq):
    print(f"Processing {input_path}...")
    
    # Read raw data. Assume row 0 is the tcode row, row 1 is the header if standard.
    # Actually, in build_weekly_data.py we injected tcodes as the very first row (index 0).
    df_raw = pd.read_excel(input_path)
    
    # The first row contains the tcodes.
    # The first column is 'Date' or 'date'
    date_col = df_raw.columns[0]
    
    # Extract t-codes and apply overrides
    tcode_row = df_raw.iloc[0]
    tcodes = {}
    overrides_applied = []
    for col in df_raw.columns[1:]:
        tc = tcode_row[col]
        if col in TCODE_OVERRIDES:
            overrides_applied.append((col, tc, TCODE_OVERRIDES[col]))
            tc = TCODE_OVERRIDES[col]
        tcodes[col] = tc
        
    if overrides_applied:
        print(f"  TCODE OVERRIDES applied: {len(overrides_applied)}")
        for var, orig, new in overrides_applied:
            print(f"    - {var}: {orig} -> {new}")
            
    # Drop the tcode row
    df_data = df_raw.drop(0).copy()
    
    # Convert date and set index
    df_data[date_col] = pd.to_datetime(df_data[date_col])
    df_data = df_data.set_index(date_col)
    
    # Convert all columns to numeric
    for col in df_data.columns:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        
    # Apply transformations
    df_tf = pd.DataFrame(index=df_data.index)
    warnings_list = []
    errors = []
    for col in df_data.columns:
        try:
            df_tf[col] = apply_tcode(df_data[col], tcodes[col], warnings_list)
        except Exception as e:
            errors.append((col, tcodes[col], str(e)))
            df_tf[col] = np.nan
            
    if errors:
        print(f"  TRANSFORMATION ERRORS ({len(errors)}):")
        for c, tc, msg in errors:
            print(f"    - {c} (tcode={tc}): {msg}")
    if warnings_list:
        print(f"  DATA HYGIENE WARNINGS ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"    - {w}")
        
    # Add COVID dummies based on frequency
    if freq == 'monthly':
        df_tf = add_covid_dummies_monthly(df_tf)
    elif freq == 'weekly':
        df_tf = add_covid_dummies_weekly(df_tf)
        
    # Lowercase all columns
    df_tf.columns = [str(c).lower() for c in df_tf.columns]
        
    # Save to CSV
    df_tf.reset_index().to_csv(output_path, index=False)
    print(f"Saved transformed data to {output_path}")

if __name__ == "__main__":
    base_dir = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
    
    monthly_input = os.path.join(base_dir, "data_raw_monthl.xlsx")
    monthly_output = os.path.join(base_dir, "data_tf_monthly.csv")
    
    weekly_input = os.path.join(base_dir, "data_weekly_aligned.xlsx")
    weekly_output = os.path.join(base_dir, "data_tf_weekly.csv")
    
    if os.path.exists(monthly_input):
        process_file(monthly_input, monthly_output, freq='monthly')
    else:
        print(f"Error: {monthly_input} not found!")
        
    if os.path.exists(weekly_input):
        process_file(weekly_input, weekly_output, freq='weekly')
    else:
        print(f"Error: {weekly_input} not found!")
