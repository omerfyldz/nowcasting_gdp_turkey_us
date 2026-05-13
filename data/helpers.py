"""
helpers.py — single source of truth for all Python model notebooks.

Import this in every notebook. Never redefine these functions locally.
If a function needs modifying, modify it here and all notebooks benefit.

Functions that are IDENTICAL to the original repo:
    gen_lagged_data, mean_fill_dataset, flatten_data

Functions added for Pipeline B:
    get_features, split_for_scaler, load_data
"""
import json
import os
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # nowcasting_benchmark-main/
VAR_LISTS = os.path.join(BASE, "variable_lists.json")
MONTHLY_CSV = os.path.join(BASE, "data_tf_monthly.csv")
WEEKLY_CSV = os.path.join(BASE, "data_tf_weekly.csv")
METADATA_CSV = os.path.join(BASE, "meta_data.csv")
PREDICTIONS_DIR = os.path.join(ROOT, "predictions")

# ── COVID dummy column names (as they appear in data_tf_monthly.csv) ─────────
COVID = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]

# ── Load variable lists once at import time ──────────────────────────────────
with open(VAR_LISTS, "r") as f:
    _var_cfg = json.load(f)

TARGET = _var_cfg["target"]  # "gdpc1"


def get_features(category="cat3", with_covid=True):
    """
    Return list of feature column names for a given category.

    category : "cat2" or "cat3"
        cat2 = VAR/OLS (4 vars: outbs, outnfb, gcec1, houstne)
        cat3 = all other models (53 vars from L95 U E95 U R95 U S100)

    with_covid : bool
        If True, append covid_2020q2/q3/q4 to the returned list.
        COVID dummies are zero-variance in 1959-2007 training window.
        They must NOT be standardized. Use split_for_scaler() to
        separate them before any StandardScaler call.

    Returns
    -------
    list of str : feature column names (lowercase, as they appear in the CSV)
    """
    cfg = _var_cfg.get(category)
    if cfg is None:
        raise ValueError("Unknown category '{}'. Use 'cat2' or 'cat3'.".format(category))
    feats = list(cfg["features"])
    if with_covid:
        feats = feats + COVID
    return feats


def split_for_scaler(features):
    """
    Split a feature list into (cols_to_scale, cols_to_pass_through).

    COVID dummies must NEVER be standardized — they are zero/one by
    construction and have zero variance in pre-2020 training folds.
    Standardizing them produces NaN and silently kills the scaler.

    Returns
    -------
    scaled : list of str   — columns safe for StandardScaler
    unscaled : list of str — columns to pass through unchanged (COVID dummies)
    """
    scaled = [c for c in features if c not in COVID]
    unscaled = [c for c in features if c in COVID]
    return scaled, unscaled


def load_data():
    """
    Load monthly data, weekly data, and metadata.

    Returns
    -------
    monthly : pd.DataFrame  — data_tf_monthly.csv with parsed dates
    weekly  : pd.DataFrame  — data_tf_weekly.csv with parsed dates
    metadata : pd.DataFrame — meta_data.csv
    """
    monthly = pd.read_csv(MONTHLY_CSV, parse_dates=["date"])
    weekly = pd.read_csv(WEEKLY_CSV, parse_dates=["Date"])
    metadata = pd.read_csv(METADATA_CSV)
    return monthly, weekly, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Below: functions byte-for-byte identical to the original Hopp (2023) repo.
# DO NOT MODIFY without cross-checking against the original model_ols.ipynb.
# ──────────────────────────────────────────────────────────────────────────────


def gen_lagged_data(metadata, data, last_date, lag):
    """
    Apply ragged-edge publication-lag mask to simulate data availability
    at a specific forecast vintage.

    For each column, masks the most recent `months_lag` observations
    relative to `last_date`. This simulates which data had actually been
    released by that point in time.

    MUST be called BEFORE mean_fill_dataset. Mean-fill replaces NaNs
    with training-fold means; if the mask is applied after, the ragged-
    edge information is destroyed.

    Parameters
    ----------
    metadata : pd.DataFrame  — meta_data.csv (must have 'series' and 'months_lag')
    data     : pd.DataFrame  — monthly dataframe with 'date' column
    last_date : str/timestamp — forecast vintage date
    lag      : int           — months ahead (negative) or behind (positive)

    Returns
    -------
    pd.DataFrame with trailing NaNs where data was not yet released
    """
    lagged_data = data.loc[data.date <= last_date, :].reset_index(drop=True)
    for col in lagged_data.columns[1:]:
        pub_lag = metadata.loc[metadata.series == col, "months_lag"].values[0]
        lagged_data.loc[(len(lagged_data) - pub_lag + lag - 1):, col] = np.nan
    return lagged_data


def flatten_data(data, target_variable, n_lags):
    """
    Create lagged copies of each variable for models that don't handle
    time series natively (OLS, Lasso, trees, MLP).

    Each variable V gets columns: V, V_1, V_2, ..., V_{n_lags}
    representing current value plus n_lags monthly lags.

    Parameters
    ----------
    data            : pd.DataFrame  — must have 'date' column
    target_variable : str           — column to exclude from lagging (typically 'gdpc1')
    n_lags          : int           — number of lagged copies per variable

    Returns
    -------
    pd.DataFrame with (1 + n_lags) * N_vars + date + target columns
    """
    flattened_data = data.copy()
    orig_index = flattened_data.index
    for i in range(1, n_lags + 1):
        lagged_indices = orig_index - i
        lagged_indices = lagged_indices[lagged_indices >= 0]
        tmp = data.loc[lagged_indices, :].copy()
        tmp.date = tmp.date + pd.DateOffset(months=i)
        tmp = tmp.drop([target_variable], axis=1)
        tmp.columns = [j + "_" + str(i) if j != "date" else j for j in tmp.columns]
        flattened_data = flattened_data.merge(tmp, how="left", on="date")
    return flattened_data


def mean_fill_dataset(training, test):
    """
    Fill missing values in `test` with column means computed from `training`.

    The training DataFrame provides the means — this prevents information
    leakage from test into training. Call this inside the rolling loop
    with the current training fold as the first argument.

    Parameters
    ----------
    training : pd.DataFrame  — training fold (means computed from this)
    test     : pd.DataFrame  — data to fill (can be same as training for
                               initial fill, or a single test row)

    Returns
    -------
    pd.DataFrame with NaN values replaced by training-fold column means
    """
    mean_dict = {}
    for col in training.columns[1:]:
        mean_dict[col] = np.nanmean(training[col])
    filled = test.copy()
    for col in training.columns[1:]:
        filled.loc[pd.isna(filled[col]), col] = mean_dict[col]
    return filled
