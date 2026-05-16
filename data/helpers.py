"""
helpers.py — single source of truth for all Python model notebooks.

Import this in every notebook. Never redefine these functions locally.
If a function needs modifying, modify it here and all notebooks benefit.

Functions that are IDENTICAL to the original repo:
    gen_lagged_data, mean_fill_dataset, flatten_data

Functions added for Pipeline B:
    get_features, split_for_scaler, load_data, gen_vintage_data
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
# Below: shared ragged-edge and supervised-learning helpers used across models.
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


def gen_vintage_data(metadata, data, target_date, vintage_date):
    """
    Apply an explicit target-quarter / information-date ragged-edge mask.

    This is the Python analogue of data/helpers.R::gen_vintage_data. It keeps
    rows through the target quarter so current-quarter methods can form a
    nowcast, while masking each series according to what would have been
    available at the simulated vintage month.

    Parameters
    ----------
    metadata : pd.DataFrame  -- must have 'series' and 'months_lag'
    data : pd.DataFrame      -- monthly dataframe with 'date' column
    target_date : date-like  -- quarter-end target month retained in output
    vintage_date : date-like -- simulated information date

    Returns
    -------
    pd.DataFrame with rows through target_date and trailing unavailable values
    set to NaN. For months_lag=0, the current vintage month is still treated as
    not yet released, matching the R helper and original ragged-edge convention.
    """
    target_date = pd.Timestamp(target_date)
    vintage_date = pd.Timestamp(vintage_date)
    vintage_data = data.loc[data.date <= target_date, :].reset_index(drop=True).copy()

    lag_lookup = metadata.set_index("series")["months_lag"].to_dict()
    for col in vintage_data.columns[1:]:
        if col not in lag_lookup:
            continue
        available_through = vintage_date - pd.DateOffset(months=int(lag_lookup[col]) + 1)
        vintage_data.loc[vintage_data["date"] > available_through, col] = np.nan

    return vintage_data


def make_supervised_vintage_frame(
    metadata,
    data,
    target_variable,
    features,
    train_start_date,
    target_date,
    vintage_date,
    n_lags,
):
    """
    Build train/test matrices for flattened supervised models at a target quarter.

    The returned test row is always the target quarter row, while `vintage_date`
    controls which monthly indicators are visible. This matters for post-quarter
    horizons such as post1: the information set is later, but the forecast target
    remains the same GDP quarter.
    """
    target_date = pd.Timestamp(target_date)
    vintage_date = pd.Timestamp(vintage_date)
    cols = ["date", target_variable] + [
        col for col in features if col in data.columns and col != target_variable
    ]
    raw = data.loc[
        (data["date"] >= pd.Timestamp(train_start_date)) & (data["date"] <= target_date),
        cols,
    ].reset_index(drop=True)
    vintage = gen_vintage_data(metadata, raw, target_date, vintage_date)
    vintage.loc[vintage["date"] == target_date, target_variable] = np.nan

    train_context = vintage.loc[vintage["date"] < target_date].copy()
    filled = mean_fill_dataset(train_context, vintage)
    flat = flatten_data(filled, target_variable, n_lags)
    flat.loc[flat["date"] == target_date, target_variable] = np.nan

    quarter_rows = flat["date"].dt.month.isin([3, 6, 9, 12])
    train_flat = flat.loc[quarter_rows & (flat["date"] < target_date), :].copy()
    train_flat = train_flat.dropna(axis=0, how="any").reset_index(drop=True)
    test_flat = flat.loc[flat["date"] == target_date, :].tail(1).copy()
    return train_flat, test_flat, filled


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
