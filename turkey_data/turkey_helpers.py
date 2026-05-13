"""
turkey_helpers.py
==================
Turkey-specific shared module for model notebooks.
Imports core functions from US helpers.py and overrides paths/configs.

Core functions (imported, never redefined):
    gen_lagged_data   — ragged-edge publication-lag mask
    flatten_data      — create lagged monthly columns (UMIDAS)
    mean_fill_dataset — fill NaN with training-fold means
    split_for_scaler  — separate COVID dummies before scaling
"""
# Category guide:
#   cat1 = ARMA (target only: ngdprsaxdctrq)
#   cat2 = VAR/OLS (4 vars: ipi_sa, usd_try_avg, cpi_sa, fin_acc + COVID = 7)
#   cat3 = most models (22 vars at >=34% training coverage + COVID = 25)
#   "dfm" = DFM only (cat3 22 vars + tier_c 32 vars = 54, no COVID in list —
#            DFM R notebook appends COVID separately)

import sys, os, json
import numpy as np
import pandas as pd

# Import core functions from US helpers.py — NEVER redefine these
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from helpers import (
    gen_lagged_data,
    flatten_data,
    mean_fill_dataset,
    split_for_scaler,
)

# Turkey-specific paths
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
VAR_LISTS = os.path.join(BASE, "turkey_variable_lists.json")
MONTHLY_CSV = os.path.join(BASE, "data_tf_monthly_tr.csv")
WEEKLY_CSV = os.path.join(BASE, "data_tf_weekly_tr.csv")
METADATA_CSV = os.path.join(BASE, "meta_data_tr.csv")
PREDICTIONS_DIR = os.path.join(ROOT, "turkey_predictions")

TARGET = "ngdprsaxdctrq"
COVID = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]

with open(VAR_LISTS, "r", encoding="utf-8") as f:
    _var_cfg = json.load(f)


def get_features(category="cat3", with_covid=True):
    """
    Return feature column names for a given category.

    category : "cat1", "cat2", "cat3", or "dfm"
        cat1 = ARMA (target only: ngdprsaxdctrq)
        cat2 = VAR/OLS (4 vars from RF+Stability ensemble)
        cat3 = most models (22 vars at >=34% training coverage)
        dfm  = DFM (cat3 22 vars + tier_c 32 vars = 54 vars total)

    with_covid : bool
        If True, append covid_2020q2/q3/q4 to the returned list.
        Has no effect when category="cat1".
    """
    if category == "dfm":
        return get_dfm_features(with_covid=with_covid)
    cfg = _var_cfg.get(category)
    if cfg is None:
        raise ValueError(
            "Unknown category '{}'. Use 'cat1', 'cat2', 'cat3', or 'dfm'.".format(category)
        )
    feats = list(cfg["features"])
    if with_covid and category != "cat1":
        feats = feats + COVID
    return feats


def get_dfm_features(with_covid=True):
    """
    Return the full 54-variable list for the Turkey DFM notebook.

    DFM uses ALL variables: cat3 (22) + tier_c_dfm_only (32) = 54.
    The Kalman filter handles ragged edges / missing data internally,
    so Tier C variables (short history, <34% training coverage) are safe to include.

    with_covid : bool
        If True, append the 3 COVID dummies (total = 57 columns passed to DFM).
        The R DFM notebook passes these as exogenous regressors, not as factors.
    """
    feats = list(_var_cfg["cat3"]["features"]) + list(_var_cfg["tier_c_dfm_only"])
    if with_covid:
        feats = feats + COVID
    return feats


def load_data():
    """Load monthly transformed data, weekly transformed data, and metadata."""
    monthly = pd.read_csv(MONTHLY_CSV, parse_dates=["date"])
    weekly = pd.read_csv(WEEKLY_CSV, parse_dates=["Date"])
    metadata = pd.read_csv(METADATA_CSV)
    return monthly, weekly, metadata
