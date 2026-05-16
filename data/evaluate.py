"""
Country-aware evaluation for Pipeline B nowcasting outputs.

This script treats the existing prediction CSVs as the source of truth. It:
1. Audits US and Turkey prediction files for expected models, vintages, dates,
   columns, row counts, and finite predictions.
2. Computes panel-level RMSFE, MAE, relative RMSFE vs ARMA, and
   Diebold-Mariano tests vs ARMA.
3. Writes country-specific CSV outputs and a concise root-level summary.

Usage:
    python data/evaluate.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

MODELS = [
    "arma",
    "ols",
    "var",
    "lasso",
    "ridge",
    "elasticnet",
    "rf",
    "xgboost",
    "gb",
    "dt",
    "mlp",
    "lstm",
    "deepvar",
    "bvar",
    "midas",
    "midasml",
    "dfm",
]
DEFAULT_VINTAGES = ["m1", "m2", "m3"]
REQUIRED_COLUMNS = ["date", "actual", "prediction"]


@dataclass(frozen=True)
class CountryConfig:
    code: str
    label: str
    prediction_dir: str
    output_path: str
    expected_start: str
    expected_end: str
    expected_rows: int
    panels: Dict[str, Tuple[str, str]]
    vintages: Tuple[str, ...] = tuple(DEFAULT_VINTAGES)
    turkey_filenames: bool = False


COUNTRIES = [
    CountryConfig(
        code="us",
        label="United States",
        prediction_dir=os.path.join(ROOT, "predictions"),
        output_path=os.path.join(BASE, "evaluation_results_us.csv"),
        expected_start="2017-03-01",
        expected_end="2025-12-01",
        expected_rows=36,
        panels={
            "pre_covid": ("2017-01-01", "2019-12-31"),
            "covid": ("2020-04-01", "2021-12-31"),
            "post_covid": ("2022-01-01", "2025-12-31"),
            "full": ("2017-01-01", "2025-12-31"),
        },
        vintages=("m1", "m2", "m3", "post1"),
    ),
    CountryConfig(
        code="tr",
        label="Turkey",
        prediction_dir=os.path.join(ROOT, "turkey_predictions"),
        output_path=os.path.join(ROOT, "turkey_data", "evaluation_results_tr.csv"),
        expected_start="2018-03-01",
        expected_end="2025-12-01",
        expected_rows=32,
        panels={
            "pre_crisis": ("2018-01-01", "2019-12-31"),
            "covid": ("2020-04-01", "2021-12-31"),
            "post_covid": ("2022-01-01", "2025-12-31"),
            "full": ("2018-01-01", "2025-12-31"),
        },
        vintages=("m1", "m2", "m3", "post1", "post2"),
        turkey_filenames=True,
    ),
]


def prediction_path(config: CountryConfig, model: str, vintage: str) -> str:
    """Return the expected prediction path, accepting Turkey legacy names."""
    if config.turkey_filenames:
        preferred = os.path.join(config.prediction_dir, f"{model}_tr_{vintage}.csv")
        legacy = os.path.join(config.prediction_dir, f"{model}_{vintage}.csv")
        if os.path.exists(preferred):
            return preferred
        return legacy
    return os.path.join(config.prediction_dir, f"{model}_{vintage}.csv")


def expected_dates(config: CountryConfig) -> pd.DatetimeIndex:
    return pd.date_range(config.expected_start, config.expected_end, freq="QS-DEC")


def normalize_quarter_dates(dates: pd.Series) -> pd.Series:
    """Normalize quarter-start/end labels to quarter-end-month first day.

    Some notebooks write the forecast quarter as Jan/Apr/Jul/Oct, while most
    write Mar/Jun/Sep/Dec. Both identify the same quarter. Evaluation uses the
    latter convention without mutating raw prediction files.
    """
    periods = pd.to_datetime(dates).dt.to_period("Q")
    return periods.dt.asfreq("M", "end").dt.to_timestamp()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.nanmean(np.abs(actual - predicted)))


def dm_test(loss_model: np.ndarray, loss_arma: np.ndarray) -> Tuple[float, float]:
    """Two-sided Diebold-Mariano test for equal squared-error loss."""
    mask = np.isfinite(loss_model) & np.isfinite(loss_arma)
    d = loss_model[mask] - loss_arma[mask]
    if len(d) < 5:
        return (np.nan, np.nan)
    var_d = np.var(d, ddof=1) / len(d)
    if not np.isfinite(var_d) or var_d <= 0:
        return (0.0, 1.0)
    stat = float(d.mean() / math.sqrt(var_d))
    pval = float(2 * (1 - norm.cdf(abs(stat))))
    return stat, pval


def load_prediction(config: CountryConfig, model: str, vintage: str) -> pd.DataFrame:
    path = prediction_path(config, model, vintage)
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = normalize_quarter_dates(df["date"])
    return df


def audit_file(config: CountryConfig, model: str, vintage: str) -> List[str]:
    issues: List[str] = []
    path = prediction_path(config, model, vintage)
    label = f"{config.code}:{model}:{vintage}"

    if not os.path.exists(path):
        return [f"{label}: missing file {path}"]

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # pragma: no cover - diagnostic path
        return [f"{label}: could not read CSV: {exc}"]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"{label}: missing columns {missing_cols}")
        return issues

    df["date"] = normalize_quarter_dates(df["date"])

    if len(df) != config.expected_rows:
        issues.append(f"{label}: expected {config.expected_rows} rows, found {len(df)}")

    duplicate_count = int(df["date"].duplicated().sum())
    if duplicate_count:
        issues.append(f"{label}: date has {duplicate_count} duplicate rows")

    if len(df) > 0:
        if df["date"].iloc[0] != pd.Timestamp(config.expected_start):
            issues.append(
                f"{label}: expected first date {config.expected_start}, "
                f"found {df['date'].iloc[0].date()}"
            )
        if df["date"].iloc[-1] != pd.Timestamp(config.expected_end):
            issues.append(
                f"{label}: expected last date {config.expected_end}, "
                f"found {df['date'].iloc[-1].date()}"
            )

    expected = expected_dates(config)
    if len(df) == len(expected) and not df["date"].reset_index(drop=True).equals(pd.Series(expected)):
        issues.append(f"{label}: dates do not match expected quarterly sequence")

    for col in ["actual", "prediction"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.isna() | ~np.isfinite(vals)
        if bad.any():
            issues.append(f"{label}: {col} has {int(bad.sum())} missing/non-finite values")

    return issues


def audit_country(config: CountryConfig) -> None:
    issues: List[str] = []
    for model in MODELS:
        for vintage in config.vintages:
            issues.extend(audit_file(config, model, vintage))

    if issues:
        message = "\n".join(issues)
        raise RuntimeError(f"Prediction audit failed for {config.label}:\n{message}")


def evaluate_country(config: CountryConfig) -> pd.DataFrame:
    audit_country(config)

    rows = []
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for model in MODELS:
        for vintage in config.vintages:
            df = load_prediction(config, model, vintage)
            df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
            cache[(model, vintage)] = df

            for panel_name, (start, end) in config.panels.items():
                panel = df[(df["date"] >= start) & (df["date"] <= end)]
                actual = panel["actual"].to_numpy()
                pred = panel["prediction"].to_numpy()
                rows.append(
                    {
                        "country": config.code,
                        "model": model,
                        "vintage": vintage,
                        "panel": panel_name,
                        "n_obs": len(panel),
                        "RMSFE": rmse(actual, pred),
                        "MAE": mae(actual, pred),
                    }
                )

    results = pd.DataFrame(rows)

    arma = results[results["model"] == "arma"][
        ["vintage", "panel", "RMSFE"]
    ].rename(columns={"RMSFE": "arma_RMSFE"})
    results = results.merge(arma, on=["vintage", "panel"], how="left")
    results["rel_RMSFE_vs_ARMA"] = results["RMSFE"] / results["arma_RMSFE"]
    results = results.drop(columns=["arma_RMSFE"])

    dm_rows = []
    for model in MODELS:
        if model == "arma":
            continue
        for vintage in config.vintages:
            df_model = cache[(model, vintage)]
            df_arma = cache[("arma", vintage)]
            merged = df_model.merge(df_arma, on="date", suffixes=("_m", "_a"))
            for panel_name, (start, end) in config.panels.items():
                panel = merged[(merged["date"] >= start) & (merged["date"] <= end)]
                loss_model = (panel["actual_m"] - panel["prediction_m"]) ** 2
                loss_arma = (panel["actual_a"] - panel["prediction_a"]) ** 2
                stat, pval = dm_test(loss_model.to_numpy(), loss_arma.to_numpy())
                dm_rows.append(
                    {
                        "model": model,
                        "vintage": vintage,
                        "panel": panel_name,
                        "dm_stat_vs_ARMA": stat,
                        "dm_pval_vs_ARMA": pval,
                    }
                )

    dm = pd.DataFrame(dm_rows)
    results = results.merge(dm, on=["model", "vintage", "panel"], how="left")
    results.to_csv(config.output_path, index=False)
    return results


def format_top_table(results: pd.DataFrame, country: str) -> List[str]:
    full_m3 = results[(results["panel"] == "full") & (results["vintage"] == "m3")]
    full_m3 = full_m3.sort_values("RMSFE")
    lines = [
        f"### {country}: Full Panel, m3 Vintage",
        "",
        "| Rank | Model | RMSFE | MAE | Relative RMSFE vs ARMA |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(full_m3.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.model} | {row.RMSFE:.6f} | "
            f"{row.MAE:.6f} | {row.rel_RMSFE_vs_ARMA:.3f} |"
        )
    lines.append("")
    return lines


def write_summary(all_results: Dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        "Generated from prediction CSVs by `python data/evaluate.py`.",
        "",
        "Reproducible figures are generated by `python data/generate_figures.py`; see `figures/FIGURE_INDEX.md`.",
        "",
        "## Scope Notes",
        "",
        "- US BVAR uses a Lasso-80 reduced predictor set because full Cat3 BVAR is computationally infeasible in `mfbvar`.",
        "- US BVAR 2025-Q4 m3 and post1 use documented Cat2 fallbacks because the reduced BVAR covariance matrix is singular at those vintages.",
        "- Turkey BVAR uses the locked Cat2 predictor set plus the GDP target for the same computational reason.",
        "- Turkey DFM uses a validation-selected Cat2 monthly predictor set plus target; Cat2 was selected on 2012-2017 validation RMSFE over m1/m2/m3, then retrained through 2017 for 2018-2025 testing. Tier-C variables remain excluded after `nowcastDFM` failed on the full sparse panel.",
        "- Turkey MIDAS-ML uses documented fixed-penalty `sglfit(lambda = 0.01)` after `cv.sglfit` aborted the Jupyter process during the post-release rerun.",
        "- R MIDAS/MIDAS-ML/DFM/BVAR notebooks use explicit target-quarter and vintage-date masking via `gen_vintage_data`.",
        "- Turkey `post1` and `post2` are robustness horizons, not replacements for the symmetric `m1`/`m2`/`m3` cross-country benchmark.",
        "- Turkey MLP m1/m2 outputs are finite but unstable; interpret them as model instability rather than a headline result. US MLP is stabilized in the current rerun.",
        "- Raw prediction filenames are preserved; Turkey legacy names are handled by the evaluator.",
        "",
        "## Headline Rankings",
        "",
    ]
    lines.extend(format_top_table(all_results["us"], "United States"))
    lines.extend(format_top_table(all_results["tr"], "Turkey"))

    out_path = os.path.join(ROOT, "evaluation_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    all_results: Dict[str, pd.DataFrame] = {}
    for config in COUNTRIES:
        results = evaluate_country(config)
        all_results[config.code] = results
        print(f"{config.label}: wrote {len(results)} rows to {config.output_path}")
    write_summary(all_results)
    print(f"Wrote {os.path.join(ROOT, 'evaluation_summary.md')}")


if __name__ == "__main__":
    main()
