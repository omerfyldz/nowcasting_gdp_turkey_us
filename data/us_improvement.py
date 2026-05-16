"""
US-only empirical improvements for the nowcasting benchmark.

This script intentionally does not touch Turkey outputs. It keeps the original
17-model prediction files as source data, adds transparent forecast-combination
predictions for the United States, and evaluates both individual and combination
models on robustness panels that are useful for the paper.

Usage:
    python data/us_improvement.py
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PRED_DIR = os.path.join(ROOT, "predictions")
OUT_EVAL = os.path.join(BASE, "evaluation_results_us_improved.csv")
OUT_DIAG = os.path.join(ROOT, "docs", "us_model_diagnostics.md")
FIG_DIR = os.path.join(ROOT, "figures")

BASE_MODELS = [
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

VINTAGES = ["m1", "m2", "m3", "post1"]

COMBINATIONS: Dict[str, Tuple[str, ...]] = {
    "combo_top3_lstm_bvar_midas": ("lstm", "bvar", "midas"),
    "combo_econometric_bvar_midas_dfm": ("bvar", "midas", "dfm"),
    "combo_ml_lstm_rf_gb_xgboost_mlp": ("lstm", "rf", "gb", "xgboost", "mlp"),
    "combo_linear_regularized": ("lasso", "ridge", "elasticnet"),
    "combo_all_median": tuple(BASE_MODELS),
    "combo_all_trimmed_mean": tuple(BASE_MODELS),
    "combo_no_mlp_median": tuple(m for m in BASE_MODELS if m != "mlp"),
}

PANELS: Dict[str, Tuple[str, str]] = {
    "pre_covid": ("2017-01-01", "2019-12-31"),
    "covid": ("2020-04-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2025-12-31"),
    "full": ("2017-01-01", "2025-12-31"),
    "full_ex_2020q2": ("2017-01-01", "2025-12-31"),
    "non_covid": ("2017-01-01", "2025-12-31"),
}


def normalize_quarter_dates(dates: pd.Series) -> pd.Series:
    periods = pd.to_datetime(dates).dt.to_period("Q")
    return periods.dt.asfreq("M", "end").dt.to_timestamp()


def prediction_path(model: str, vintage: str) -> str:
    return os.path.join(PRED_DIR, f"{model}_{vintage}.csv")


def load_prediction(model: str, vintage: str) -> pd.DataFrame:
    path = prediction_path(model, vintage)
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = normalize_quarter_dates(df["date"])
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    return df[["date", "actual", "prediction"]].sort_values("date").reset_index(drop=True)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))) if mask.any() else np.nan


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.mean(np.abs(actual[mask] - pred[mask]))) if mask.any() else np.nan


def dm_test(loss_model: np.ndarray, loss_arma: np.ndarray) -> Tuple[float, float]:
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


def write_combination_predictions() -> List[str]:
    written: List[str] = []
    for vintage in VINTAGES:
        base_frames = {
            model: load_prediction(model, vintage).rename(columns={"prediction": model})
            for model in BASE_MODELS
        }
        merged = base_frames["arma"][["date", "actual", "arma"]].copy()
        for model, frame in base_frames.items():
            if model == "arma":
                continue
            keep = frame[["date", model]]
            merged = merged.merge(keep, on="date", how="inner")

        for combo, members in COMBINATIONS.items():
            values = merged[list(members)].to_numpy(dtype=float)
            if combo.endswith("trimmed_mean"):
                sorted_values = np.sort(values, axis=1)
                if sorted_values.shape[1] > 2:
                    pred = np.nanmean(sorted_values[:, 1:-1], axis=1)
                else:
                    pred = np.nanmean(sorted_values, axis=1)
            elif combo.endswith("median"):
                pred = np.nanmedian(values, axis=1)
            else:
                pred = np.nanmean(values, axis=1)

            out = merged[["date", "actual"]].copy()
            out["prediction"] = pred
            path = prediction_path(combo, vintage)
            out.to_csv(path, index=False)
            written.append(path)
    return written


def panel_mask(df: pd.DataFrame, panel: str, start: str, end: str) -> pd.Series:
    mask = (df["date"] >= start) & (df["date"] <= end)
    if panel == "full_ex_2020q2":
        mask &= df["date"] != pd.Timestamp("2020-06-01")
    elif panel == "non_covid":
        mask &= ~((df["date"] >= "2020-04-01") & (df["date"] <= "2021-12-31"))
    return mask


def evaluate_models(models: Iterable[str]) -> pd.DataFrame:
    rows = []
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for model in models:
        for vintage in VINTAGES:
            df = load_prediction(model, vintage)
            cache[(model, vintage)] = df
            for panel, (start, end) in PANELS.items():
                part = df[panel_mask(df, panel, start, end)]
                actual = part["actual"].to_numpy()
                pred = part["prediction"].to_numpy()
                rows.append(
                    {
                        "country": "us",
                        "model": model,
                        "vintage": vintage,
                        "panel": panel,
                        "n_obs": len(part),
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
    for model in models:
        if model == "arma":
            continue
        for vintage in VINTAGES:
            merged = cache[(model, vintage)].merge(
                cache[("arma", vintage)], on="date", suffixes=("_m", "_a")
            )
            for panel, (start, end) in PANELS.items():
                part = merged[panel_mask(merged, panel, start, end)]
                loss_model = (part["actual_m"] - part["prediction_m"]) ** 2
                loss_arma = (part["actual_a"] - part["prediction_a"]) ** 2
                stat, pval = dm_test(loss_model.to_numpy(), loss_arma.to_numpy())
                dm_rows.append(
                    {
                        "model": model,
                        "vintage": vintage,
                        "panel": panel,
                        "dm_stat_vs_ARMA": stat,
                        "dm_pval_vs_ARMA": pval,
                    }
                )
    results = results.merge(pd.DataFrame(dm_rows), on=["model", "vintage", "panel"], how="left")
    return results


def worst_errors(model: str, vintage: str, top_n: int = 6) -> pd.DataFrame:
    df = load_prediction(model, vintage)
    df["error"] = df["prediction"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    return df.sort_values("abs_error", ascending=False).head(top_n)


def ranking_table(results: pd.DataFrame, vintage: str, panel: str, top_n: int | None = None) -> pd.DataFrame:
    df = results[(results["vintage"] == vintage) & (results["panel"] == panel)].copy()
    df = df.sort_values("RMSFE").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    if top_n:
        df = df.head(top_n)
    return df


def write_diagnostics(results: pd.DataFrame, written: List[str]) -> None:
    os.makedirs(os.path.dirname(OUT_DIAG), exist_ok=True)
    lines = [
        "# US Model Diagnostics and Improvements",
        "",
        "Generated by `python data/us_improvement.py`.",
        "",
        "## Files Written",
        "",
        f"- `data/evaluation_results_us_improved.csv`: US-only evaluation including combinations and robustness panels.",
    ]
    lines.extend(
        f"- `{os.path.relpath(path, ROOT).replace(os.sep, '/')}`" for path in written
    )

    lines.extend(
        [
            "",
            "## Full-Sample m3 Ranking Including Combinations",
            "",
            "| Rank | Model | RMSFE | MAE | Relative RMSFE vs ARMA |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in ranking_table(results, "m3", "full").itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.model} | {row.RMSFE:.6f} | {row.MAE:.6f} | {row.rel_RMSFE_vs_ARMA:.3f} |"
        )

    lines.extend(
        [
            "",
            "## COVID-Outlier Robustness: Full Sample Excluding 2020-Q2",
            "",
            "| Rank | Model | RMSFE | MAE | Relative RMSFE vs ARMA |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in ranking_table(results, "m3", "full_ex_2020q2", top_n=12).itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.model} | {row.RMSFE:.6f} | {row.MAE:.6f} | {row.rel_RMSFE_vs_ARMA:.3f} |"
        )

    lines.extend(
        [
            "",
            "## DFM Worst Errors",
            "",
            "DFM is first in `m1` but deteriorates in later vintages. The table below shows that the deterioration is concentrated in COVID observations, especially 2020-Q2 and 2020-Q3.",
            "",
            "| Vintage | Date | Actual | Prediction | Error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for vintage in VINTAGES:
        for row in worst_errors("dfm", vintage, top_n=4).itertuples(index=False):
            lines.append(
                f"| {vintage} | {row.date.date()} | {row.actual:.6f} | {row.prediction:.6f} | {row.error:.6f} |"
            )

    lines.extend(
        [
            "",
            "## MLP Worst Errors",
            "",
            "MLP remains unstable in early vintages. It is included in the benchmark, but paper conclusions should not rely on m1/m2 MLP performance.",
            "",
            "| Vintage | Date | Actual | Prediction | Error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for vintage in VINTAGES:
        for row in worst_errors("mlp", vintage, top_n=4).itertuples(index=False):
            lines.append(
                f"| {vintage} | {row.date.date()} | {row.actual:.6f} | {row.prediction:.6f} | {row.error:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The original US headline result is robust: LSTM, BVAR, and MIDAS remain among the strongest individual m3 models.",
            "- Simple forecast combinations are now available as robustness benchmarks. These should be discussed separately from the 17 pre-specified models.",
            "- DFM weakness is not uniform across the sample; it is primarily a COVID-shock sensitivity issue in later vintages.",
            "- MLP early-vintage instability persists and should be presented as a limitation of generic neural networks in small macro samples.",
            "- The `full_ex_2020q2` and `non_covid` panels make the paper less dependent on a single extreme GDP observation.",
        ]
    )

    with open(OUT_DIAG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def write_figures(results: pd.DataFrame) -> List[str]:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    paths: List[str] = []

    top = ranking_table(results, "m3", "full", top_n=15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["model"][::-1], top["RMSFE"][::-1], color="#4c78a8")
    ax.set_xlabel("RMSFE")
    ax.set_title("United States: m3 RMSFE ranking with forecast combinations")
    for y, value in enumerate(top["RMSFE"][::-1]):
        ax.text(value, y, f" {value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "us_improved_m3_ranking.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    robust = ranking_table(results, "m3", "full_ex_2020q2", top_n=15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(robust["model"][::-1], robust["RMSFE"][::-1], color="#54a24b")
    ax.set_xlabel("RMSFE")
    ax.set_title("United States: m3 RMSFE excluding 2020-Q2")
    for y, value in enumerate(robust["RMSFE"][::-1]):
        ax.text(value, y, f" {value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "us_improved_m3_ex_2020q2_ranking.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    profile_models = [
        "arma",
        "lstm",
        "bvar",
        "midas",
        "combo_top3_lstm_bvar_midas",
        "combo_econometric_bvar_midas_dfm",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in profile_models:
        df = results[
            (results["model"] == model)
            & (results["panel"] == "full")
            & (results["vintage"].isin(VINTAGES))
        ].sort_values("vintage")
        ax.plot(df["vintage"], df["RMSFE"], marker="o", lw=1.8, label=model)
    ax.set_title("United States: full-sample RMSFE by vintage")
    ax.set_ylabel("RMSFE")
    ax.set_xlabel("Vintage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "us_improved_vintage_profiles.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def main() -> None:
    written = write_combination_predictions()
    models = BASE_MODELS + list(COMBINATIONS.keys())
    results = evaluate_models(models)
    results.to_csv(OUT_EVAL, index=False)
    write_diagnostics(results, written)
    figure_paths = write_figures(results)
    print(f"Wrote {len(written)} combination prediction files")
    print(f"Wrote {len(figure_paths)} US improvement figures")
    print(f"Wrote {OUT_EVAL} with {len(results)} rows")
    print(f"Wrote {OUT_DIAG}")


if __name__ == "__main__":
    main()
