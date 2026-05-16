"""
Generate result-analysis figures from finalized evaluation outputs.

These figures complement docs/results_analysis.md and focus on period,
vintage, post-release, COVID, and DFM-validation distinctions.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
FIGURES_DIR = os.path.join(ROOT, "figures")

MODEL_LABELS = {
    "arma": "ARMA",
    "ols": "OLS",
    "var": "VAR",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "gb": "Gradient Boosting",
    "dt": "Decision Tree",
    "mlp": "MLP",
    "lstm": "LSTM",
    "deepvar": "DeepVAR",
    "bvar": "BVAR",
    "midas": "MIDAS",
    "midasml": "MIDAS-ML",
    "dfm": "DFM",
}

COUNTRY_LABELS = {"us": "United States", "tr": "Turkey"}
PANEL_LABELS = {
    "pre_covid": "Pre-COVID",
    "pre_crisis": "Pre-crisis",
    "covid": "COVID",
    "post_covid": "Post-COVID",
    "full": "Full",
}


def setup_plotting() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    us = pd.read_csv(os.path.join(BASE, "evaluation_results_us.csv"))
    tr = pd.read_csv(os.path.join(ROOT, "turkey_data", "evaluation_results_tr.csv"))
    us_improved = pd.read_csv(os.path.join(BASE, "evaluation_results_us_improved.csv"))
    return us, tr, us_improved


def label_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def save_period_rankings(us: pd.DataFrame, tr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=False)
    configs = [
        ("us", us, ["pre_covid", "covid", "post_covid", "full"]),
        ("tr", tr, ["pre_crisis", "covid", "post_covid", "full"]),
    ]
    for row, (country, df, panels) in enumerate(configs):
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            sub = (
                df[(df["panel"] == panel) & (df["vintage"] == "m3")]
                .sort_values("RMSFE")
                .head(6)
                .iloc[::-1]
            )
            colors = ["#4c78a8" if country == "us" else "#f58518"] * len(sub)
            ax.barh([label_model(m) for m in sub["model"]], sub["RMSFE"], color=colors)
            ax.set_title(f"{COUNTRY_LABELS[country]}: {PANEL_LABELS[panel]}")
            ax.set_xlabel("RMSFE")
            for y, value in enumerate(sub["RMSFE"]):
                ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Top m3 Models by Evaluation Period", y=0.995)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_period_rankings_m3.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_vintage_gain(us: pd.DataFrame, tr: pd.DataFrame) -> str:
    rows: List[Dict[str, float | str]] = []
    for country, df in [("us", us), ("tr", tr)]:
        for model in sorted(df["model"].unique()):
            full = df[(df["panel"] == "full") & (df["model"] == model)].set_index("vintage")
            if {"m1", "m3"} <= set(full.index):
                gain = (full.loc["m1", "RMSFE"] - full.loc["m3", "RMSFE"]) / full.loc["m1", "RMSFE"] * 100
                rows.append({"country": country, "model": model, "gain": gain})
    gain_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=False)
    for ax, country in zip(axes, ["us", "tr"]):
        sub = gain_df[gain_df["country"] == country].sort_values("gain", ascending=False).head(12).iloc[::-1]
        color = "#4c78a8" if country == "us" else "#f58518"
        ax.barh([label_model(m) for m in sub["model"]], sub["gain"], color=color)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(COUNTRY_LABELS[country])
        ax.set_xlabel("RMSFE improvement from m1 to m3 (%)")
        for y, value in enumerate(sub["gain"]):
            ax.text(value, y, f" {value:.1f}%", va="center", fontsize=8)
    fig.suptitle("Information Gain Across Vintages")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_vintage_gain_m1_to_m3.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_post_release(us: pd.DataFrame, tr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=False)
    panels = [
        ("US post1", us, "post1", "#4c78a8"),
        ("Turkey post1", tr, "post1", "#f58518"),
        ("Turkey post2", tr, "post2", "#e45756"),
    ]
    for ax, (title, df, vintage, color) in zip(axes, panels):
        sub = df[(df["panel"] == "full") & (df["vintage"] == vintage)].sort_values("RMSFE").head(8).iloc[::-1]
        ax.barh([label_model(m) for m in sub["model"]], sub["RMSFE"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Full-sample RMSFE")
        for y, value in enumerate(sub["RMSFE"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Post-Release Robustness Horizons")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_post_release_rankings.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_covid_sensitivity(us: pd.DataFrame, tr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=False)
    for ax, country, df in zip(axes, ["us", "tr"], [us, tr]):
        sub = df[(df["panel"] == "covid") & (df["vintage"] == "m3")].sort_values("RMSFE").head(10).iloc[::-1]
        color = "#4c78a8" if country == "us" else "#f58518"
        ax.barh([label_model(m) for m in sub["model"]], sub["RMSFE"], color=color)
        ax.set_title(COUNTRY_LABELS[country])
        ax.set_xlabel("COVID-period RMSFE, m3")
        for y, value in enumerate(sub["RMSFE"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("COVID Stress-Test Rankings")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_covid_sensitivity_m3.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_top3_robustness(us: pd.DataFrame, tr: pd.DataFrame) -> str:
    rows: List[Dict[str, int | str]] = []
    for country, df, panels in [
        ("us", us, ["pre_covid", "covid", "post_covid", "full"]),
        ("tr", tr, ["pre_crisis", "covid", "post_covid", "full"]),
    ]:
        counts = {model: 0 for model in df["model"].unique()}
        for panel in panels:
            top = df[(df["panel"] == panel) & (df["vintage"] == "m3")].sort_values("RMSFE").head(3)
            for model in top["model"]:
                counts[model] += 1
        for model, count in counts.items():
            if count:
                rows.append({"country": country, "model": model, "count": count})
    count_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for ax, country in zip(axes, ["us", "tr"]):
        sub = count_df[count_df["country"] == country].sort_values(["count", "model"], ascending=[True, False])
        color = "#4c78a8" if country == "us" else "#f58518"
        ax.barh([label_model(m) for m in sub["model"]], sub["count"], color=color)
        ax.set_title(COUNTRY_LABELS[country])
        ax.set_xlabel("Top-3 appearances across period panels")
        ax.set_xticks([0, 1, 2, 3, 4])
    fig.suptitle("Model Robustness Across Evaluation Periods")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_top3_period_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_dfm_validation() -> str | None:
    path = os.path.join(ROOT, "archive", "logs", "turkey_dfm_validation_selection", "selection_table.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("selection_RMSFE").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(df["spec"], df["selection_RMSFE"], color="#72b7b2")
    ax.set_xlabel("Validation RMSFE, average across m1/m2/m3")
    ax.set_title("Turkey DFM Validation Selection")
    for y, value in enumerate(df["selection_RMSFE"]):
        ax.text(value, y, f" {value:.4f}", va="center")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_dfm_validation_selection.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_us_combinations(us_improved: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=False)
    configs = [("full", "Full sample"), ("full_ex_2020q2", "Full excluding 2020-Q2")]
    for ax, (panel, title) in zip(axes, configs):
        sub = (
            us_improved[(us_improved["panel"] == panel) & (us_improved["vintage"] == "m3")]
            .sort_values("RMSFE")
            .head(10)
            .iloc[::-1]
        )
        colors = ["#54a24b" if str(m).startswith("combo_") else "#4c78a8" for m in sub["model"]]
        labels = [m.replace("combo_", "combo: ") for m in sub["model"]]
        ax.barh(labels, sub["RMSFE"], color=colors)
        ax.set_title(title)
        ax.set_xlabel("RMSFE")
        for y, value in enumerate(sub["RMSFE"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("US Forecast-Combination Robustness")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "results_us_combination_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def update_index(paths: Iterable[str]) -> None:
    rel_paths = [os.path.relpath(path, ROOT).replace("\\", "/") for path in paths if path]
    with open(os.path.join(FIGURES_DIR, "RESULTS_FIGURE_INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("# Results Figure Index\n\n")
        fh.write("Generated by `python data/generate_results_visuals.py`.\n\n")
        for rel in rel_paths:
            fh.write(f"- `{rel}`\n")


def main() -> None:
    setup_plotting()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    us, tr, us_improved = load_results()
    paths: List[str | None] = [
        save_period_rankings(us, tr),
        save_vintage_gain(us, tr),
        save_post_release(us, tr),
        save_covid_sensitivity(us, tr),
        save_top3_robustness(us, tr),
        save_dfm_validation(),
        save_us_combinations(us_improved),
    ]
    update_index([p for p in paths if p])
    print(f"Wrote {sum(p is not None for p in paths)} results figures")


if __name__ == "__main__":
    main()
