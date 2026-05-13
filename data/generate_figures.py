"""
Generate paper-facing figures from the finalized nowcasting evaluation outputs.

The original Hopp repository ships README figures and appendix plots. This
project needs the same reproducibility layer, adapted to the two-country
US/Turkey design and to the final evaluation CSVs produced by data/evaluate.py.

Usage:
    python data/generate_figures.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
IMAGES_DIR = os.path.join(ROOT, "images")
FIGURES_DIR = os.path.join(ROOT, "figures")

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

MODEL_FAMILIES = {
    "arma": "Classical benchmarks",
    "ols": "Classical benchmarks",
    "var": "Classical benchmarks",
    "bvar": "Bayesian/factor",
    "dfm": "Bayesian/factor",
    "midas": "Mixed-frequency",
    "midasml": "Mixed-frequency",
    "lasso": "Penalized linear",
    "ridge": "Penalized linear",
    "elasticnet": "Penalized linear",
    "dt": "Tree ML",
    "rf": "Tree ML",
    "gb": "Tree ML",
    "xgboost": "Tree ML",
    "mlp": "Neural networks",
    "lstm": "Neural networks",
    "deepvar": "Neural networks",
}

FAMILY_COLORS = {
    "Classical benchmarks": "#4c78a8",
    "Bayesian/factor": "#f58518",
    "Mixed-frequency": "#54a24b",
    "Penalized linear": "#b279a2",
    "Tree ML": "#e45756",
    "Neural networks": "#72b7b2",
}


@dataclass(frozen=True)
class Country:
    code: str
    label: str
    eval_path: str
    prediction_dir: str
    turkey_filenames: bool = False


COUNTRIES = [
    Country(
        code="us",
        label="United States",
        eval_path=os.path.join(BASE, "evaluation_results_us.csv"),
        prediction_dir=os.path.join(ROOT, "predictions"),
    ),
    Country(
        code="tr",
        label="Turkey",
        eval_path=os.path.join(ROOT, "turkey_data", "evaluation_results_tr.csv"),
        prediction_dir=os.path.join(ROOT, "turkey_predictions"),
        turkey_filenames=True,
    ),
]


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


def ensure_dirs() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def prediction_path(country: Country, model: str, vintage: str) -> str:
    if country.turkey_filenames:
        preferred = os.path.join(country.prediction_dir, f"{model}_tr_{vintage}.csv")
        legacy = os.path.join(country.prediction_dir, f"{model}_{vintage}.csv")
        return preferred if os.path.exists(preferred) else legacy
    return os.path.join(country.prediction_dir, f"{model}_{vintage}.csv")


def load_prediction(country: Country, model: str, vintage: str) -> pd.DataFrame:
    path = prediction_path(country, model, vintage)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "actual", "prediction"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["model"] = model
    df["vintage"] = vintage
    df["country"] = country.code
    return df.sort_values("date")


def load_eval(country: Country) -> pd.DataFrame:
    df = pd.read_csv(country.eval_path)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["family"] = df["model"].map(MODEL_FAMILIES)
    return df


def full_m3(eval_df: pd.DataFrame) -> pd.DataFrame:
    return eval_df[(eval_df["panel"] == "full") & (eval_df["vintage"] == "m3")].copy()


def revision_volatility(country: Country, model: str) -> float:
    frames = [
        load_prediction(country, model, vintage)[["date", "prediction"]].rename(
            columns={"prediction": vintage}
        )
        for vintage in ["m1", "m2", "m3"]
    ]
    merged = frames[0].merge(frames[1], on="date").merge(frames[2], on="date")
    revisions = pd.concat(
        [(merged["m2"] - merged["m1"]).abs(), (merged["m3"] - merged["m2"]).abs()],
        axis=0,
    )
    return float(revisions.mean())


def save_data_example() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, country in zip(axes, COUNTRIES):
        actual = load_prediction(country, "arma", "m3")[["date", "actual"]]
        ax.plot(actual["date"], actual["actual"], color="#2f2f2f", lw=1.8)
        ax.axhline(0, color="#8a8a8a", lw=0.8)
        ax.set_title(f"{country.label}: real GDP target")
        ax.set_xlabel("Forecast date")
        ax.set_ylabel("Quarterly transformed growth")
    fig.suptitle("Evaluation target series used in the finalized pipeline", y=1.02)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, "data_example.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_revision_scatter(country: Country, eval_df: pd.DataFrame, name: str) -> str:
    df = full_m3(eval_df)
    df["revision_volatility"] = [
        revision_volatility(country, model) for model in df["model"]
    ]
    df = df.sort_values("RMSFE")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for family, color in FAMILY_COLORS.items():
        part = df[df["model"].map(MODEL_FAMILIES) == family]
        if part.empty:
            continue
        ax.scatter(
            part["revision_volatility"],
            part["RMSFE"],
            s=72,
            c=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            label=family,
        )

    label_models = set(df.head(5)["model"])
    label_models.add("arma")
    label_models.add(df.loc[df["RMSFE"].idxmax(), "model"])
    label_models.add(df.loc[df["revision_volatility"].idxmax(), "model"])
    for _, row in df[df["model"].isin(label_models)].iterrows():
        ax.annotate(
            MODEL_LABELS[row["model"]],
            (row["revision_volatility"], row["RMSFE"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
        )
    ax.set_title(f"{country.label}: error vs nowcast revision volatility")
    ax.set_xlabel("Average absolute nowcast revision across m1/m2/m3")
    ax.set_ylabel("Full-panel RMSFE, m3 vintage")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_appendix_model_plot(model: str, index: int) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    vintage_colors = {"m1": "#4c78a8", "m2": "#f58518", "m3": "#54a24b"}

    for ax, country in zip(axes, COUNTRIES):
        frames = {v: load_prediction(country, model, v) for v in ["m1", "m2", "m3"]}
        base = frames["m3"]
        ax.plot(base["date"], base["actual"], color="#1f1f1f", lw=2.0, label="Actual")
        for vintage, df in frames.items():
            ax.plot(
                df["date"],
                df["prediction"],
                color=vintage_colors[vintage],
                lw=1.25,
                alpha=0.9,
                label=vintage,
            )
        ax.axhline(0, color="#9a9a9a", lw=0.7)
        ax.set_title(country.label)
        ax.set_ylabel("Growth")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Forecast date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{MODEL_LABELS[model]}: actual vs nowcasts by vintage", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(IMAGES_DIR, f"app{index}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_full_m3_rankings(eval_us: pd.DataFrame, eval_tr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=False)
    for ax, country, df in zip(axes, COUNTRIES, [full_m3(eval_us), full_m3(eval_tr)]):
        df = df.sort_values("RMSFE", ascending=True)
        y = np.arange(len(df))
        colors = [FAMILY_COLORS[MODEL_FAMILIES[m]] for m in df["model"]]
        ax.barh(y, df["RMSFE"], color=colors, alpha=0.9)
        ax.set_yticks(y, [MODEL_LABELS[m] for m in df["model"]])
        ax.invert_yaxis()
        ax.set_title(country.label)
        ax.set_xlabel("RMSFE")
        for idx, value in enumerate(df["RMSFE"]):
            ax.text(value, idx, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Full-panel m3 RMSFE ranking")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "full_m3_rmsfe_rankings.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_relative_comparison(eval_us: pd.DataFrame, eval_tr: pd.DataFrame) -> str:
    us = full_m3(eval_us)[["model", "rel_RMSFE_vs_ARMA"]].rename(
        columns={"rel_RMSFE_vs_ARMA": "United States"}
    )
    tr = full_m3(eval_tr)[["model", "rel_RMSFE_vs_ARMA"]].rename(
        columns={"rel_RMSFE_vs_ARMA": "Turkey"}
    )
    df = us.merge(tr, on="model")
    df["average"] = df[["United States", "Turkey"]].mean(axis=1)
    df = df.sort_values("average")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["United States"], width, label="United States", color="#4c78a8")
    ax.bar(x + width / 2, df["Turkey"], width, label="Turkey", color="#f58518")
    ax.axhline(1.0, color="#333333", lw=1.0, linestyle="--", label="ARMA parity")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in df["model"]], rotation=45, ha="right")
    ax.set_ylabel("Relative RMSFE vs ARMA")
    ax.set_title("Cross-country relative performance, full-panel m3")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "full_m3_relative_rmsfe_us_tr.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def heatmap_matrix(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = eval_df[eval_df["vintage"] == "m3"].copy()
    order = (
        df[df["panel"] == "full"]
        .sort_values("rel_RMSFE_vs_ARMA")["model"]
        .tolist()
    )
    matrix = df.pivot(index="model", columns="panel", values="rel_RMSFE_vs_ARMA")
    preferred_cols = [col for col in ["pre_covid", "pre_crisis", "covid", "post_covid", "full"] if col in matrix.columns]
    return matrix.loc[order, preferred_cols]


def save_panel_heatmaps(eval_us: pd.DataFrame, eval_tr: pd.DataFrame) -> str:
    matrices = [heatmap_matrix(eval_us), heatmap_matrix(eval_tr)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=False, sharey=False)
    vmin = min(float(m.min().min()) for m in matrices)
    vmax = max(float(m.max().max()) for m in matrices)
    for ax, country, matrix in zip(axes, COUNTRIES, matrices):
        im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"{country.label}: m3 relative RMSFE by panel")
        ax.set_yticks(np.arange(len(matrix.index)), [MODEL_LABELS[m] for m in matrix.index])
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{matrix.iloc[y, x]:.2f}", ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.18, wspace=0.35)
    cbar_ax = fig.add_axes([0.91, 0.24, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Relative RMSFE vs ARMA")
    fig.suptitle("Panel robustness of m3 nowcasts")
    out = os.path.join(FIGURES_DIR, "panel_relative_rmsfe_heatmaps.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_vintage_profiles(eval_us: pd.DataFrame, eval_tr: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=False)
    for ax, country, eval_df in zip(axes, COUNTRIES, [eval_us, eval_tr]):
        top = full_m3(eval_df).sort_values("RMSFE").head(5)["model"].tolist()
        selected = ["arma"] + [m for m in top if m != "arma"]
        for model in selected:
            df = eval_df[
                (eval_df["panel"] == "full")
                & (eval_df["model"] == model)
                & (eval_df["vintage"].isin(["m1", "m2", "m3"]))
            ].sort_values("vintage")
            ax.plot(
                df["vintage"],
                df["RMSFE"],
                marker="o",
                lw=1.8,
                label=MODEL_LABELS[model],
            )
        ax.set_title(country.label)
        ax.set_xlabel("Vintage")
        ax.set_ylabel("Full-panel RMSFE")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("Full-panel RMSFE profiles across vintages")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "vintage_rmsfe_profiles.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_family_summary(eval_us: pd.DataFrame, eval_tr: pd.DataFrame) -> str:
    records = []
    for country, eval_df in zip(COUNTRIES, [eval_us, eval_tr]):
        df = full_m3(eval_df).copy()
        df["family"] = df["model"].map(MODEL_FAMILIES)
        grouped = df.groupby("family", as_index=False)["rel_RMSFE_vs_ARMA"].mean()
        grouped["country"] = country.label
        records.append(grouped)
    out_df = pd.concat(records, ignore_index=True)
    pivot = out_df.pivot(index="family", columns="country", values="rel_RMSFE_vs_ARMA")
    pivot = pivot.loc[
        [
            "Classical benchmarks",
            "Bayesian/factor",
            "Mixed-frequency",
            "Penalized linear",
            "Tree ML",
            "Neural networks",
        ]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.38
    ax.bar(x - width / 2, pivot["United States"], width, label="United States", color="#4c78a8")
    ax.bar(x + width / 2, pivot["Turkey"], width, label="Turkey", color="#f58518")
    ax.axhline(1.0, color="#333333", lw=1.0, linestyle="--")
    ax.set_xticks(x, pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Average relative RMSFE vs ARMA")
    ax.set_title("Model-family performance, full-panel m3")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "model_family_relative_rmsfe.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_figure_index(paths: Iterable[str], appendix_models: List[str]) -> str:
    index_path = os.path.join(FIGURES_DIR, "FIGURE_INDEX.md")
    rel_paths = [os.path.relpath(path, ROOT).replace("\\", "/") for path in paths]
    appendix_lines = [
        f"- `images/app{i}.png`: {MODEL_LABELS[model]} actual-vs-nowcast appendix plot."
        for i, model in enumerate(appendix_models, start=1)
    ]
    body = [
        "# Figure Index",
        "",
        "Generated by `python data/generate_figures.py` from finalized prediction and evaluation CSVs.",
        "",
        "## Main Paper Figures",
        "",
        "- `figures/full_m3_rmsfe_rankings.png`: country-specific full-panel m3 RMSFE rankings.",
        "- `figures/full_m3_relative_rmsfe_us_tr.png`: cross-country relative RMSFE vs ARMA.",
        "- `figures/panel_relative_rmsfe_heatmaps.png`: panel robustness for m3 relative RMSFE.",
        "- `figures/vintage_rmsfe_profiles.png`: full-panel RMSFE across m1/m2/m3 for ARMA and top models.",
        "- `figures/model_family_relative_rmsfe.png`: family-level average relative RMSFE.",
        "",
        "## Original-Repo-Style README Figures",
        "",
        "- `images/data_example.png`: target-series overview for the evaluation windows.",
        "- `images/fig1.png`: United States RMSFE vs nowcast revision volatility.",
        "- `images/fig2.png`: Turkey RMSFE vs nowcast revision volatility.",
        "",
        "The scatter x-axis is average absolute nowcast revision across m1/m2/m3 for the same forecast date. It is not historical GDP data-revision volatility.",
        "",
        "## Appendix Nowcast Plots",
        "",
        *appendix_lines,
        "",
        "## Files Written",
        "",
        *[f"- `{path}`" for path in rel_paths],
        "",
    ]
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    return index_path


def main() -> None:
    setup_plotting()
    ensure_dirs()

    eval_us = load_eval(COUNTRIES[0])
    eval_tr = load_eval(COUNTRIES[1])

    paths: List[str] = []
    paths.append(save_data_example())
    paths.append(save_revision_scatter(COUNTRIES[0], eval_us, "fig1.png"))
    paths.append(save_revision_scatter(COUNTRIES[1], eval_tr, "fig2.png"))

    appendix_models = sorted(MODELS)
    for i, model in enumerate(appendix_models, start=1):
        paths.append(save_appendix_model_plot(model, i))

    paths.append(save_full_m3_rankings(eval_us, eval_tr))
    paths.append(save_relative_comparison(eval_us, eval_tr))
    paths.append(save_panel_heatmaps(eval_us, eval_tr))
    paths.append(save_vintage_profiles(eval_us, eval_tr))
    paths.append(save_family_summary(eval_us, eval_tr))
    index_path = write_figure_index(paths, appendix_models)

    print(f"Wrote {len(paths)} figure files")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
