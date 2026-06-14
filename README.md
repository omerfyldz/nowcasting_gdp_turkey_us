# Comparing GDP Nowcasting Models in the United States and Türkiye

📄 **[Read the full paper (PDF)](<paper .pdf>)**

The paper runs a pseudo-real-time comparison of **seventeen GDP nowcasting models**
under one common framework across two contrasting macroeconomic environments. The
model line-up is held **identical across both countries** while the predictor
panels are country-specific, so the exercise asks not whether a single model wins,
but whether the *ranking of model families* is stable across very different data
environments.

- **United States** — a data-rich advanced economy. Test sample 2017–2025 (36 quarters).
- **Türkiye** — a shorter-history, more volatile emerging market. Test sample 2018–2025 (32 quarters).

## Contents

- [Abstract](#abstract)
- [Main Finding](#main-finding)
- [Data](#data)
- [Model Set](#model-set)
- [Forecast Vintages](#forecast-vintages)
- [Methodology](#methodology)
- [Evaluation Protocol](#evaluation-protocol)
- [Headline Results](#headline-results)
- [Discussion: Why the Rankings Differ](#discussion-why-the-rankings-differ)
- [Figures](#figures)
- [Reproduce the Final Evaluation](#reproduce-the-final-evaluation)
- [Repository Layout](#repository-layout)
- [Limitations and Future Research](#limitations-and-future-research)
- [Important Caveats](#important-caveats)

## Abstract

Official GDP statistics are released with substantial delays, creating an
informational gap for policymakers and market participants who must assess economic
conditions in real time. This paper conducts a pseudo-real-time comparison of
seventeen GDP nowcasting models — spanning classical time-series benchmarks,
penalized linear regressions, tree-based machine learning, neural networks,
mixed-frequency regressions, and Bayesian/factor methods — across the United States
(a data-rich advanced economy) and Türkiye (a shorter-history, more volatile
emerging market). Information availability is simulated through publication-lag and
ragged-edge masking at three standardized within-quarter vintages, while GDP targets
use final revised data.

The central finding is that **model rankings are not portable across countries**. In
the United States, flexible high-dimensional methods perform best: the LSTM roughly
halves the RMSFE of the ARMA benchmark at the late within-quarter vintage, followed
by the MLP and a reduced-dimensional Bayesian VAR. In Türkiye, shrinkage and
parsimonious econometric models dominate: Lasso, ElasticNet, and BVAR lead, while the
LSTM falls behind the univariate benchmark. These advantages are concentrated in
turbulent periods; in calm quarters the differences between models shrink and the
univariate benchmark is hard to beat. Accuracy also improves substantially as
within-quarter information accumulates, with the largest gains going to each
country's best models. The results indicate that data richness, sample length, and
macroeconomic volatility shape the relative usefulness of econometric and
machine-learning nowcasting methods, cautioning against one-size-fits-all model
choices in real-time policy analysis.

## Main Finding

**Model rankings are not portable across countries.**

In the United States, flexible high-dimensional methods perform best: at the most
informative within-quarter vintage the **LSTM** roughly halves the RMSFE of the ARMA
benchmark, followed by the **MLP** and a reduced-dimensional **Bayesian VAR**. In
Türkiye the same line-up produces nearly the opposite ordering: **Lasso**,
**ElasticNet**, and **BVAR** lead, parsimonious **OLS**/**VAR** stay competitive, and
the **LSTM falls behind the univariate ARMA benchmark**.

These advantages are **concentrated in turbulent periods** (the COVID window). In
calm quarters the differences between models shrink and the univariate benchmark is
hard to beat — in Türkiye's post-COVID quarters ARMA is the single most accurate
model. **Data richness, sample length, and macroeconomic volatility** shape the
relative usefulness of econometric vs. machine-learning nowcasting methods,
cautioning against one-size-fits-all model choices.

## Data

The target in both countries is the **quarterly log change in seasonally adjusted
real GDP**, placed at the end-of-quarter month, evaluated against the latest revised
value in the **2026-03** data vintage (pseudo-real-time — not historical
first-release vintages).

| | United States | Türkiye |
|---|---|---|
| GDP target series | `GDPC1` (FRED-QD) | `NGDPRSAXDCTRQ` |
| Predictor sources | FRED-MD, FRED-QD, ~20 manual series (CFNAI, NFCI, ISM PMIs, regional Fed surveys, S&P 500) | CBRT/EVDS, TÜİK, Investing.com (BIST 100) |
| Transformed predictor pool | 296 monthly | 54 monthly |
| Weekly series (mixed-frequency only) | 4 (ICSA, weekly NFCI, trade-weighted USD) | 2 (consumer-loan rate, deposit rate) |
| Training window | 1959-Q1 → 2007-Q4 | 1995-Q1 → 2011-Q4 |
| Validation window | 2008-Q1 → 2016-Q4 | 2012-Q1 → 2017-Q4 |
| Test window | 2017-Q1 → 2025-Q4 (36q) | 2018-Q1 → 2025-Q4 (32q) |

Mixed-frequency models (MIDAS, MIDAS-ML) start later because they require the weekly
indicators: 1967-Q1 for the US and 2002-Q1 for Türkiye. Three COVID-quarter dummies
(2020-Q2/Q3/Q4) are added to every model as a simple adjustment for the pandemic
observations.

**Stationarity.** Transformations follow the McCracken–Ng seven-code (`tcode`)
system. US codes come from the FRED release files (only `NWPIx` and `HWIx` are
overridden to log first differences). Türkiye codes are assigned manually with a
structural-break-aware procedure: a Zivot–Andrews test first, then ADF on a sequence
of transforms — yielding 7 series in levels, 24 in first differences, and 24 in log
first differences. Transformed series are re-checked with a battery of tests (ADF,
KPSS, PP for the US; plus Zivot–Andrews for Türkiye).

**Feature selection and predictor categories.** A four-method ensemble
(cross-validated Lasso, cross-validated ElasticNet, tuned random forest with
permutation importance, and Lasso stability selection) is run **once on each
country's training window** and frozen before estimation. All cross-validation
respects the time ordering, and standardization happens inside each CV fold. Results
map to three information sets:

- **Category 1** — target only (the ARMA benchmark).
- **Category 2** — small set for the unpenalized models (OLS, VAR): US `{outbs,
  outnfb, gcec1, houstne}`, Türkiye `{ipi_sa, usd_try_avg, cpi_sa, fin_acc}` (4
  predictors + 3 COVID dummies = 7 features).
- **Category 3** — full panel for the regularized/ensemble/shrinkage models: US 53
  predictors (56 features), Türkiye 22 predictors (25 features). BVAR and DFM use
  further-reduced or validation-selected sets for computational feasibility.

## Model Set

Seventeen models spanning one univariate benchmark and six methodological families:

| Family | Models | Predictor set |
|---|---|---|
| Benchmark (univariate) | `arma` | Cat 1 (target only) |
| Classical econometric | `ols`, `var` | Cat 2 |
| Penalized linear | `ridge`, `lasso`, `elasticnet` | Cat 3 |
| Tree-based ML | `dt`, `rf`, `gb`, `xgboost` | Cat 3 |
| Neural networks | `mlp`, `lstm`, `deepvar` | Cat 3 |
| Mixed-frequency | `midas`, `midasml` | Cat 3 + weekly |
| Bayesian / factor | `bvar`, `dfm` | reduced / validation-selected |

## Forecast Vintages

For each test quarter `Q` with closing month `T`, every model produces nowcasts at
standardized information dates. All vintages target the *same* realization of `Q` and
differ only in the information available when the forecast is made.

| Vintage | Information date | Interpretation |
|---|---|---|
| `m1` | T−2 | Earliest; closest to a one-quarter-ahead forecast |
| `m2` | T−1 | Mid-quarter |
| `m3` | T | The nowcast with the most within-quarter information |
| `post1` | T+1 | Robustness: post-quarter data, target still unpublished |
| `post2` | T+2 | Robustness, **Türkiye only** |

`m1`/`m2`/`m3` are the symmetric cross-country benchmark. `post1`/`post2` are
post-release robustness horizons; they are operationally meaningful for Türkiye,
whose official GDP is released ~2 months after quarter-end, and are reported
separately. Because the GDP target carries a two-month publication-lag mask, the ARMA
nowcasts at `m1` and `m2` coincide by construction (multivariate models still differ
across all vintages as their indicator sets update monthly).

## Methodology

All 17 models share a common pseudo-real-time design, with model-specific steps only
where an estimator requires them.

- **Expanding window.** Most models are re-estimated at every forecast date on the
  data available at that exact milestone; the estimation window grows as the exercise
  moves forward. The exception is the DFM, which is estimated once on the training
  window and run forward with the Kalman filter.
- **Time-ordered tuning.** Hyperparameters are chosen with time-ordered
  cross-validation on the training-plus-validation window and then frozen for the
  rolling test loop. Standard K-fold CV is deliberately avoided (it would mix future
  and past observations within a fold).
- **Ragged-edge masking.** At each forecast date, each variable's publication lag is
  applied so the model only sees data that would actually have been released; the GDP
  target row for the current quarter is kept in the panel but its value is hidden.
- **Preprocessing of flattened supervised models.** Lag-exclude → mean-fill missing
  values (training slice only) → scale where needed (standard scaling for penalized
  regressions, robust scaling for the MLP; COVID dummies left unscaled) → fit →
  write the nowcast. Monthly predictors are expanded to current value + three lags,
  then reduced to quarter-end rows.
- **State-space / sequence models bypass the flattened pipeline.** DFM (Kalman
  filter), BVAR (mixed-frequency state space, `mfbvar`), MIDAS/MIDAS-ML (predictors
  at native frequency), and LSTM/DeepVAR (internal sequence construction and
  scaling).
- **Seeded ensembling.** Tree ensembles, LSTM, and DeepVAR are averaged across ten
  seeded runs; the MLP uses the median of ten runs; the single decision tree is run
  once.
- **Leakage audit.** Feature selection, imputation, scaling, and tuning use only
  pre-test data. A post-pipeline audit confirmed saved files contain no missing or
  non-finite values, targets match across models, and no tuning relies on test-period
  outcomes.

## Evaluation Protocol

To prevent a single pooled summary from masking regime-dependent performance, results
are reported over four country-specific panels:

| Panel | United States | Türkiye |
|---|---|---|
| Pre-COVID / pre-crisis | 2017-Q1–2019-Q4 (12q) | 2018-Q1–2019-Q4 (8q) |
| COVID | 2020-Q2–2021-Q4 (7q) | 2020-Q2–2021-Q4 (7q) |
| Post-COVID | 2022-Q1–2025-Q4 (16q) | 2022-Q1–2025-Q4 (16q) |
| Full | 2017-Q1–2025-Q4 (36q) | 2018-Q1–2025-Q4 (32q) |

(For Türkiye the early panel is labelled *pre-crisis* rather than *pre-COVID* because
it already contains the 2018 currency crisis.) Each model/country/vintage/panel is
scored with **RMSFE** and **MAE**, plus the **relative RMSFE vs. the ARMA benchmark**
(values below 1 beat ARMA). Statistical comparisons use the **Diebold–Mariano** test
on the squared-error loss differential; because the panels are short (7–36 quarters),
DM results are reported as *directional* evidence and interpreted cautiously.

## Headline Results

### Full panel, `m3` vintage

**United States** (relative RMSFE vs. ARMA):

| Rank | Model | RMSFE | Rel. RMSFE |
|---:|---|---:|---:|
| 1 | LSTM | 0.0116 | 0.502 |
| 2 | MLP | 0.0137 | 0.592 |
| 3 | BVAR | 0.0146 | 0.633 |
| 4 | Ridge | 0.0152 | 0.660 |
| 5 | MIDAS | 0.0182 | 0.791 |
| … | ARMA | 0.0231 | 1.000 |
| 17 | DFM | 0.0366 | 1.588 |

**Türkiye** (relative RMSFE vs. ARMA):

| Rank | Model | RMSFE | Rel. RMSFE |
|---:|---|---:|---:|
| 1 | Lasso | 0.0190 | 0.503 |
| 2 | ElasticNet | 0.0194 | 0.515 |
| 3 | BVAR | 0.0205 | 0.544 |
| 4 | OLS | 0.0259 | 0.688 |
| 5 | VAR | 0.0265 | 0.703 |
| 14 | ARMA | 0.0377 | 1.000 |
| 16 | LSTM | 0.0428 | 1.137 |
| 17 | MIDAS-ML | 0.0530 | 1.407 |

At the family level the **neural** family leads in the US (avg. rel. RMSFE 0.646)
while the **penalized-linear** family leads in Türkiye (0.574). The
reduced-dimensional **BVAR is the only model in the top three for both countries**.
Turkish nowcast errors are roughly 1.4–1.9× larger than US errors rank-by-rank,
consistent with Turkish GDP growth being ~1.8× as volatile (σ ≈ 0.035 vs. 0.019).

### Performance is concentrated in crisis periods

Full-sample rankings are heavily crisis-weighted. The leaders earn most of their edge
during the COVID window; in calm quarters the gaps collapse:

| Model | Pre-COVID/crisis | COVID | Post-COVID | Full |
|---|---:|---:|---:|---:|
| US — LSTM | 1.250 | 0.460 | 1.024 | 0.502 |
| US — BVAR | 1.972 | 0.510 | 2.413 | 0.633 |
| TR — Lasso | 0.527 | 0.390 | 1.447 | 0.503 |
| TR — BVAR | 0.636 | 0.333 | 1.899 | 0.544 |

In **post-COVID Türkiye, the univariate ARMA is the most accurate model overall.**

### Within-quarter information gains (`m1` → `m3`)

Accuracy improves substantially as within-quarter data accumulates, and the biggest
gains go to each country's top models:

- **United States:** LSTM −40.7%, MLP −30.1%, Ridge −25.4%, BVAR −23.5%.
- **Türkiye:** BVAR −47.3%, Lasso −46.7%, ElasticNet −45.7%.

### Post-release horizons

US `post1` mirrors the `m3` ranking (LSTM 0.437, MLP 0.561). For Türkiye the
post-release gains are large: **BVAR** tops `post1` (0.459) and the validation-selected
**DFM** surges to second at `post2` (0.524, behind Lasso 0.517) — matching the
dynamic-factor tradition in the Turkish nowcasting literature.

### Forecast combinations (US robustness extension)

A constructed-after-the-fact robustness layer adds seven equal-weight combinations to
the 17 base models (24 specifications). An equal-weight **BVAR + MIDAS + DFM**
combination attains a full-sample `m3` RMSFE of **0.0074 (rel. 0.319)**, outperforming
every individual model through error cancellation in the pandemic quarters (its edge
disappears once the COVID window is excluded).

## Discussion: Why the Rankings Differ

The same 17-model line-up produces nearly opposite rankings in the two countries,
driven by three differences in their data environments:

1. **Sample length.** US models train on ~200 quarterly observations; Türkiye's
   effective informative sample is on the order of 50 quarters. Flexible,
   high-capacity methods (LSTM, MLP) exploit the long US history without overfitting,
   but struggle on the short Turkish panel.
2. **Data richness.** The US Category-3 panel offers 53 indicators (from a
   296-variable pool) plus weekly financial series; Türkiye has only 22 usable
   indicators, compressing the advantage of high-dimensional methods.
3. **Volatility and structural instability.** Turkish GDP growth is ~1.8× as volatile,
   and the sample contains the 2018 currency crisis, COVID, and the post-2021 inflation
   surge. In such a noisy environment estimation variance is costly and shrinkage pays.

**Model-specific lessons.**

- **BVAR is the most portable model** — the only one in the top three for both
  countries — but its skill is concentrated in crisis regimes (a "portable crisis
  hedge" rather than a uniformly dominant benchmark).
- **The neural models are the sharpest expression of the reversal**: best in the US,
  but below the ARMA benchmark in Türkiye. High parameter capacity is an asset only
  when the data environment can support it.
- **The DFM's weakness is an implementation result, not a verdict on factor models.**
  The US DFM ranks last purely because of 2020 (fixed pre-test parameters, no COVID
  dummies); the validation-selected Turkish DFM is competitive within the quarter and
  rises to second at `post2`.
- **MIDAS-ML's last-place Turkish result is a constrained build** (fixed-penalty
  fallback after CV failed); its cross-validated US counterpart ranks sixth.

**Implications for practice.** (i) Model choice must be conditioned on the local data
environment — importing a US-validated architecture into an emerging market is
unsupported. (ii) Within-quarter information *timing* matters as much as model choice
(gaps at `m1` are small in both countries). (iii) For Türkiye specifically, long
publication lags make post-release horizons operationally vital, where Lasso, the
validation-selected DFM, and BVAR perform best.

## Figures

Generated paper figures live in `figures/` (see
[`figures/FIGURE_INDEX.md`](figures/FIGURE_INDEX.md) and
[`figures/RESULTS_FIGURE_INDEX.md`](figures/RESULTS_FIGURE_INDEX.md) for the full
inventory). Key figures:

- `full_m3_relative_rmsfe_us_tr.png` — relative RMSFE vs. ARMA, both countries (the core contrast).
- `full_m3_rmsfe_rankings.png` — absolute RMSFE rankings by country.
- `model_family_relative_rmsfe.png` — family-level averages.
- `results_covid_sensitivity_m3.png` — COVID stress test.
- `vintage_rmsfe_profiles.png` — `m1`/`m2`/`m3` accuracy profiles.
- `results_post_release_rankings.png` — post-release horizons.
- `panel_relative_rmsfe_heatmaps.png` — robustness across panels.
- `results_us_combination_robustness.png` — US forecast combinations.

## Reproduce the Final Evaluation

**Requirements.** Python (see `requirements.txt`: pandas, numpy, scikit-learn, scipy,
statsmodels, matplotlib, openpyxl). Several model notebooks additionally need R
packages (`mfbvar` for BVAR, `nowcastDFM` for the DFM, `midasr` and `midasml` for the
mixed-frequency models) and the `nowcast_lstm` Python library (LSTM); DeepVAR uses a
PyTorch implementation. These are documented inside the individual notebooks.

The four scripts below score the saved per-model nowcasts and regenerate the figures.
Run from the repository root:

```bash
pip install -r requirements.txt

python data/evaluate.py                 # builds the evaluation result CSVs
python data/us_improvement.py           # US forecast-combination robustness layer
python data/generate_figures.py         # main paper figures
python data/generate_results_visuals.py # results-analysis & robustness figures
```

Expected outputs:

| Output | Expected size |
|---|---|
| `data/evaluation_results_us.csv` | 272 rows = 17 models × 4 vintages × 4 panels |
| `turkey_data/evaluation_results_tr.csv` | 340 rows = 17 models × 5 vintages × 4 panels |
| `data/evaluation_results_us_improved.csv` | 576 rows = 24 models/combinations × 4 vintages × 6 panels |
| `figures/*.png` | 15 figures (+ 2 index files) |

The evaluation reads the per-model nowcasts already saved in `predictions/` (US) and
`turkey_predictions/` (Türkiye). To regenerate a model's nowcasts from scratch, run
the corresponding notebook in `model_notebooks/` or `turkey_model_notebooks/`.

## Repository Layout

```text
.
├── paper .pdf                          # Full paper (PDF)
├── README.md                           # This file
├── requirements.txt                    # Core Python dependencies
│
├── data/                               # United States: data pipeline, evaluation, results
│   ├── build_raw_data.py               #   assemble raw monthly panel (FRED-MD/QD + manual)
│   ├── build_weekly_data.py            #   weekly series for mixed-frequency models
│   ├── build_final_tf_data.py          #   apply stationarity transforms → data_tf_monthly.csv
│   ├── build_metadata.py               #   publication lags, tcodes, variable metadata
│   ├── run_stationarity.py             #   ADF / KPSS / PP stationarity battery
│   ├── feature_selection_ensemble.py   #   four-method ensemble feature selection
│   ├── rank_variables.py               #   variable importance ranking report
│   ├── rank_by_rules.py                #   rule-based ranking report
│   ├── evaluate.py                     #   score predictions → evaluation_results_us.csv
│   ├── us_improvement.py               #   forecast combinations + extended panels
│   ├── generate_figures.py             #   main paper figures
│   ├── generate_results_visuals.py     #   results-analysis & robustness figures
│   ├── helpers.py, helpers.R           #   shared utilities
│   ├── *.md                            #   evaluation_protocol, target_specification,
│   │                                   #   scaling_policy, tuning_policy, variable_selection_framework
│   ├── fred-md.xlsx, fred-qd.xlsx, …   #   data inputs (+ data_tf_monthly.csv, us_master_monthly.xlsx)
│   ├── evaluation_results_us.csv       #   US results (all models × vintages × panels)
│   └── evaluation_results_us_improved.csv  # US combinations + extended panels
│
├── turkey_data/                        # Türkiye: data pipeline + results (mirrors data/)
│   ├── build_raw_data_tr.py            #   assemble raw monthly panel (CBRT/EVDS, TÜİK, BIST)
│   ├── build_final_tf_data_tr.py       #   apply stationarity transforms
│   ├── build_metadata_tr.py            #   publication lags + metadata
│   ├── determine_tcodes_tr.py          #   Zivot–Andrews / ADF transform-code selection
│   ├── run_stationarity_tr.py          #   stationarity battery
│   ├── feature_selection_tr.py         #   feature selection
│   ├── turkey_helpers.py               #   shared utilities
│   ├── tr_monthly_series.xlsx, …       #   data inputs (+ data_tf_monthly_tr.csv)
│   └── evaluation_results_tr.csv       #   Türkiye results
│
├── model_notebooks/                    # 17 US model notebooks  (model_<name>.ipynb)
├── turkey_model_notebooks/             # 17 Türkiye notebooks (+ R scripts for BVAR/MIDAS)
│
├── predictions/                        # US nowcasts: <model>_<vintage>.csv  (17 models + 7 combos)
├── turkey_predictions/                 # Türkiye nowcasts: <model>_<vintage>.csv  (17 models)
│
├── figures/                            # generated figures + FIGURE_INDEX.md / RESULTS_FIGURE_INDEX.md
└── docs/
    └── bvar_us_fallback_log.csv        # per-vintage log of the US BVAR Category-2 fallback
```

Filename conventions: prediction files are `<model>_<vintage>.csv` (US) and
`<model>_<vintage>.csv` (Türkiye, with `_tr` on the multivariate model names); the US
`predictions/` folder also contains `combo_*` files for the seven forecast
combinations.

## Limitations and Future Research

- **Pseudo-real-time, not historical-vintage.** Publication lags and ragged edges are
  simulated, but the underlying series are final revised data, so revision uncertainty
  is not measured. A true historical-vintage evaluation (e.g., via ALFRED) would
  measure the revision channel this study sets aside.
- **Coarse vintages.** Three within-quarter information sets are evaluated, not the
  full release-by-release data flow, and no news decomposition is performed.
  Release-level news decomposition in the DFM/BVAR framework would convert the
  vintage-level results into indicator-level attribution.
- **Short panels.** Diebold–Mariano comparisons are directional rather than decisive.
- **Alternative high-frequency data** (card transactions, payments) could relax the
  early-vintage information constraint that most compresses model performance at `m1`,
  especially for Türkiye.

## Important Caveats

Reproduced from the paper's implementation-caveats table; where a constrained
implementation underperforms, the result is attributed to the implementation rather
than the model class:

- **BVAR (both countries):** reduced-dimensional — US uses a Lasso-80 predictor set
  with documented Category-2 fallbacks at the 2025-Q4 `m3` and `post1` vintages
  (singular covariance matrix; logged in `docs/bvar_us_fallback_log.csv`); Türkiye uses
  locked Category-2 predictors plus the target.
- **US DFM:** estimated once on the pre-test sample and filtered forward; quarterly
  indicators and COVID dummies excluded, so its pandemic-era errors are
  implementation-specific (it ranks last at `m3` solely because of 2020).
- **Türkiye DFM:** validation-selected Category-2 monthly panel plus target (chosen on
  2012–2017 validation RMSFE, retrained through 2017); short-history Tier-C variables
  excluded after estimation failed on the sparse panel.
- **MIDAS:** single-indicator nowcasts combined with in-sample-RMSE weights, with a
  weighting rule that differs across countries (an implementation artifact).
- **Türkiye MIDAS-ML:** fixed-penalty `sglfit(lambda = 0.01)` after rolling
  cross-validation aborted; a constrained build (its cross-validated US counterpart
  ranks 6th).
- **Türkiye MLP:** finite but unstable `m1`/`m2` outputs — read as model instability,
  not a headline result.
- **DeepVAR (both countries):** nowcasts are nearly identical across `m1`/`m2`/`m3`
  because tail-information changes are smoothed by imputation.
- **Diebold–Mariano:** simple loss-differential variance and small panels — interpret
  significance cautiously.
