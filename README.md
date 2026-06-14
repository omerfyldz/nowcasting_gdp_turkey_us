# Comparing GDP Nowcasting Models in the United States and Türkiye

Replication and empirical package for the paper **"Comparing GDP Nowcasting
Models in the United States and Türkiye"** by **Ahmet Alkan** and **Ömer Faruk
Yıldız** (Boğaziçi University, Department of Economics).

📄 **[Read the full paper (PDF)](<paper .pdf>)**

The paper conducts a pseudo-real-time comparison of **seventeen GDP nowcasting
models** across two contrasting macroeconomic environments:

- **United States** — a data-rich advanced economy (test sample 2017–2025, 36 quarters).
- **Türkiye** — a shorter-history, more volatile emerging-market economy (test sample 2018–2025, 32 quarters).

The model line-up is held **identical across both countries** while the predictor
panels are country-specific, so the exercise asks not whether a single model wins,
but whether the *ranking of model families* is stable across very different data
environments.

## Main Finding

**Model rankings are not portable across countries.**

In the United States, flexible high-dimensional methods perform best: at the most
informative within-quarter vintage the **LSTM** roughly halves the RMSFE of the
ARMA benchmark, followed by the **MLP** and a reduced-dimensional **Bayesian VAR**.
In Türkiye the same line-up produces nearly the opposite ordering: **Lasso**,
**ElasticNet**, and **BVAR** lead, parsimonious **OLS**/**VAR** stay competitive,
and the **LSTM falls behind the univariate ARMA benchmark**.

These advantages are **concentrated in turbulent periods** (the COVID window). In
calm quarters the differences between models shrink and the univariate benchmark
is hard to beat — in Türkiye's post-COVID quarters ARMA is the single most
accurate model. The results indicate that **data richness, sample length, and
macroeconomic volatility** shape the relative usefulness of econometric vs.
machine-learning nowcasting methods, cautioning against one-size-fits-all model
choices.

## Model Set

Seventeen models spanning one univariate benchmark and six methodological families:

| Family | Models |
|---|---|
| Benchmark (univariate) | `arma` |
| Classical econometric | `ols`, `var` |
| Penalized linear | `ridge`, `lasso`, `elasticnet` |
| Tree-based ML | `dt`, `rf`, `gb`, `xgboost` |
| Neural networks | `mlp`, `lstm`, `deepvar` |
| Mixed-frequency | `midas`, `midasml` |
| Bayesian / factor | `bvar`, `dfm` |

## Forecast Vintages

For each test quarter `Q` with closing month `T`, every model produces nowcasts at
standardized information dates. All vintages target the *same* realization of `Q`
and differ only in the information available when the forecast is made.

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
separately. (Because the GDP target carries a two-month publication-lag mask, the
ARMA nowcasts at `m1` and `m2` coincide by construction.)

## Data

The target in both countries is the **quarterly log change in seasonally adjusted
real GDP**, placed at the end-of-quarter month, evaluated against the latest
revised value in the **2026-03** data vintage (pseudo-real-time, not historical
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

Stationarity transformations follow the McCracken–Ng seven-code system (US codes
from the FRED release files; Türkiye codes assigned manually with a
structural-break-aware Zivot–Andrews / ADF procedure). Three COVID-quarter dummies
(2020-Q2/Q3/Q4) are added to every model.

### Predictor categories (feature selection)

A four-method ensemble (cross-validated Lasso, cross-validated ElasticNet, tuned
random forest with permutation importance, and Lasso stability selection) is run
**once on each country's training window** and frozen before estimation:

- **Category 1** — target only (the ARMA benchmark).
- **Category 2** — small information set for the unpenalized models (OLS, VAR):
  US `{outbs, outnfb, gcec1, houstne}`, Türkiye `{ipi_sa, usd_try_avg, cpi_sa, fin_acc}` (4 predictors + 3 COVID dummies).
- **Category 3** — full panel for the regularized/ensemble/shrinkage models:
  US 53 predictors (56 features), Türkiye 22 predictors (25 features).

## Evaluation Protocol

To prevent a single pooled summary from masking regime-dependent performance,
results are reported over four country-specific panels:

| Panel | United States | Türkiye |
|---|---|---|
| Pre-COVID / pre-crisis | 2017-Q1–2019-Q4 (12q) | 2018-Q1–2019-Q4 (8q) |
| COVID | 2020-Q2–2021-Q4 (7q) | 2020-Q2–2021-Q4 (7q) |
| Post-COVID | 2022-Q1–2025-Q4 (16q) | 2022-Q1–2025-Q4 (16q) |
| Full | 2017-Q1–2025-Q4 (36q) | 2018-Q1–2025-Q4 (32q) |

Each model/country/vintage/panel is scored with **RMSFE** and **MAE**, plus the
**relative RMSFE vs. the ARMA benchmark** (values below 1 beat ARMA). Statistical
comparisons use the **Diebold–Mariano** test on the squared-error loss
differential; because the panels are short (7–36 quarters), DM results are reported
as *directional* evidence and interpreted cautiously.

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

At the family level, the **neural** family leads in the US (avg. rel. RMSFE 0.646)
while the **penalized-linear** family leads in Türkiye (0.574). The
reduced-dimensional **BVAR is the only model in the top three for both countries**.
Turkish nowcast errors are roughly 1.4–1.9× larger than US errors rank-by-rank,
consistent with Turkish GDP growth being ~1.8× as volatile (σ ≈ 0.035 vs. 0.019).

### Performance is concentrated in crisis periods

Full-sample rankings are heavily crisis-weighted. The leaders earn most of their
edge during the COVID window; in calm quarters the gaps collapse:

| Model | Pre-COVID/crisis | COVID | Post-COVID | Full |
|---|---:|---:|---:|---:|
| US — LSTM | 1.250 | 0.460 | 1.024 | 0.502 |
| US — BVAR | 1.972 | 0.510 | 2.413 | 0.633 |
| TR — Lasso | 0.527 | 0.390 | 1.447 | 0.503 |
| TR — BVAR | 0.636 | 0.333 | 1.899 | 0.544 |

In **post-COVID Türkiye, the univariate ARMA is the most accurate model overall**.

### Within-quarter information gains (`m1` → `m3`)

Accuracy improves substantially as within-quarter data accumulates, and the biggest
gains go to each country's top models:

- **United States:** LSTM −40.7%, MLP −30.1%, Ridge −25.4%, BVAR −23.5%.
- **Türkiye:** BVAR −47.3%, Lasso −46.7%, ElasticNet −45.7%.

### Post-release horizons

US `post1` mirrors the `m3` ranking (LSTM 0.437, MLP 0.561). For Türkiye the
post-release gains are large: **BVAR** tops `post1` (0.459) and the
validation-selected **DFM** surges to second at `post2` (0.524, behind Lasso 0.517)
— matching the dynamic-factor tradition in the Turkish nowcasting literature.

### Forecast combinations (US robustness extension)

A constructed-after-the-fact robustness layer (24 specifications = 17 models + 7
combinations). An equal-weight **BVAR + MIDAS + DFM** combination attains a
full-sample `m3` RMSFE of **0.0074 (rel. 0.319)**, outperforming every individual
model through error cancellation in the pandemic quarters (its edge disappears once
the COVID window is excluded).

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

Requires Python (see `requirements.txt`); several model notebooks additionally
require R packages (`mfbvar`, `nowcastDFM`, `midasr`, `midasml`) and the
`nowcast_lstm` library, as documented inside the notebooks.

From the repository root:

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

The pipeline reads the per-model nowcasts already saved in `predictions/`
(US) and `turkey_predictions/` (Türkiye); rerunning a model from scratch means
executing the corresponding notebook in `model_notebooks/` or
`turkey_model_notebooks/`.

## Repository Layout

```text
.
├── paper .pdf                  Full paper (PDF)
├── README.md
├── requirements.txt            Core Python dependencies
├── data/                       US data builders, helpers, evaluation & figure scripts, policy docs, US results
├── turkey_data/                Türkiye data builders, metadata, helpers, final data, Türkiye results
├── model_notebooks/            17 US model notebooks
├── turkey_model_notebooks/     17 Türkiye model notebooks (+ R scripts for BVAR/MIDAS)
├── predictions/                US nowcast CSVs (17 models + 7 combinations, per vintage)
├── turkey_predictions/         Türkiye nowcast CSVs (17 models × 5 vintages)
├── figures/                    Paper figures + figure indexes
└── docs/                       BVAR fallback log + repository layout note
```

## Key Outputs

- `data/evaluation_results_us.csv` — US results (all models, vintages, panels).
- `turkey_data/evaluation_results_tr.csv` — Türkiye results.
- `data/evaluation_results_us_improved.csv` — US combinations + extended panels.
- `data/evaluation_protocol.md`, `data/target_specification.md` — protocol and target definitions.
- `docs/bvar_us_fallback_log.csv` — per-vintage log of the US BVAR Category-2 fallback.
- `figures/` — all paper-facing figures.

## Important Caveats

Reproduced from the paper's implementation-caveats table; where a constrained
implementation underperforms, the result is attributed to the implementation
rather than the model class:

- **Pseudo-real-time:** information sets are simulated via publication-lag and
  ragged-edge masking, but GDP targets are *final revised* values; revision
  uncertainty is not measured.
- **Coarse vintages:** three within-quarter information sets, not the full
  release-by-release data flow; no news decomposition.
- **BVAR (both countries):** reduced-dimensional — US uses a Lasso-80 predictor set
  with documented Category-2 fallbacks at the 2025-Q4 `m3` and `post1` vintages
  (singular covariance matrix; see `docs/bvar_us_fallback_log.csv`); Türkiye uses
  locked Category-2 predictors plus the target.
- **US DFM:** estimated once on the pre-test sample and filtered forward;
  quarterly indicators and COVID dummies excluded, so its pandemic-era errors are
  implementation-specific (it ranks last at `m3` solely because of 2020).
- **Türkiye DFM:** validation-selected Category-2 monthly panel plus target
  (chosen on 2012–2017 validation RMSFE, retrained through 2017); short-history
  Tier-C variables excluded after estimation failed on the sparse panel.
- **MIDAS:** single-indicator nowcasts combined with in-sample-RMSE weights, with a
  weighting rule that differs across countries (an implementation artifact).
- **Türkiye MIDAS-ML:** fixed-penalty `sglfit(lambda = 0.01)` after rolling
  cross-validation aborted; a constrained build (its cross-validated US counterpart
  ranks 6th).
- **Türkiye MLP:** finite but unstable `m1`/`m2` outputs — read as model
  instability, not a headline result.
- **DeepVAR (both countries):** nowcasts are nearly identical across `m1`/`m2`/`m3`
  because tail-information changes are smoothed by imputation.
- **Diebold–Mariano:** simple loss-differential variance and small panels —
  interpret significance cautiously.

## Citation

If you use this code or data, please cite:

> Alkan, A. and Yıldız, Ö. F. (2026). *Comparing GDP Nowcasting Models in the
> United States and Türkiye.* Boğaziçi University, Department of Economics.

Code: https://github.com/omerfyldz/nowcasting_gdp_turkey_us
