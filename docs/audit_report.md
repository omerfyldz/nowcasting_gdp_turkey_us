# Project Audit Report

Generated: 2026-05-13

## Bottom Line

The empirical package is ready to support paper writing, with explicit caveats. The project now has:

- 17 model families for the United States and Turkey
- 51 prediction files per country
- m1/m2/m3 vintages for every model
- country-aware evaluation outputs
- a root-level summary table
- repaired Turkey ARMA, Turkey MIDAS, Turkey DFM, Turkey BVAR, and R-side vintage handling

The project should be described as a **pseudo-real-time nowcasting benchmark using final revised targets and simulated publication lags**, not as a fully real-time vintage-data study.

## Literature Alignment

The project is broadly aligned with the nowcasting literature:

- Giannone, Reichlin, and Small (2008) motivate nowcasting with large datasets, staggered releases, and jagged-edge information sets. The repaired `gen_vintage_data()` logic now explicitly represents the m1/m2/m3 information sets for R models.
- Banbura, Giannone, Modugno, and Reichlin (2013) emphasize the real-time data-flow problem. This project captures the data-flow dimension through publication-lag masking, but it does not use historical vintage databases such as ALFRED.
- Bok, Caratelli, Giannone, Sbordone, and Tambalotti (2018) support the big-data nowcasting framing and the use of factor-style information extraction.
- Hopp (2022) is the closest benchmark ancestor. The US results are qualitatively consistent with Hopp's finding that LSTM and BVAR can be highly competitive in US GDP nowcasting.
- Richardson, van Florenstein Mulder, and Vehbi (2021) support the inclusion of machine-learning models in real-time GDP nowcasting comparisons. The US results show ML strength, while Turkey shows more mixed performance, which is plausible for a shorter and more volatile emerging-market sample.
- Foroni and Marcellino (2014) support the inclusion of MIDAS/mixed-frequency approaches for ragged-edge macroeconomic nowcasting. Turkey MIDAS is now repaired and produces vintage-specific predictions.
- Cascaldi-Garcia, Luciani, and Modugno (2023) support cross-country comparison as a meaningful exercise, while also reinforcing the need to report country-specific data constraints.

## Data and Target Audit

US:

- Target: `gdpc1`
- Evaluation window: 2017-Q1 to 2025-Q4
- Rows per prediction file: 36
- Date labels: 2017-03-01 through 2025-12-01

Turkey:

- Target: `ngdprsaxdctrq`
- Evaluation window: 2018-Q1 to 2025-Q4
- Rows per prediction file: 32
- Date labels: 2018-03-01 through 2025-12-01

Validation result:

- All required prediction files exist.
- All prediction files have required columns: `date`, `actual`, `prediction`.
- No missing or non-finite predictions remain.
- Actual values are consistent across files up to floating-point precision.

## Vintage Construction Audit

The most important repair was making R-side vintage construction explicit.

Before repair, several R notebooks used:

```r
gen_lagged_data(metadata, test, target_date, lag = -2/-1/0)
```

That could approximate m1/m2/m3, but it was ambiguous and caused real issues for Turkey MIDAS because target dates were quarter-start labels.

Current R convention:

```r
gen_vintage_data(metadata, data, target_date, vintage_date)
```

where:

- m1 = target quarter end minus 2 months
- m2 = target quarter end minus 1 month
- m3 = target quarter end

This keeps the target-quarter row available for MIDAS/DFM prediction, while masking all variables according to the information available at the vintage date.

## Repaired Issues

### Turkey ARMA

Previous issue:

- Fit once on 1995-2011 and reused the same one-step forecast for all test quarters.
- Produced one constant prediction per vintage.

Repair:

- ARMA now refits on the GDP history available at each vintage.
- m1/m2 remain equal, which is plausible under the two-month GDP publication lag.
- m3 differs from m1/m2.

### Turkey MIDAS

Previous issue:

- Used quarter-start labels and produced identical m1/m2/m3 predictions.

Repair:

- Uses quarter-end target labels.
- Uses explicit vintage dates.
- Weekly variables are truncated to the vintage date.
- m1/m2/m3 predictions now differ.

### Turkey MIDAS-ML

Previous issue:

- Fixed-penalty output existed but vintage construction was ambiguous.

Repair:

- Uses quarter-end target labels.
- Uses explicit vintage dates.
- Weekly data after the vintage date are masked, while the weekly calendar is preserved so `midasml` can build a valid design matrix.

Remaining caveat:

- Uses fixed `sglfit(lambda = 0.01)`, not rolling `cv.sglfit`.
- Some rows still have equal vintage predictions, but the full output is no longer broken or all-identical.

### Turkey DFM

Previous issue:

- Code used only 19 Cat3 variables despite documentation implying a broader DFM set.

Repair:

- Uses the complete 22-variable Cat3 monthly set plus target.
- Tier-C variables were tested but excluded because the full sparse panel fails inside `nowcastDFM`.

Paper caveat:

- Turkey DFM must be described as Cat3 + target, not a 54-variable Tier-C DFM.

### BVAR

US:

- Uses Lasso-80 reduced predictor set.
- Full Cat3 BVAR is computationally infeasible in `mfbvar`.
- The 2025-Q4 m3 reduced-BVAR fit hits a singular covariance matrix; the saved prediction uses the documented Cat2 fallback forecast.

Turkey:

- Uses locked Cat2 predictors plus target.
- Repaired to use explicit vintage-date masking.

Paper caveat:

- Both BVARs are reduced-dimensional BVARs.

## Evaluation Audit

Evaluator:

- `data/evaluate.py`

Outputs:

- `data/evaluation_results_us.csv`
- `turkey_data/evaluation_results_tr.csv`
- `evaluation_summary.md`

Final evaluator result:

- US: 204 rows = 17 models x 3 vintages x 4 panels
- Turkey: 204 rows = 17 models x 3 vintages x 4 panels
- Relative RMSFE vs ARMA populated for all rows
- DM statistics and p-values populated for all non-ARMA rows

## Figure Artifact Audit

The original repository includes README-level scatter plots and appendix nowcast plots. This project now has a reproducible figure-generation layer:

- `data/generate_figures.py`
- `figures/FIGURE_INDEX.md`
- `images/data_example.png`
- `images/fig1.png`
- `images/fig2.png`
- `images/app1.png` through `images/app17.png`
- `figures/full_m3_rmsfe_rankings.png`
- `figures/full_m3_relative_rmsfe_us_tr.png`
- `figures/panel_relative_rmsfe_heatmaps.png`
- `figures/vintage_rmsfe_profiles.png`
- `figures/model_family_relative_rmsfe.png`

Validation result:

- 25 PNG figure files were regenerated from the final prediction and evaluation CSVs.
- All generated PNGs were opened successfully with PIL and have nonzero dimensions.
- The original-style scatter plots use **nowcast revision volatility across m1/m2/m3**, not historical GDP data revisions. This distinction should be stated if the figures are used in the paper.

## Headline Results

US full panel, m3:

1. LSTM
2. BVAR
3. MIDAS
4. MLP
5. MIDAS-ML

Turkey full panel, m3:

1. MIDAS
2. VAR
3. LSTM
4. BVAR
5. Lasso

Interpretation:

- US results are consistent with Hopp-style evidence that LSTM and BVAR can be very competitive in a data-rich setting.
- Turkey results are plausible for a shorter, more volatile emerging-market sample: parsimonious econometric and mixed-frequency models are more competitive.
- MLP m1/m2 instability should be treated as an empirical finding/caveat, not hidden.

## Paper-Readiness Caveats

These caveats should be stated explicitly in the methodology or limitations section:

1. The exercise uses final revised GDP values as targets, not real-time historical GDP vintages.
2. The information sets are simulated using publication lags and ragged-edge masking.
3. US BVAR and Turkey BVAR are reduced-dimensional models.
4. Turkey DFM excludes Tier-C variables after package failure on the full sparse panel.
5. Turkey MIDAS-ML uses a fixed sparse-group-lasso penalty.
6. Diebold-Mariano results are computed with a simple iid loss-differential variance; small-panel p-values should be interpreted cautiously.
7. MLP m1/m2 results are unstable and should not be overinterpreted.

## Archived Non-Core Files

These files/directories were checked for live references and moved into `archive/` because they are generated scratch artifacts, old diagnostics, stale logs, handoff notes, or historical reference material. They were not deleted.

Previously removed diagnostic/patch scripts are documented here for audit continuity:

- `diagnose_dfm.R`
- `diagnose_dfm2.R`
- `inspect_dfm_cell.R`
- `inspect_dfm_cell3.py`
- `inspect_dfm_cells2.py`
- `inspect_dfm_full.R`
- `patch_dfm_cell3.py`
- `patch_dfm_chardate.py`
- `patch_dfm_final.py`
- `patch_dfm_train.py`
- `patch_lstm.py`
- `patch_midasml_gindex.py`
- `patch_midasml_gindex2.py`
- `patch_r_notebooks.py`
- `patch_r_notebooks2.py`
- `test_dfm_iq.R`
- `test_sapply_empty.R`
- `midas_cell0.txt`
- `midas_dump.txt`
- `midasml_all.txt`

Generated execution scripts previously removed during cleanup:

- `model_notebooks/model_midas_exec.r`
- `model_notebooks/model_midasml_exec.r`
- `turkey_model_notebooks/model_dfm_tr_exec.r`
- `turkey_model_notebooks/model_midasml_tr_exec.txt`
- `turkey_model_notebooks/model_midasml_tr_try.txt`
- `turkey_model_notebooks/model_midasml_tr_try2.txt`

Historical duplicate notebook tree previously removed during cleanup:

- `turkey_data/model_notebooks/`

Old logs previously removed during cleanup:

- `model_notebooks/dfm_run.log`
- `turkey_data/model_notebooks/logs/`
- `nb_logs/`

Python caches previously removed during cleanup:

- `data/__pycache__/`
- `turkey_data/__pycache__/`

Current archive locations:

- `archive/assistant_context/`
- `archive/handoff_notes/`
- `archive/diagnostics/`
- `archive/logs/`
- `archive/original_reference/`

## Final Assessment

The empirical pipeline is now internally coherent enough to be turned into paper tables and a methodology section. The strongest paper framing is:

> A pseudo-real-time comparison of GDP nowcasting models across a data-rich advanced economy and a shorter-history, more volatile emerging-market economy, using harmonized model families, publication-lag masking, and country-specific predictor panels.

The project should avoid claiming full real-time vintage evaluation.

## References Used for Cross-Check

- Giannone, D., Reichlin, L., & Small, D. (2008). "Nowcasting: The real-time informational content of macroeconomic data." Journal of Monetary Economics. https://doi.org/10.1016/j.jmoneco.2008.05.010
- Banbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). "Now-Casting and the Real-Time Data Flow." Handbook of Economic Forecasting. https://doi.org/10.1016/B978-0-444-53683-9.00004-9
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., & Tambalotti, A. (2018). "Macroeconomic Nowcasting and Forecasting with Big Data." Annual Review of Economics. https://doi.org/10.1146/annurev-economics-080217-053214
- Hopp, D. (2022). "Benchmarking econometric and machine learning methodologies in nowcasting." UNCTAD Research Paper No. 83. https://unctad.org/publication/benchmarking-econometric-and-machine-learning-methodologies-nowcasting
- Richardson, A., van Florenstein Mulder, T., & Vehbi, T. (2021). "Nowcasting GDP using machine-learning algorithms: A real-time assessment." International Journal of Forecasting. https://doi.org/10.1016/j.ijforecast.2020.10.005
- Foroni, C., & Marcellino, M. (2014). "A comparison of mixed frequency approaches for nowcasting Euro area macroeconomic aggregates." International Journal of Forecasting. https://doi.org/10.1016/j.ijforecast.2013.01.010
- Cascaldi-Garcia, D., Luciani, M., & Modugno, M. (2023). "Lessons from Nowcasting GDP across the World." Federal Reserve IFDP No. 1385. https://doi.org/10.17016/IFDP.2023.1385
