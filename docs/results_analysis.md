# Results Analysis

Generated from `data/evaluation_results_us.csv`, `turkey_data/evaluation_results_tr.csv`, and `data/evaluation_results_us_improved.csv`.

## Visual Guide

- `figures/results_period_rankings_m3.png`: top models by period and country.
- `figures/results_vintage_gain_m1_to_m3.png`: information gain from early to late within-quarter vintages.
- `figures/results_post_release_rankings.png`: US `post1` and Turkey `post1`/`post2` robustness rankings.
- `figures/results_covid_sensitivity_m3.png`: COVID-period stress-test rankings.
- `figures/results_top3_period_robustness.png`: models that repeatedly appear in the top three across periods.
- `figures/results_dfm_validation_selection.png`: Turkey DFM validation-selection evidence.
- `figures/results_us_combination_robustness.png`: US forecast-combination robustness.

## Reading Guide

- `m1`, `m2`, and `m3` are the symmetric nowcast vintages used for US-Turkey comparison.
- US `post1` and Turkey `post1`/`post2` are robustness horizons and should be discussed separately.
- Lower RMSFE is better. Relative RMSFE below 1 means the model beats ARMA in the same country, panel, and vintage.
- Period panels are country-specific: US has pre-COVID, COVID, post-COVID; Turkey has pre-crisis, COVID, post-COVID.

## Main Full-Sample Rankings

### United States, Full Sample, m3
| Rank | Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- | --- |
| 1 | lstm | 0.0116 | 0.0062 | 0.5021 |
| 2 | mlp | 0.0137 | 0.0065 | 0.5921 |
| 3 | bvar | 0.0146 | 0.0102 | 0.6333 |
| 4 | ridge | 0.0152 | 0.0070 | 0.6595 |
| 5 | midas | 0.0182 | 0.0076 | 0.7914 |
| 6 | midasml | 0.0186 | 0.0076 | 0.8069 |
| 7 | elasticnet | 0.0189 | 0.0078 | 0.8185 |
| 8 | lasso | 0.0189 | 0.0078 | 0.8185 |
| 9 | gb | 0.0192 | 0.0078 | 0.8312 |
| 10 | dt | 0.0193 | 0.0080 | 0.8357 |

### Turkey, Full Sample, m3
| Rank | Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- | --- |
| 1 | lasso | 0.0190 | 0.0123 | 0.5035 |
| 2 | elasticnet | 0.0194 | 0.0125 | 0.5146 |
| 3 | bvar | 0.0205 | 0.0158 | 0.5442 |
| 4 | ols | 0.0259 | 0.0168 | 0.6881 |
| 5 | var | 0.0265 | 0.0176 | 0.7027 |
| 6 | midas | 0.0265 | 0.0151 | 0.7046 |
| 7 | ridge | 0.0266 | 0.0160 | 0.7049 |
| 8 | dfm | 0.0276 | 0.0180 | 0.7317 |
| 9 | mlp | 0.0297 | 0.0176 | 0.7890 |
| 10 | gb | 0.0316 | 0.0183 | 0.8396 |

**Interpretation.** The US ranking favors neural/high-dimensional methods, especially LSTM and MLP, with BVAR close behind. Turkey favors penalized linear shrinkage and reduced-dimensional econometric models: Lasso, ElasticNet, and BVAR lead at `m3`, while validation-selected DFM becomes competitive after the DFM repair pass.

## Period-Specific Rankings

### United States, m3
| Period | Top 5 models, m3 |
| --- | --- |
| Pre-COVID | gb (0.0026), ols (0.0027), var (0.0027), deepvar (0.0027), midas (0.0027) |
| COVID | lstm (0.0235), bvar (0.0260), mlp (0.0288), ridge (0.0326), midas (0.0399) |
| Post-COVID | deepvar (0.0043), xgboost (0.0044), midasml (0.0045), midas (0.0045), ols (0.0045) |
| Full | lstm (0.0116), mlp (0.0137), bvar (0.0146), ridge (0.0152), midas (0.0182) |

### Turkey, m3
| Period | Top 5 models, m3 |
| --- | --- |
| Pre-crisis | mlp (0.0085), elasticnet (0.0094), dfm (0.0095), ridge (0.0098), lasso (0.0098) |
| COVID | bvar (0.0253), lasso (0.0296), elasticnet (0.0311), ols (0.0430), var (0.0431) |
| Post-COVID | arma (0.0116), midas (0.0117), deepvar (0.0119), lstm (0.0120), rf (0.0121) |
| Full | lasso (0.0190), elasticnet (0.0194), bvar (0.0205), ols (0.0259), var (0.0265) |

**Interpretation.** The COVID panel is the stress test. Models that rank well in normal periods do not automatically dominate during COVID. In Turkey, BVAR and penalized linear models lead the COVID `m3` ranking, while DFM becomes much more competitive after validation selection but does not dominate the COVID panel.

## Family-Level Patterns, Full Sample m3

### United States
| Family | Average RMSFE | Best RMSFE | Average rel. ARMA | Best model |
| --- | --- | --- | --- | --- |
| Neural | 0.0149 | 0.0116 | 0.6463 | lstm |
| Penalized linear | 0.0176 | 0.0152 | 0.7655 | ridge |
| Mixed-frequency | 0.0184 | 0.0182 | 0.7992 | midas |
| Tree ML | 0.0193 | 0.0192 | 0.8384 | gb |
| Classical | 0.0196 | 0.0194 | 0.8515 | ols |
| Benchmark | 0.0231 | 0.0231 | 1.0000 | arma |
| Bayesian/factor | 0.0256 | 0.0146 | 1.1109 | bvar |

### Turkey
| Family | Average RMSFE | Best RMSFE | Average rel. ARMA | Best model |
| --- | --- | --- | --- | --- |
| Penalized linear | 0.0216 | 0.0190 | 0.5743 | lasso |
| Bayesian/factor | 0.0240 | 0.0205 | 0.6379 | bvar |
| Classical | 0.0262 | 0.0259 | 0.6954 | ols |
| Tree ML | 0.0338 | 0.0316 | 0.8972 | gb |
| Neural | 0.0358 | 0.0297 | 0.9498 | mlp |
| Benchmark | 0.0377 | 0.0377 | 1.0000 | arma |
| Mixed-frequency | 0.0398 | 0.0265 | 1.0558 | midas |

**Interpretation.** The US benefits more from neural models and BVAR-type high-dimensional information extraction. Turkey rewards shrinkage and parsimony: penalized linear models have the best average family performance, and the repaired DFM is useful but not dominant at `m3`.

## Vintage Gains

### United States: Largest m1 to m3 Gains
| Model | m1 RMSFE | m3 RMSFE | m1 to m3 gain % | post1 RMSFE | m3 to post1 gain % |
| --- | --- | --- | --- | --- | --- |
| lstm | 0.0195 | 0.0116 | 40.7356 | 0.0101 | 12.9030 |
| mlp | 0.0195 | 0.0137 | 30.0821 | 0.0129 | 5.3080 |
| ridge | 0.0204 | 0.0152 | 25.3674 | 0.0147 | 3.2352 |
| bvar | 0.0191 | 0.0146 | 23.5298 | 0.0154 | -5.3718 |
| elasticnet | 0.0199 | 0.0189 | 5.0395 | 0.0188 | 0.5789 |
| lasso | 0.0199 | 0.0189 | 5.0395 | 0.0188 | 0.5789 |
| midas | 0.0192 | 0.0182 | 4.8872 | 0.0181 | 0.9273 |
| midasml | 0.0195 | 0.0186 | 4.5374 | 0.0186 | -0.0086 |
| xgboost | 0.0195 | 0.0194 | 0.7589 | 0.0194 | 0.2162 |
| gb | 0.0192 | 0.0192 | 0.4336 | 0.0190 | 1.0307 |

### Turkey: Largest m1 to m3 Gains
| Model | m1 RMSFE | m3 RMSFE | m1 to m3 gain % | post2 RMSFE | m3 to post2 gain % |
| --- | --- | --- | --- | --- | --- |
| bvar | 0.0389 | 0.0205 | 47.2824 | 0.0211 | -2.8810 |
| lasso | 0.0356 | 0.0190 | 46.6562 | 0.0195 | -2.6915 |
| elasticnet | 0.0357 | 0.0194 | 45.7378 | 0.0200 | -3.3220 |
| ridge | 0.0361 | 0.0266 | 26.3459 | 0.0269 | -1.3657 |
| ols | 0.0351 | 0.0259 | 26.2154 | 0.0205 | 21.0562 |
| var | 0.0352 | 0.0265 | 24.7840 | 0.0204 | 22.9489 |
| midas | 0.0351 | 0.0265 | 24.4617 | 0.0281 | -5.8673 |
| dfm | 0.0341 | 0.0276 | 19.2723 | 0.0197 | 28.4186 |
| mlp | 0.0360 | 0.0297 | 17.4988 | 0.0297 | 0.1139 |
| xgboost | 0.0364 | 0.0316 | 13.1348 | 0.0336 | -6.1260 |

**Interpretation.** Positive gains mean the model improves as more within-quarter information becomes available. Turkey has particularly large information-timing effects for DFM, BVAR, Lasso, ElasticNet, and OLS/VAR. This supports treating vintage timing as substantive, not mechanical.

## Post-Release Robustness

### US Full Sample post1
| Rank | Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- | --- |
| 1 | lstm | 0.0101 | 0.0059 | 0.4373 |
| 2 | mlp | 0.0129 | 0.0065 | 0.5607 |
| 3 | ridge | 0.0147 | 0.0070 | 0.6382 |
| 4 | bvar | 0.0154 | 0.0104 | 0.6674 |
| 5 | midas | 0.0181 | 0.0076 | 0.7841 |
| 6 | midasml | 0.0186 | 0.0076 | 0.8070 |
| 7 | elasticnet | 0.0188 | 0.0078 | 0.8137 |
| 8 | lasso | 0.0188 | 0.0078 | 0.8137 |

### Turkey Full Sample post1
| Rank | Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- | --- |
| 1 | bvar | 0.0173 | 0.0135 | 0.4586 |
| 2 | lasso | 0.0194 | 0.0127 | 0.5151 |
| 3 | elasticnet | 0.0199 | 0.0129 | 0.5284 |
| 4 | var | 0.0212 | 0.0162 | 0.5638 |
| 5 | ols | 0.0213 | 0.0155 | 0.5660 |
| 6 | dfm | 0.0221 | 0.0149 | 0.5867 |
| 7 | ridge | 0.0262 | 0.0156 | 0.6964 |
| 8 | midas | 0.0272 | 0.0151 | 0.7228 |

### Turkey Full Sample post2
| Rank | Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- | --- |
| 1 | lasso | 0.0195 | 0.0128 | 0.5170 |
| 2 | dfm | 0.0197 | 0.0145 | 0.5237 |
| 3 | elasticnet | 0.0200 | 0.0129 | 0.5317 |
| 4 | var | 0.0204 | 0.0173 | 0.5414 |
| 5 | ols | 0.0205 | 0.0171 | 0.5432 |
| 6 | bvar | 0.0211 | 0.0171 | 0.5599 |
| 7 | ridge | 0.0269 | 0.0162 | 0.7145 |
| 8 | midas | 0.0281 | 0.0155 | 0.7459 |

**Interpretation.** Post-release horizons answer a different question: how much nowcast accuracy improves after some post-quarter indicators are available while the GDP target is still fixed. Turkey DFM becomes very competitive at `post2`, ranking second behind Lasso. BVAR is strongest at Turkey `post1`.

## Robustness Across Periods

### Top-3 Appearances Across Period Panels, US m3
| Model | Top-3 appearances | Periods |
| --- | --- | --- |
| bvar | 2 | COVID, Full |
| lstm | 2 | COVID, Full |
| mlp | 2 | COVID, Full |
| deepvar | 1 | Post-COVID |
| gb | 1 | Pre-COVID |
| midasml | 1 | Post-COVID |
| ols | 1 | Pre-COVID |
| var | 1 | Pre-COVID |
| xgboost | 1 | Post-COVID |

### Top-3 Appearances Across Period Panels, Turkey m3
| Model | Top-3 appearances | Periods |
| --- | --- | --- |
| elasticnet | 3 | Pre-crisis, COVID, Full |
| bvar | 2 | COVID, Full |
| lasso | 2 | COVID, Full |
| arma | 1 | Post-COVID |
| deepvar | 1 | Post-COVID |
| dfm | 1 | Pre-crisis |
| midas | 1 | Post-COVID |
| mlp | 1 | Pre-crisis |

**Interpretation.** Top-3 counts separate robust models from models that win only one regime. US LSTM/BVAR/MLP are robust. Turkey Lasso, ElasticNet, BVAR, and DFM are the most important recurring models across regimes.

## COVID Shock Sensitivity

### United States, m3: Lowest COVID RMSFE
| Model | Pre RMSFE | COVID RMSFE | Post RMSFE | COVID / Pre | COVID / Post |
| --- | --- | --- | --- | --- | --- |
| lstm | 0.0034 | 0.0235 | 0.0049 | 6.8948 | 4.8333 |
| bvar | 0.0054 | 0.0260 | 0.0114 | 4.8487 | 2.2750 |
| mlp | 0.0032 | 0.0288 | 0.0048 | 9.0952 | 6.0030 |
| ridge | 0.0031 | 0.0326 | 0.0049 | 10.6661 | 6.6111 |
| midas | 0.0027 | 0.0399 | 0.0045 | 14.8743 | 8.9314 |
| midasml | 0.0029 | 0.0407 | 0.0045 | 14.0062 | 9.1348 |
| elasticnet | 0.0027 | 0.0414 | 0.0045 | 15.3073 | 9.1487 |
| lasso | 0.0027 | 0.0414 | 0.0045 | 15.3073 | 9.1487 |

### Turkey, m3: Lowest COVID RMSFE
| Model | Pre RMSFE | COVID RMSFE | Post RMSFE | COVID / Pre | COVID / Post |
| --- | --- | --- | --- | --- | --- |
| bvar | 0.0118 | 0.0253 | 0.0220 | 2.1416 | 1.1515 |
| lasso | 0.0098 | 0.0296 | 0.0168 | 3.0246 | 1.7676 |
| elasticnet | 0.0094 | 0.0311 | 0.0168 | 3.3224 | 1.8517 |
| ols | 0.0167 | 0.0430 | 0.0195 | 2.5681 | 2.1992 |
| var | 0.0167 | 0.0431 | 0.0209 | 2.5848 | 2.0578 |
| dfm | 0.0095 | 0.0455 | 0.0237 | 4.7999 | 1.9174 |
| ridge | 0.0098 | 0.0481 | 0.0178 | 4.9295 | 2.7001 |
| midas | 0.0182 | 0.0501 | 0.0117 | 2.7536 | 4.2840 |

**Interpretation.** COVID performance should be discussed separately because it can dominate full-sample RMSE. The US COVID period favors LSTM/BVAR/MLP, while Turkey COVID performance favors BVAR, Lasso, and ElasticNet. Validation-selected DFM improves relative to the earlier Cat3 diagnostic, but the COVID ranking still supports the broader Turkey conclusion: shrinkage and reduced-dimensional models are safer than unrestricted high-dimensional designs.

## US Forecast Combinations

The US improvement layer evaluates forecast combinations separately from the 17-model benchmark. These are robustness checks, not replacements for the pre-specified model classes.

### US Improved Layer, Full Sample m3
| Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- |
| combo_econometric_bvar_midas_dfm | 0.0074 | 0.0055 | 0.3190 |
| lstm | 0.0116 | 0.0062 | 0.5021 |
| mlp | 0.0137 | 0.0065 | 0.5921 |
| combo_top3_lstm_bvar_midas | 0.0139 | 0.0073 | 0.6024 |
| bvar | 0.0146 | 0.0102 | 0.6333 |
| ridge | 0.0152 | 0.0070 | 0.6595 |
| combo_ml_lstm_rf_gb_xgboost_mlp | 0.0165 | 0.0072 | 0.7165 |
| combo_all_trimmed_mean | 0.0173 | 0.0074 | 0.7518 |
| combo_linear_regularized | 0.0176 | 0.0075 | 0.7640 |
| midas | 0.0182 | 0.0076 | 0.7914 |

### US Improved Layer, Full Sample Excluding 2020-Q2, m3
| Model | RMSFE | MAE | Rel. RMSFE vs ARMA |
| --- | --- | --- | --- |
| combo_econometric_bvar_midas_dfm | 0.0065 | 0.0050 | 0.3495 |
| lstm | 0.0093 | 0.0052 | 0.5012 |
| mlp | 0.0109 | 0.0052 | 0.5887 |
| combo_top3_lstm_bvar_midas | 0.0111 | 0.0061 | 0.6018 |
| midasml | 0.0115 | 0.0053 | 0.6194 |
| combo_ml_lstm_rf_gb_xgboost_mlp | 0.0116 | 0.0053 | 0.6240 |
| ridge | 0.0119 | 0.0056 | 0.6420 |
| combo_all_trimmed_mean | 0.0121 | 0.0055 | 0.6553 |
| dt | 0.0125 | 0.0057 | 0.6771 |
| midas | 0.0126 | 0.0056 | 0.6780 |

**Interpretation.** Combinations improve US robustness, especially combinations involving BVAR, MIDAS, and DFM. These should be presented as a robustness extension because they are constructed after seeing base-model behavior.

## Paper-Facing Conclusions

1. Model rankings are country-specific. US results favor neural/high-dimensional approaches; Turkey favors shrinkage, BVAR, and a validation-selected DFM.
2. Period matters. COVID rankings differ materially from pre- and post-COVID rankings, so full-sample tables should not be the only evidence.
3. Vintage timing matters. Accuracy generally improves from `m1` to `m3`, and post-release horizons provide additional information, especially for Turkey DFM and BVAR.
4. DFM is specification-sensitive. Turkey DFM was weak under the broad Cat3 diagnostic but became competitive after validation-based Cat2 selection. This should be framed as disciplined specification selection, not test-set tuning.
5. ML superiority is not universal. Neural models are strong in the US, but Turkey penalized linear models and reduced-dimensional econometric models are more reliable in the main `m3` benchmark.
6. Caveats remain central: pseudo-real-time evaluation, final revised GDP targets, reduced BVAR panels, constrained MIDAS-ML, small-panel DM tests, and Turkey MLP instability.
