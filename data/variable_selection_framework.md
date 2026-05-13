# Variable Selection Framework — Pipeline B

## 1. Feature Selection Methods

Four methods run on 1959-2007 training window (T=196 quarterly obs, n_lags=3).
Tuned hyperparameters via TimeSeriesSplit(5):

| Method | Hyperparameter | Value |
|---|---|---|
| LassoCV | alpha | 2.477e-05 |
| ElasticNetCV | alpha, l1_ratio | 0.0001, 0.250 |
| RF (perm imp) | max_features, min_samples_leaf | 0.33, 1 |
| Lasso stability | alpha, n_resamples, subsample | 2.477e-05, 100, 75% |

All data in `feature_selection_ensemble.xlsx`.

---

## 2. Rule-Based Cumulative Cutoffs

No arbitrary top-35. Each method stops at its cumulative importance threshold.

| Threshold | Lasso | EN | RF | Stability |
|---|---|---|---|---|
| 50% | 1 | 1 | 1 | 14 |
| 70% | 4 | 4 | 2 | 21 |
| 80% | 15 | 15 | **2** | 25 |
| **90%** | **34** | **35** | **2** | **30** |
| **95%** | 49 | 50 | **4** | 32 |
| 99% | 69 | 70 | 22 | 35 |

### RF Pathological Concentration

outbs alone = 62.5% of all RF importance. top-2 = 91.2%. The 80% and 90%
cutoffs hit at just 2 variables — zero unique signal beyond what Lasso/EN
already captured. RF >=95% is the minimum usable cumulative threshold.

---

## 3. Selected Thresholds (L90, E90, R95, S90)

| Set | Count | Cutoff at |
|---|---|---|
| Lasso 90% (L90) | **34** | dhlcrg3q086sbea |
| EN 90% (E90) | **35** | dhlcrg3q086sbea |
| RF 95% (R95) | **4** | ophpbs |
| Stab 90% cum (S90) | **30** | clf16ov |

### Pairwise Overlaps

| Pair | Overlap | Interpretation |
|---|---|---|
| L90 ∩ E90 | 34 | Nearly identical — same linear class |
| L90 ∩ R95 | 2 | Linear vs non-linear — massive disagreement |
| L90 ∩ S90 | 26 | Most Stab vars are in Lasso 90% |
| R95 ∩ S90 | 2 | Only outbs, outnfb cross both worlds |
| **L90 ∩ E90 ∩ R95 ∩ S90** | **2** | outbs, outnfb |

### Union = **40 vars**

### RF 95% Unique Contribution

`gpdic1`, `ophpbs` — these 2 vars are found ONLY by RF. They represent
pure non-linear signal invisible to Lasso/EN/Stability.

---

## 4. RF Cumulative — Unique Var Analysis

How many RF vars are *unique* (not in L90, E90, or S90) at each cutoff:

| RF threshold | Total RF | Unique | Unique names |
|---|---|---|---|
| 50% | 1 | 0 | none |
| 60% | 1 | 0 | none |
| 70% | 2 | 0 | none |
| 80% | 2 | 0 | none |
| 90% | 2 | 0 | none |
| **95%** | **4** | **2** | **gpdic1, ophpbs** |
| 99% | 22 | 15 | gpdic1, ophpbs, fpix, unratestx, dmanemp, indpro, hoanbs, pcecc96, ipmat, payems, ipmansics, cfnai, hwix, pcdgx, usgood |

80% and 90% are unusable — zero unique signal. 95% is the minimum threshold
where RF contributes variables unseen by any other method.

---

## 5. Stability >= 50% Frequency (Meinshausen-Bühlmann 2010)

The established literature rule. Variables selected in >=50% of 100 Lasso
bootstrap resamples (75% subsample).

**26 vars**: a014re1q156nbea, acogno, amdmuox, andenox, busloans,
ces1021000001, ces2000000008, ces9091000001, ces9092000001, compapff,
cusr0000sas, ddurrg3m086sbea, gcec1, houstne, housts, hwiuratio,
hwiuratiox, invest, mortg10yrx, nonrevsl, outbs, outnfb, slcex,
tlbsnncbbdix, ulcbs, usserv

### Overlap with Lasso 90%

25 of 26 Stab≥50% vars are inside Lasso 90%. Only **ulcbs** is outside —
RF finds it (RF#10, imp=0.00147), Stab confirms it (51%), but Lasso barely
notices (Lasso rank #241).

---

## 6. Bok et al. (2018) Original 24 Variables — Cross-Reference

| Measure | Count |
|---|---|
| Present in our 296 | 16 / 24 |
| Missing (retired/renamed since 2018) | 8 |
| In Union (40 vars) | 4 (indpro, payems, tcu, ulcnfb) |
| In Lasso 90% (34 vars) | 1 (ulcnfb) |
| In RF 95% (4 vars) | 2 (outbs, outnfb — not original Bok vars) |
| In Stab >=50% (26 vars) | 2 (ulcnfb, unrate) |

### Bok variables displaced by close relatives

| Bok var | Displaced by | L90 rank |
|---|---|---|
| houst | houstne (Northeast), housts (South) | L#3, L#10 |
| unrate | demographic CPS variants, unratestx | RF#6 |
| permit | permitw (West) | L#23 |
| rsafs | rsafsx (FRED-MD version) | RF#25 |

The original 24 hand-picked variables are almost entirely displaced at
296-variable scale. Regional breakdowns, demographic variants, and broader
economic measures dominate data-driven selection.

---

## 7. Final Category Allocation

| Cat | Models | Rule | N vars | +COVID |
|---|---|---|---|---|
| **1** | ARMA | gdpc1 only | 1 | 1 |
| **2** | VAR, OLS | Top 2 RF (importance) + Top 3 Stab (frequency) | **4** | **7** |
| **3** | Lasso, Ridge, EN, BVAR, MIDAS, MIDASML, RF, XGB, GB, DT, MLP, LSTM, DeepVAR, DFM | L95 ∪ E95 ∪ R95 ∪ S100 (53) | **53** | **56** |

### Category 2 — VAR, OLS (7 total: 4 features + 3 COVID)

```
outbs       (RF#1, Stab#1 at 100%)
outnfb      (RF#2, Stab#6 at 86%)
gcec1       (Stab#2 at 99%)
houstne     (Stab#3 at 88%)
covid_2020q2, covid_2020q3, covid_2020q4
```

Union of top 2 RF variables by permutation importance and top 3 Stability
variables by selection frequency. Outbs appears in both lists (4 unique vars).

**DoF constraint**: With 4 vars and p=1 VAR lag, parameters = 4² + 4 = 20 for
T=196 quarterly obs. T/params = 9.8 — safe per Sims (1980) T/3 rule.
For OLS with n_lags=3: 4 × 4 columns + intercept = 17 params, T/params = 11.5 —
very safe.

**Literature**: Sims (1980), Stock-Watson (2001) use 3–7 variables for
unpenalized VAR. Litterman (1986) uses 6. Our N=4 is standard.

### Category 3 — All other models (56 total: 53 features + 3 COVID)

```
a014re1q156nbea, acogno, ahetpix, amdmuox, andenox, awotman, busloans,
ce16ov, ces1021000001, ces2000000008, ces9091000001, ces9092000001,
clf16ov, compapff, cusr0000sas, ddurrg3m086sbea, dhlcrg3q086sbea,
difsrg3q086sbea, dodgrg3q086sbea, dongrg3q086sbea, dspic96, expgsc1,
fpix, gcec1, gpdic1, houstne, housts, hwiuratio, hwiuratiox, invest,
ipdcongd, liabpix, lns13023705, m2sl, manemp, mortg10yrx, nonrevsl,
ophpbs, outbs, outnfb, permitw, realln, slcex, spcs10rsa, tlbsnncbbdix,
uemp15t26, uemp27ov, uemplt5, ulcbs, ulcnfb, unrate, usgovt, usserv
covid_2020q2, covid_2020q3, covid_2020q4
```

Union of rule-determined sets at higher cumulative thresholds:
Lasso 95%, ElasticNet 95%, RF 95%, Stability 100% (all 35).

**Of 53 vars**: 26 have Stab≥50% confidence, 27 do not. The 27 without
bootstrap confidence are included because they carry linear/RF signal that
penalized models (Lasso Ridge EN) and internally-regularized models (RF, XGB,
GB, DT, MLP, LSTM, DeepVAR, DFM) can handle natively.

**Features/observations ratio**: 53 vars × 4 lags (n_lags=3, so current + 3 lag
columns) = 212 input features for T≈194 quarterly training obs. Ratio = 1.09 —
above 1:1 but manageable with regularization. See tuning_policy.md for MLP/LSTM
regularization requirements.

**Literature support**:
- Penalized (Lasso, Ridge, EN): De Mol-Giannone-Reichlin (2008) — regularization
  handles N>>T. Medeiros (2021) uses 122 vars.
- BVAR: Bańbura et al. (2010) — Minnesota prior handles 130 vars.
- Trees (RF, XGB, GB, DT): Medeiros (2021) uses 122 vars. Coulombe (2022) uses
  full FRED-MD (~134 vars).
- Neural (MLP, LSTM): Standard FRED-MD ~122 vars. Internal dropout/weight decay.
- DeepVAR: Standard ~125 vars.
- DFM: Stock-Watson (2002) uses 132 vars. Giannone-Reichlin-Small (2008) ~200.


---

## 8. Justification Summary

**Category 2 (VAR, OLS)**: Degres-of-freedom constraint. Unpenalized models
cannot handle N>>T. Top 2 RF vars capture the strongest non-linear signal;
top 3 Stab vars capture the most robust linear signal. Union = 4 vars —
within the Sims (1980) 3–7 variable range for small VAR. Both signals are
represented in a compact, defensible set.

**Category 3 (All others)**: Union of all four rule-determined sets at high
cumulative thresholds. Each method contributes on its own terms:
- Lasso/EN at 95% |coef|: captures virtually all linear signal (49/50 vars)
- RF at 95% perm imp: adds 2 unique non-linear signals (gpdic1, ophpbs)
- Stab at 100%: ensures all bootstrap-confirmed variables are included,
  adding realln (the only Stab var Lasso/EN/RF all miss)

The 53-variable union is what survives after all four filters. Every variable
has a defensible reason for inclusion: either strong linear coefficient, strong
RF permutation importance, or bootstrap selection frequency. Regularization
(L1/L2 penalties, dropout, bagging) handles the features/observations ratio
natively.

---

## 9. Data Files Reference

| File | Purpose |
|---|---|
| `variable_rankings_by_rule.txt` | Full Lasso/EN rankings (90 non-zero vars each), overlap tables, 95%/99% comparisons |
| `variable_rankings.txt` | Earlier top-35 RF/Stab rankings with category allocation tables |
| `variable_selection_framework.md` | This document — complete framework |
| `rank_by_rules.py` | Fits Lasso/EN with tuned HPs, computes all cumulative rankings |
| `rank_variables.py` | Reads existing RF/Stab rankings, outputs cutoff analysis |
| `feature_selection_ensemble.xlsx` | Raw feature selection output (8 sheets: Lasso, EN, RF, Stab, Union, Intersection, Overlaps, Run_metadata) |
