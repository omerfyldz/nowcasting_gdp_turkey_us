# HANDOFF — Nowcasting Benchmark Pipeline

**Document purpose**: Structured handoff for the next LLM (or human collaborator) picking up this project. Read top to bottom; this is a self-contained briefing, not a chat log. Where deeper context is useful, pointers to the plan files and prior chats are at the bottom.

**Project**: Nowcasting US real GDP (`gdpc1`) on top of `nowcasting_benchmark-main`, a published 17-model benchmark by Hopp (Bańbura-style framework). Original repo uses 24 hand-picked variables; we have rebuilt it into "**Pipeline B**" with 296 variables, COVID dummies, ensemble feature selection, rigorous train/val/test windows, and per-model scaling/tuning policies.

**Current status**: Data pipeline is **gate-clear for notebook adaptation**. Model notebooks have NOT been adapted yet.

**Author of this handoff**: Claude Sonnet 4.7 in Claude Code, session of 2026-05-02.

---

## 1. Quick orientation — where we are right now

| Stage | State |
|---|---|
| Source data ingestion (FRED-MD + FRED-QD + user data) | ✅ Done, hardened with hard-fail range checks |
| Tcode transformations + COVID dummies | ✅ Done |
| Metadata (freq / blocks / publication lags) | ✅ Done, 303 rows, source-based logic |
| Stationarity audit | ✅ 220 STATIONARY / 76 INCONCLUSIVE / 0 NON_STATIONARY |
| Ensemble feature selection (Lasso + ElasticNet + RF + Lasso stability) | ✅ Done, rigorous |
| Policy docs (scaling / tuning / target / evaluation) | ✅ Written, all consistent |
| Visual sanity check | ✅ Plot exists |
| **Model notebook adaptation** | ⏳ **NOT STARTED — this is the next phase** |

---

## 2. Critical history — bugs we caught and corrected

This project went through **two prior LLM sessions** (Claude Code, then Gemini) before reaching the current state. Several substantive bugs were introduced and later corrected. Knowing them prevents reintroducing them:

### 2.1 The "60 % NaN ⇒ quarterly" heuristic (Gemini introduced, we removed)
- **What it was**: For the 296 unknown variables, Gemini wrote `if >60% NaN over last 10 years, tag as freq=q`.
- **Why it broke**: Some monthly variables (e.g. `compapff`) had recent gaps in the FRED file → mis-tagged as quarterly. Silent contamination of the entire metadata.
- **Current fix**: `build_metadata.py` uses **source-based assignment** — FRED-MD columns → `m`, FRED-QD-only → `q`, user dictionary → per `Frequency` column. No NaN counting anywhere. The bug cannot recur.

### 2.2 `gdpc1` `months_lag` (initially 1, then 4, finally 2)
- **What happened**: Gemini set `lag=1` then later `lag=4` on the wrong reasoning that "Q1 data released in April = month 4".
- **Why both were wrong**: On a monthly grid where the Q-end value sits in the final month of the quarter (March for Q1), `lag=2` is the safe Bok et al. (2018) convention — it prevents the m1 vintage from seeing data not yet released by BEA's Advance estimate (~30 days post-quarter-end).
- **Current fix**: All NIPA-quarterly hard-data series have `lag=2` in `meta_data.csv` via the `_QUARTERLY_LAG2` set in `build_metadata.py`.

### 2.3 Per-variable override dicts (Gemini introduced, we eliminated)
- **What it was**: `TOP35_BLOCKS = {35 hand-coded entries}`, leaving 270 variables defaulted to "Global only". Methodologically indefensible.
- **Current fix**: `build_metadata.py` now classifies ALL 296 variables systematically:
  - FRED-MD: column position → McCracken-Ng (2020) group → Bok block
  - FRED-QD-only: documented economic-category sets
  - User vars: from `Economic Category` column in `US_data_dictionary.xlsx`
  - HWI / HWIURATIO cross-listed in Soft + Labor (Bok 2018 convention)
  - All 8 lookup tables now have `# References:` header citing McCracken-Ng 2020, Bok et al. 2018, Bańbura-Modugno 2014.

### 2.4 Scale corruption in xlsx files (user found, we hardened)
- **What happened**: When user converted source CSVs to xlsx, some values were silently rescaled (`2.04 → 204`, `2.0 → 20`). Discovered when raw range checks reported `UNRATE max=148`, `FEDFUNDS max=1908`, `CPIAUCSL max=326588`.
- **Old behaviour**: Range checks printed FAIL but were marked "non-blocking, pre-existing" → bug slipped through to feature selection.
- **Current fix**:
  1. User uploaded corrected xlsx (sheet name now `'Worksheet'`, scales correct).
  2. `build_raw_data.py` range checks promoted to **hard `assert`** → any future scale corruption halts the pipeline immediately.
  3. Pipeline rebuilt from scratch on corrected data; UNRATE max=14.8, FEDFUNDS max=19.1, CPIAUCSL max=327.46. All 15 raw checks PASS.

### 2.5 The Q3 `apply_tcode` patch — introduced, then REVERTED
- **What I claimed was a bug**: The original `s.dropna().diff()` collapses the time grid; if a series has a mid-stream NaN, the diff at the next observation spans multiple periods (a "1-month log diff" that is actually a 2-month log diff).
- **What I patched it to**: `s.where(s>0)` to keep the original grid; let NaN propagate through diff naturally.
- **Why I reverted**: My patch was wrong-headed. For quarterly series on a monthly grid (gdpc1 has values only at Q-end months), the original `dropna()` correctly computes Q-to-Q log diffs. With my patched version, `diff()` between Q-end (March) and the previous month (February, NaN) yields NaN — `gdpc1` ended up with **0 non-NaN values**. Catastrophic regression.
- **Current state**: Reverted to the original `dropna()`-first approach. `gdpc1` first-3 Q-end values now correctly equal `(0.022, 0.0007, 0.003)`. The `inf`-handling addition (for `pct_change` with zero denominator) is kept — that part of the patch was correct.
- **Lesson**: For irregular missingness, `.diff()` after `dropna()` is "change since last observation" — which is the right quantity for both irregular gaps and systematic quarterly NaN structure. My original "bug diagnosis" overstated the problem.

---

## 3. The data folder — file inventory

All files live in `C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\data\`.

### 3.1 Source data (input)
| File | Description |
|---|---|
| `fred-md.xlsx` | FRED-MD 2026-03 release. Sheet `Worksheet`. 126 monthly series. Row 0 = `Transform:` + tcodes; rows 1+ = data. Date column `sasdate`. |
| `fred-qd.xlsx` | FRED-QD 2026-03 release. Sheet `Worksheet`. 245 series. Row 0 = `factors`; row 1 = `transform` + tcodes; rows 2+ = data. |
| `us_master_monthly.xlsx` | User's own collected data. Sheet `MASTER`. 40 columns including some FRED overlaps (deduped via `USER_MONTHLY_KEEP` in `build_raw_data.py`) and unique series (CFNAI, NFCI_monthly_avg, BOPGSTB, ...). |
| `daily_weekly_series.xlsx` | Raw input for `build_weekly_data.py`. Has DTWEXM/DTWEXBGS daily + ICSA_weekly, NFCI_weekly. |
| `US_data_dictionary.xlsx` | 44 rows. Columns: Variable, Frequency, Economic Category, Release Lag, Source, Notes. **Authoritative for user vars**. Used by `build_metadata.py` for freq + block + lag of user series. |

### 3.2 Build scripts
| Script | Purpose |
|---|---|
| `build_weekly_data.py` | Daily DTWEX → EoP weekly (`.resample('W-SAT').last()`); native weekly (ICSA, NFCI) Saturday-aligned. Drops Fed-averaged DTWEXM_weekly / DTWEXBGS_weekly per user decision. Output: `data_weekly_aligned.xlsx` (3095 × 4 + tcode row + Date). |
| `build_raw_data.py` | Merges FRED-MD + FRED-QD-only + user-monthly + user-quarterly into single monthly grid. Hard-fail range checks (UNRATE, FEDFUNDS, CPIAUCSL, GDPC1, TCU, CFNAI). Output: `data_raw_monthl.xlsx` (1289 × 297). |
| `build_final_tf_data.py` | Applies McCracken-Ng tcodes (`apply_tcode`) + COVID dummies + lowercase column names. `TCODE_OVERRIDES = {'NWPIx': 5, 'HWIx': 5}` for stationarity. Outputs: `data_tf_monthly.csv` (1288 × 300), `data_tf_weekly.csv` (3095 × 8). |
| `build_metadata.py` | Source-based freq + position-based FRED-MD blocks + QD-only category sets + user-dict block mapping + HWI cross-listing + comprehensive lag lookup. Output: `meta_data.csv` (303 rows). |
| `run_stationarity.py` | ADF + KPSS + PP triple test on each transformed series. Output: `stationarity_report.xlsx`. |
| `feature_selection_ensemble.py` | **Rigorous** ensemble: LassoCV (TimeSeriesSplit, wide α path), ElasticNetCV (widened l1_ratio grid `[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]`), tuned RF + permutation_importance with `n_jobs=1`, Lasso stability selection (100 resamples, 75% subsample). 1959-2007 training window. Output: `feature_selection_ensemble.xlsx` (8 sheets) + `.txt` companion. |
| `visual_sanity_check.py` | Plots `gdpc1` + 5 representative top-35 series. Output: `visual_sanity_check.png`. |

### 3.3 Output / build artefacts
| File | Shape / Description |
|---|---|
| `data_raw_monthl.xlsx` | 1289 × 297 (1 tcode row + 1288 data rows × 296 vars + date). All 15 range checks PASS. |
| `data_tf_monthly.csv` | (1288, 300) = 296 transformed vars + 3 COVID dummies + date. `gdpc1` first-3 = (0.022, 0.0007, 0.003) ✅ |
| `data_tf_weekly.csv` | (3095, 8) = 4 weekly vars (icsa_weekly, nfci_weekly, dtwexbgs, dtwexm) + 3 COVID dummies + Date. |
| `data_weekly_aligned.xlsx` | (3095, 4) Saturday-aligned. |
| `meta_data.csv` | 303 rows. freq {m:149, q:150, w:4}. Lag {0:107, 1:80, 2:116}. Blocks {g:303, s:12, r:130, l:76}. **Full data↔meta consistency verified.** |
| `stationarity_report.xlsx` | monthly sheet: 220 STATIONARY / 76 INCONCLUSIVE / 0 NON_STATIONARY. Weekly sheet: 3 STATIONARY / 1 INCONCLUSIVE. |
| `feature_selection_ensemble.xlsx` | 8 sheets: `Lasso`, `ElasticNet`, `RF_perm_imp`, `Stability`, `Union`, `Intersection`, `Overlaps`, `Run_metadata`. Programmatically loadable via `pd.read_excel(..., sheet_name='Lasso')`. |
| `visual_sanity_check.png` | Six-panel matplotlib plot. |

### 3.4 Policy / decision documents (markdown)
| File | Purpose |
|---|---|
| `scaling_policy.md` | Per-model standardization decisions + emphatic ordering rules for the rolling loop (gen_lagged_data → mean_fill → scaler.fit → model.fit, training-fold only). |
| `tuning_policy.md` | Per-model HP-tuning decisions + grids. Train/val/test split (1959-2007 / 2008-2016 / 2017-Q4 2025). |
| `target_specification.md` | What `gdpc1` is, how `y_t` is constructed, m1/m2/m3 vintage definitions, test horizon (2017-Q1 to 2025-Q4 = 35 quarters). |
| `evaluation_protocol.md` | RMSFE / MAE definitions, AR(1) and RW benchmarks, Diebold-Mariano test, 4-panel sub-window reporting (pre-COVID / COVID / post-COVID / full). Model table with per-model variable-list source. |
| `requirements.txt` | Pinned package versions for the data pipeline (pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, openpyxl). Notebook dependencies (nowcast_lstm, midasmlpy, torch) NOT yet pinned — add at notebook start. |

### 3.5 What got DELETED (not on disk anymore)
- `lasso_feature_selection.py` and `lasso_selected_vars.txt` — superseded by ensemble selector.
- `verify_pipeline.py` — served its gate-keeping role; removed once gate cleared. If a new gate check is needed later, write a fresh one.
- `data_raw.csv` and `data_tf.csv` — stale 24-variable files from the original repo. **DANGER**: if these reappear, original-repo notebooks reading them by their literal filename would silently load the wrong data. Do not regenerate.

---

## 4. Key design decisions and their rationale

These are the load-bearing choices. Every one was deliberated and should not be reversed without a clear new reason.

### 4.1 Pipeline B (296 vars + COVID dummies + Lasso selection + tuned tcodes)
**Decision**: Build an "improved" pipeline rather than faithfully replicating the original 24-variable Bok et al. benchmark.
**Cost**: Results are no longer directly comparable to published Bok-style numbers — we are our own benchmark.
**Rationale (user-confirmed)**: User wants operational nowcasting accuracy, not benchmark replication. AR(1) and RW benchmarks within Pipeline B provide the relative-loss comparison.

### 4.2 Train: 1959-2007, Validation: 2008-2016, Test: 2017-Q1 to 2025-Q4
**Decision**: Replaced the prior split (train 1959-2010, val 2011-2016) with this one.
**Rationale**: The previous validation window (2011-2016) had **no recession**. HPs tuned there could not be evaluated for crisis robustness. The revised split puts a major shock in BOTH validation (GFC) and test (COVID), so HPs that survive GFC have a fighting chance on COVID. Cascaldi-Garcia et al. 2024 use the same partition logic.
**Test horizon cap at 2025-Q4**: FRED-QD vintage 2026-03 was cut before BEA's Q1 2026 advance estimate. Standard benchmark practice: use a frozen vintage and document the limit.

### 4.3 Source-based frequency assignment (no heuristics)
**Decision**: Every variable's frequency comes from its source file, not from NaN-counting:
- FRED-MD columns → `m`
- FRED-QD-only columns → `q`
- User dict `Frequency` column → per dictionary (`monthly` → m, `quarterly` → q, `weekly` → w, `daily` → m if pre-aggregated)
- Weekly file → `w`
- COVID dummies → `m`
**Rationale**: Heuristic frequency detection (the prior `>60% NaN ⇒ q` rule) silently mis-tagged `compapff` and others. Source-based is deterministic and documented.

### 4.4 NIPA quarterly hard-data lag = 2 (Bok convention)
**Decision**: GDPC1 and other NIPA-quarterly series in `_QUARTERLY_LAG2` get `months_lag = 2`.
**Rationale**: Q-end value sits at the last month of the quarter (March for Q1). Advance estimate releases ~30 days post-quarter-end (late April). Safe lag = 2 prevents the m1 vintage (forecast made early in the next quarter) from seeing data that wasn't yet released. Gemini's previous `lag=4` was wrong-headed; `lag=1` would leak.

### 4.5 EoP for daily-to-weekly aggregation; drop Fed-averaged FX series
**Decision**: `data_weekly_aligned.xlsx` keeps only EoP-aggregated `dtwexbgs` / `dtwexm` (via `.resample('W-SAT').last()`); the Fed-published 5-day-average `dtwexm_weekly` / `dtwexbgs_weekly` are dropped.
**Rationale**: Mixing 5-day averages with EoP timestamps creates smoothing bias that contaminates MIDAS lag weights (Bańbura-Modugno 2014). Saturday alignment matches ICSA's natural week.

### 4.6 COVID dummies (3 quarterly indicators)
**Decision**: `covid_2020q2`, `covid_2020q3`, `covid_2020q4` are 0/1 columns added to BOTH monthly and weekly transformed files. `lag = 0` in metadata.
**Rationale**: COVID 2020Q2 produced the largest CFNAI deviation in history (-18.28). Without dummies, ALL pre-COVID-trained models (DFM factors, BVAR coefficients, Lasso α, etc.) would be distorted. Dummies absorb the level shock per Lenza-Primiceri 2022. NOT a literature-universal choice (some papers use t-distributed innovations, some skip 2020) but defensible and easy.

### 4.7 Per-model standardization
**Decision** (in `scaling_policy.md`):
- Scale: BVAR, DFM, Lasso, Ridge, ElasticNet, MIDAS, MIDASML, MLP, LSTM
- Don't scale: OLS, ARMA, VAR, DeepVAR, RF, XGB, GB, DT
- COVID dummies always pass through unscaled (zero-variance in 1959-2007 → would NaN the scaler).
**Rationale**: Standardization addresses penalty fairness (Lasso/Ridge) and gradient conditioning (MLP/LSTM); it does NOT solve structural breaks. Trees and OLS are scale-invariant.

### 4.8 Per-model HP tuning
**Decision** (in `tuning_policy.md`):
- Tune via `TimeSeriesSplit(5)` on validation window: Lasso α, Ridge α, ElasticNet α + l1_ratio, RF (max_features × min_samples_leaf grid), XGB / GB / DT (small grids), MLP (architecture + alpha), LSTM (units + layers + dropout).
- Don't tune (use literature defaults): OLS, ARMA, VAR, DeepVAR, DFM (1 factor per block), BVAR (Minnesota tightness λ₁=0.1).
**Rationale**: Models with no real HPs gain nothing from tuning except CV noise. Models with HPs MUST be tuned on a window that contains at least one shock — the GFC-inclusive validation (2008-2016) ensures tuned HPs are stress-tested.

### 4.9 Ensemble feature selection (Lasso + ElasticNet + RF + Lasso stability)
**Decision**: Run all four on 1959-2007, top-35 ranked per method, save to `feature_selection_ensemble.xlsx`.
**Rationale**: Single-method selection is fragile in high-dim (1180 features × 195 obs). Lasso-vs-RF disagreement is informative (linear vs non-linear signals). Stability selection (Meinshausen-Bühlmann 2010) gives confidence intervals on Lasso picks (only 7/35 are at ≥80% frequency — the high-dim regime is genuinely noisy below the top ~10).

### 4.10 Selection lists per model class (recommended)
**Decision** (NOT YET CODED — this is for notebook stage):
| Model class | Variable list | Rationale |
|---|---|---|
| AR(1), RW | just `gdpc1` | Univariate benchmarks |
| ARMA | just `gdpc1` | Univariate |
| OLS, Lasso, Ridge, ElasticNet, MIDAS, MIDASML, BVAR, VAR | Lasso top-35 + COVID dummies | Linear class needs sparse + targeted; Lasso is interpretable |
| DFM | All 296 + 4-block structure from `meta_data.csv` | Factor model uses all data; selection done by factor extraction |
| RF, XGBoost, GB, DT, MLP, LSTM, DeepVAR | Union (~65 vars) + COVID dummies | Non-linear class can absorb wider set; trees pick what they want |
| Conservative-linear option | Lasso stability top-25 (frequency ≥ 50%) | Maximum referee defensibility |

### 4.11 No README rewrite, no helpers.py yet, no smoke test yet
**Deferred to notebook stage by user**: see Section 6.

---

## 5. The 11 methodological questions — compact verdicts

The user asked these in mid-session. Verdicts here; deep discussion in plan files.

| # | Question | Verdict | One-line answer |
|---|----------|---------|-----------------|
| Q1 | CFNAI bound widened for 2020 -18.28? | ✅ | -20 lower bound is correct. Literature **never** truncates COVID; absorb via dummies (we have). |
| Q2 | Frequencies + EoP correctness? | ✅ | Source-based rule + `.resample('W-SAT').last()` for EoP. Both correct. |
| Q3 | dropna / diff / log(0) interactions? | ✅ | After my patch-then-revert: original `dropna()`-first approach is correct for quarterly-on-monthly grid. `inf` from `pct_change` is now sanitised. |
| Q4 | Per-variable override dicts bad practice? | ✅ | Eliminated. Lookup tables citing McCracken-Ng / Bok et al. are systematic, not overrides. |
| Q5 | Log-safety holes for OLS/Lasso? | ✅ | `apply_tcode` masks; `mean_fill_dataset(train, test)` per-fold inside rolling loop. **LOCF/AR(1) imputation is wrong** for ragged-edge per Bańbura-Modugno 2014. |
| Q6 | "60% NaN ⇒ quarterly" still problematic? | ✅ | Replaced with deterministic source-based rule. Cannot recur. |
| Q7 | Structural breaks + scaling? | ✅ | Standardization ≠ structural break fix. Use dummies (COVID), rolling estimation (GFC), tcodes (Great Moderation). All three present. |
| Q8 | NaN / zero handling? | ✅ | `apply_tcode` reverted (correct now); inf neutralised; min-obs floor verified (min=104, `spcs20rsa`). |
| Q9 | Add RF + EN to selection? | ✅ | Done. Low Lasso/RF overlap (5/35) is **expected**, not a bug — linear vs non-linear methods reveal different signals. |
| Q10 | Validation set for HP tuning? | ✅ | Mandatory for tunable models. Window: 2008-2016 (includes GFC). `TimeSeriesSplit(5)` inside. |
| Q11 | What else before notebooks? | ✅ | Pipeline gate-clear. Five items deferred to notebook stage (see Section 6). |

---

## 6. Items deferred to notebook stage (DO NOT FORGET)

These were raised during pre-flight but explicitly deferred by the user. Each is a notebook-stage prerequisite, not a future enhancement. Build them BEFORE writing the first model notebook.

### 6.1 Build `helpers.py` shared module
**What**: Single-source-of-truth Python module containing `flatten_data`, `mean_fill_dataset`, `gen_lagged_data`, `get_features` (loads from `feature_selection_ensemble.xlsx`), `split_for_scaler` (separates COVID dummies).
**Why**: Without it, every notebook redefines these functions and they drift. Critique A (forwarded by user, accepted): drift kills experimental control. If LSTM beats VAR you cannot tell whether the win is architectural or data-cleaning.

### 6.2 Smoke-test `gen_lagged_data` with new metadata
**What**: Build a test for forecast date 2018-Q2 m2 vintage. Confirm:
- `gdpc1` is masked (lag=2 → last value at 2018-03 should be hidden when forecasting from 2018-05)
- `unrate` (lag=0) is NOT masked at the forecast date
- A quarterly NIPA series (lag=2) is masked deeper than monthly hard data (lag=1)

**Why**: The original repo's function does `metadata.loc[metadata.series == col, "months_lag"].values[0]`. If any column in `data_tf_monthly.csv` is missing from `meta_data.csv`, this throws `IndexError`. We verified the consistency (Section 9), but a runtime test inside a real rolling-loop pattern catches edge cases.

### 6.3 Document COVID-dummy injection rule
**What**: Add a notebook-level policy doc (or top-of-helpers.py docstring) stating:
> "Linear notebooks (DFM, BVAR, OLS, Lasso, Ridge, EN, MIDAS, MIDASML) MUST manually append `covid_2020q2`, `covid_2020q3`, `covid_2020q4` to their feature list before fitting. Lasso prunes them in selection because they are zero-variance in the 1959-2007 training window. Without this injection, the linear model has no defence against the 2020Q2 shock and RMSFE on the COVID test sub-window will be catastrophic."

**Reference implementation**:
```python
COVID = ["covid_2020q2", "covid_2020q3", "covid_2020q4"]

def get_features(method="Lasso", top_k=35, with_covid=True):
    df = pd.read_excel(f"{BASE}/feature_selection_ensemble.xlsx",
                        sheet_name=method)
    feats = df.head(top_k)["feature"].tolist()
    return feats + COVID if with_covid else feats

def split_for_scaler(features):
    """Return (cols_to_scale, cols_to_pass_through_unscaled)."""
    scaled = [c for c in features if c not in COVID]
    passed = [c for c in features if c in COVID]
    return scaled, passed
```

### 6.4 Create `data/predictions/` directory
**What**: `mkdir predictions` so each notebook can save `predictions/<model>_<vintage>.csv` per-iteration forecast records.
**Why**: `evaluation_protocol.md` requires this for the evaluation script to find outputs without re-fitting.

### 6.5 Final per-model variable-list mapping
**What**: Once notebooks start, lock in the exact list each model uses (per Section 4.10). Document the assignment in each notebook's header cell. Re-evaluate after notebook 1 if anything looks wrong.

---

## 7. Critical pitfalls — things to NOT undo

These are non-obvious choices. Any of them, if reversed without thinking, breaks the pipeline or invalidates the experiment.

1. **Do NOT remove the hard-fail range checks in `build_raw_data.py`.** They caught the scale-corruption bug. Future scale corruption will be caught only because of these asserts.
2. **Do NOT re-introduce per-variable override dicts** (`TOP35_BLOCKS`, hand-coded freq tables, etc.). Use systematic code or a citation-backed lookup. The references header in `build_metadata.py` lists what's defensible.
3. **Do NOT switch `gdpc1` lag back to 1 or 4.** Lag = 2 is the Bok et al. convention; 1 leaks Advance GDP into m1 vintage; 4 is over-conservative.
4. **Do NOT re-patch `apply_tcode` to keep the time grid in `.diff()`.** I tried this; it broke quarterly series (gdpc1 became all NaN). The original `dropna()`-first approach is correct.
5. **Do NOT use random K-fold CV anywhere.** Always `TimeSeriesSplit`. This was the most-violated rule in the original repo's notebooks.
6. **Do NOT fit the scaler / mean-fill on data that includes the test slice.** Always training-fold only, inside the rolling loop.
7. **Do NOT regenerate `data_raw.csv` or `data_tf.csv`** (the original repo's stale 24-variable files). If they reappear, original-repo notebooks reading them by literal filename would silently load wrong data.
8. **Do NOT scale COVID dummies.** They are zero-variance in the training fold (pre-2020). The scaler will produce NaN. Pass them through unchanged.
9. **Do NOT re-run feature selection inside the rolling loop without explicit decision.** The current design freezes Lasso/EN/RF picks once, computed on 1959-2007. Per-iteration re-selection adds variance and ~10x compute. If you want it as a robustness specification, that's fine — but be explicit and document.
10. **Do NOT add a 2008 dummy.** GFC is handled implicitly via rolling estimation. Bok and most subsequent papers do NOT use a GFC dummy. Adding one over-fits the training window.
11. **Do NOT use LOCF or AR(1) imputation for ragged-edge.** Bańbura-Modugno 2014, Schorfheide-Song 2015, Bok et al. 2018 all use Kalman smoothing (DFM/state-space) or unconditional mean-fill (linear/ML). LOCF biases nowcasts toward staleness.

---

## 8. Suggested first notebook to adapt

Start with the **simplest possible model** to debug the harness end-to-end before tackling complex models. Recommended order:

1. **AR(1) benchmark** (`model_ar1.ipynb` — needs to be created; the original repo doesn't have it explicitly but `model_arma.ipynb` is close). This is the no-information baseline against which every other model is judged.
2. **OLS with Lasso top-35** (`model_ols.ipynb`). Validates the rolling loop, mean-fill, gen_lagged_data, predictions/, and evaluation glue.
3. **Lasso with HP tuning** (`model_lasso.ipynb`). First test of the validation-window HP search.
4. **DFM** (`model_dfm.ipynb`). First test of the 4-block structure from `meta_data.csv`.
5. **RF** (`model_rf.ipynb`). First ML / Union-list test.

Once these five work and produce sane RMSFE numbers vs AR(1), the pattern is established and the remaining 12 notebooks follow the same template.

---

## 9. Verification checklist (run before notebook 1)

```python
import pandas as pd
BASE = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data/"

# Data files exist and have right shape
m  = pd.read_csv(BASE + "data_tf_monthly.csv", parse_dates=["date"])
w  = pd.read_csv(BASE + "data_tf_weekly.csv",  parse_dates=["Date"])
md = pd.read_csv(BASE + "meta_data.csv")
fs = pd.ExcelFile(BASE + "feature_selection_ensemble.xlsx")

assert m.shape == (1288, 300)
assert w.shape == (3095, 8)
assert len(md) == 303
assert set(fs.sheet_names) >= {"Lasso", "ElasticNet", "RF_perm_imp",
                                "Stability", "Union", "Intersection",
                                "Overlaps", "Run_metadata"}

# Data <-> metadata consistency (no orphans)
m_cols = {c.lower() for c in m.columns if c.lower() != "date"}
w_cols = {c.lower() for c in w.columns if c.lower() != "date"}
meta_set = set(md["series"].str.lower())
assert m_cols.issubset(meta_set), "monthly cols missing from meta"
assert w_cols.issubset(meta_set), "weekly cols missing from meta"

# Target sanity
assert (m[["date","gdpc1"]].dropna().head(3)["gdpc1"].round(4).tolist()
        == [0.0223, 0.0007, 0.0028])

# Date range supports test window
assert m.date.max() >= pd.Timestamp("2026-04-01")
assert m[["date","gdpc1"]].dropna().date.max() == pd.Timestamp("2025-12-01")

print("ALL GATE CHECKS PASS — proceed to notebook adaptation.")
```

---

## 10. Pointers to deeper context (read these only if you need WHY)

1. **`C:\Users\asus\.claude\plans\you-are-a-nobel-gentle-sparkle.md`**
   The "11 questions" professorial review. Detailed verdicts on each methodological question with literature citations (McCracken-Ng, Bok et al., Bańbura-Modugno, Coulombe et al., Goulet-Coulombe, Lenza-Primiceri, etc.). Includes the LOCF-is-wrong critique evaluation.

2. **`C:\Users\asus\.claude\plans\1-it-seems-we-concurrent-anchor.md`**
   The cross-check + rebuild plan. Documents the scale-corruption finding, the file-deletion rationale, the rebuild order, the deferred-to-notebook list, and the deep-think on what to attend to in model notebooks.

3. **`C:\Users\asus\Desktop\claudecodehistorynowcasting.md`**
   Earlier Claude Code session. Foundational decisions (Pipeline B, COVID dummies yes, no IQR removal, 1959+ sample, 25/35 var Lasso). Honest self-audit at the end where the LLM admits to fabricating a "5-15% RMSE improvement" number and overstating COVID dummies as "industry standard". Useful context for the design choices.

4. **`C:\Users\asus\Desktop\gemini session history the last.md`**
   Gemini session that took over from Claude Code. Built the variable deduplication logic (16 user vars dropped as FRED duplicates, 22 unique kept). Introduced the bugs we later corrected: 60%-NaN heuristic, gdpc1 lag=4, top-35-only block dict. Ended mid-task with the user instructing "take all fred-md as m, fred-qd-only as q..." which is what `build_metadata.py` now implements correctly.

5. **`C:\Users\asus\Desktop\nowcasting_benchmark-main_original_dont_touch\`**
   The original repo, untouched. **READ ONLY** — never write here. Useful for comparing original `gen_lagged_data`, `mean_fill_dataset`, `flatten_data` implementations and the original notebook templates.

---

## 11. One-paragraph summary for very fast handoff

This is a 296-variable nowcasting pipeline for US real GDP, built as "Pipeline B" on top of the published Bok-style benchmark (24 vars). The data layer is complete: source-based metadata, McCracken-Ng tcodes with one stationarity override, COVID dummies in both monthly and weekly files, EoP daily-to-weekly aggregation, hard-fail range checks, source-based train/val/test windows (1959-2007 / 2008-2016 / 2017-Q1 to 2025-Q4) with both validation and test containing a major shock (GFC and COVID respectively). Feature selection uses a four-method ensemble (Lasso, ElasticNet, RF permutation importance, Lasso stability) with TimeSeriesSplit CV on the training window, output to `feature_selection_ensemble.xlsx` with 8 programmatic sheets. Per-model scaling and HP-tuning policies are documented. The next phase is model-notebook adaptation: build `helpers.py` with `flatten_data` / `mean_fill_dataset` / `gen_lagged_data` / `get_features` / `split_for_scaler` shared module, smoke-test `gen_lagged_data`, append COVID dummies to linear notebooks' feature lists, mkdir `predictions/`, then adapt notebooks in order AR(1) → OLS → Lasso → DFM → RF → rest. Critical do-nots: do not re-introduce per-variable override dicts; do not switch `gdpc1` lag from 2; do not use random K-fold CV; do not scale COVID dummies; do not regenerate the deleted stale `data_tf.csv`.

---

*End of HANDOFF. Document length: ~430 lines markdown. All numbers in this document were verified at the time of writing (2026-05-02).*
