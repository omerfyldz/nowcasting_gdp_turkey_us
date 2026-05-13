# Project Instructions — Nowcasting Benchmark (US + Turkey Pipeline B)

## Turkey Model Notebooks

Turkey model notebooks must be **in line with US model notebooks** in architecture and structure, while **faithfully reflecting Turkey's own structure and dynamics**.

Before creating or editing any Turkey notebook, consult:
- Chat history for prior decisions
- `HANDOFFturkey.md` for Turkey-specific specs (dates, variables, paths, panel definitions)
- `turkey_data/turkey_helpers.py` and `turkey_data/turkey_variable_lists.json` for authoritative variable lists
- Existing Turkey notebooks already created (e.g. `model_ols_tr.ipynb`) for consistency

**Key Turkey differences from US (always apply):**
- Train: 1995-Q1 to 2011-Q4 / Val: 2012-Q1 to 2017-Q4 / Test: 2018-Q1 to 2025-Q4
- Target: `ngdprsaxdctrq`
- Cat3: 22 vars + 3 COVID = 25 (Python) / + `consu_i_weekly`, `deposit_i_weekly` for R MIDAS/MIDASML
- Weekly data: `data_tf_weekly_tr.csv` — columns `consu_i_weekly`, `deposit_i_weekly` (NOT `consu_i`/`deposit_i`)
- Predictions: `../turkey_predictions/<model>_tr_<vintage>.csv`
- Panels: Pre-crisis (2018-2019), COVID (2020-04-01 to 2021-12-31), Post-COVID (2022-2025), Full (2018-2025)
- sys.path for Python: `os.path.join(os.path.pardir, os.path.pardir, 'turkey_data')`
- R helpers: `source("../../data/helpers.R")`

## Deviations from Original Repo or Literature

Both the US pipeline and Turkey pipeline may contain bugs or errors. If making any deviation from the original repo, a US notebook, or established practice, **justify it by citing the literature** (e.g. Bok et al. 2018, Cascaldi-Garcia et al. 2024, Lenza-Primiceri 2022, McCracken-Ng 2016).

Do not silently deviate — state the reason and the reference.
