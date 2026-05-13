# Target Specification (Pipeline B)

## What is `gdpc1` in `data_tf_monthly.csv`?

`gdpc1` is FRED's "Real Gross Domestic Product, chained 2017 dollars,
seasonally adjusted at annual rates" series. In the raw FRED-QD file it is
quarterly. In `data_tf_monthly.csv` it is placed on a monthly grid: NaN in
non-Q-end months, and the **quarterly log first difference** at Q-end
months (March / June / September / December). This is tcode 5 applied to
the quarterly series.

## What does the model predict?

**Target y_t** = quarterly log change in real GDP, placed at the **Q-end month**.
Concretely, for quarter Q ending at month t in {Mar, Jun, Sep, Dec},
```
y_t = log(GDPC1_t) - log(GDPC1_{t-3})       # the value already in the CSV
```

This is **not** annualised in the data file. If you want to report
annualised growth in tables, multiply by 4 (or by 400 for percent
annualised). Internally, all loss functions (RMSFE, MAE) operate on `y_t`
directly so units do not matter for ranking.

## m1 / m2 / m3 vintages

For each forecast quarter Q with Q-end month t (e.g. Q1-2020 -> t = March
2020), three vintages are produced:

- **m1 vintage**: forecast made in the first month of Q (e.g. January for Q1).
  Forecast date = month t-2.
- **m2 vintage**: forecast made in the second month (February for Q1).
  Forecast date = month t-1.
- **m3 vintage**: forecast made in the third month (March for Q1).
  Forecast date = month t.

The forecast horizon is **the current quarter** for all three. m3 is the
"now-cast" with the most information; m1 is the most forward-looking
(closest to a 1-quarter-ahead forecast).

`gen_lagged_data(metadata, data, last_date=forecast_date, lag)` masks each
column's most-recent `months_lag` observations relative to `forecast_date`.
This simulates which data was available at that simulated point in time.

## Test target

We compare predictions to the **final revised value** of `gdpc1` as it
appears in our latest FRED-QD file (`fred-qd.xlsx`, vintage 2026-03).
This matches Bok et al. (2018) and most subsequent papers. We are
explicitly not doing real-time-vintage evaluation (which would require
ALFRED and is out of scope).

**Test window**: 2017-Q1 to 2025-Q4 (36 quarters). The vintage 2026-03 was
cut before BEA released the Q1 2026 advance estimate, so 2026-Q1 has no
realised target. Standard benchmark-paper practice is to use a frozen
vintage and document this rather than chase a moving target.

## Forecast horizons not produced

We forecast only the current quarter. We do not produce 1-quarter-ahead,
2-quarter-ahead, or backcast (-1Q) forecasts. This may be added later but
is out of scope for the initial benchmark.
