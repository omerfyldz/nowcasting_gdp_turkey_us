.libPaths(c("C:/Users/asus/R/library", "C:/Users/asus/AppData/Local/R/win-library/4.6", .libPaths()))

library(Rmisc)
library(tidyverse)
library(mfbvar)
library(imputeTS)

source("../data/helpers.R")

metadata <- read_csv("../turkey_data/meta_data_tr.csv", show_col_types = FALSE)
data <- read_csv("../turkey_data/data_tf_monthly_tr.csv", show_col_types = FALSE) %>%
  dplyr::arrange(date)

# Reduced Turkey BVAR predictor set.
# Full Cat3 has 22 predictors; mfbvar was infeasible for the US full Cat3
# variant. Use the locked Turkey Cat2 set for a stable, defensible BVAR:
# top RF/stability macro-financial indicators plus the quarterly GDP target.
bvar_features <- c("ipi_sa", "usd_try_avg", "cpi_sa", "fin_acc")

target_variable <- "ngdprsaxdctrq"

var_freqs <- metadata %>%
  dplyr::filter(series %in% bvar_features) %>%
  dplyr::select(series, freq)

monthly_vars <- var_freqs %>% dplyr::filter(freq == "m") %>% dplyr::pull(series)
quarterly_vars <- var_freqs %>% dplyr::filter(freq == "q") %>% dplyr::pull(series)
ordered_vars <- c(monthly_vars, quarterly_vars[quarterly_vars != target_variable], target_variable)

cat("Turkey BVAR vars:", length(ordered_vars), "(", length(monthly_vars), "monthly,",
    length(quarterly_vars), "quarterly)\n")

train_start_date <- "1995-01-01"
test_start_date <- "2018-01-01"
test_end_date <- "2025-12-01"

test <- data %>%
  dplyr::filter(date >= as.Date(train_start_date), date <= as.Date(test_end_date)) %>%
  data.frame()

for (col in colnames(test)) {
  if (is.numeric(test[, col]) && sum(is.infinite(test[, col])) > 0) {
    test[is.infinite(test[, col]), col] <- NA
  }
}

dates <- seq(as.Date(test_start_date), as.Date(test_end_date), by = "month")
dates <- dates[substr(dates, 6, 7) %in% c("03", "06", "09", "12")]

vintage_offsets <- c(m1 = -2, m2 = -1, m3 = 0)
pred_dict <- data.frame(date = dates)
for (lag_name in names(vintage_offsets)) pred_dict[, lag_name] <- NA

fit_bvar <- function(lagged_data, date, vars) {
  avail_vars <- vars[vars %in% colnames(lagged_data)]
  bvar_data <- lagged_data[, c("date", avail_vars), drop = FALSE]
  num_cols <- length(avail_vars)
  bvar_data <- bvar_data[
    rowSums(is.na(bvar_data[, -1, drop = FALSE])) < num_cols,
  ]

  mf_test <- list()
  for (col in avail_vars) {
    freq_str <- metadata %>% dplyr::filter(series == !!col) %>% dplyr::pull(freq)
    if (length(freq_str) == 0) next

    if (freq_str == "q") {
      tmp_series <- bvar_data %>%
        dplyr::filter(substr(date, 6, 7) %in% c("03", "06", "09", "12")) %>%
        dplyr::select(all_of(col)) %>% dplyr::slice(2:n()) %>% dplyr::pull()
      tmp_ts <- ts(tmp_series, start = c(1995, 2), frequency = 4)
    } else {
      tmp_series <- bvar_data %>%
        dplyr::select(all_of(col)) %>% dplyr::slice(2:n()) %>% dplyr::pull()
      tmp_ts <- ts(tmp_series, start = c(1995, 2), frequency = 12)
    }
    mf_test[[col]] <- tmp_ts
  }

  prior <- set_prior(Y = mf_test, n_lags = 4, n_reps = 20, block_exo = integer(0))
  c_interval <- t(sapply(mf_test, Rmisc::CI, ci = 0.95))
  prior_intervals <- c_interval[, c("upper", "lower")]
  moments <- interval_to_moments(prior_intervals)
  prior <- update_prior(prior, d = "intercept",
                        prior_psi_mean = moments$prior_psi_mean,
                        prior_psi_Omega = moments$prior_psi_Omega)
  prior <- update_prior(prior, n_fcst = 12)
  model <- estimate_mfbvar(prior, prior = "minn", variance = "iw")
  predict(model, pred_bands = NULL) %>%
    dplyr::filter(variable == target_variable, fcst_date == date) %>%
    dplyr::select(fcst) %>% dplyr::pull() %>% mean()
}

for (idx in seq_along(dates)) {
  date <- as.character(dates[idx])

  for (lag_name in names(vintage_offsets)) {
    vintage_date <- shift_month(date, vintage_offsets[[lag_name]])
    cat("Turkey BVAR fit", idx, "/", length(dates), lag_name, date, "\n")

    lagged_data <- gen_vintage_data(metadata, test, date, vintage_date)
    lagged_data <- data.frame(lagged_data)
    lagged_data[lagged_data$date == date, target_variable] <- NA

    n_rows <- nrow(lagged_data)
    fill_rows <- round(n_rows * 0.9)
    lagged_data[1:fill_rows, ] <- na_mean(lagged_data[1:fill_rows, ])

    pred <- tryCatch({
      fit_bvar(lagged_data, date, ordered_vars)
    }, error = function(e) {
      cat("ERROR at", date, lag_name, ":", conditionMessage(e), "\n")
      NA
    })

    pred_dict[pred_dict$date == as.Date(date), lag_name] <- pred
  }

  if (idx %% 4 == 0) cat(idx, "/", length(dates), "done\n")
}

actuals <- test %>%
  dplyr::filter(date %in% as.Date(dates)) %>%
  dplyr::select(!!target_variable) %>%
  dplyr::pull()

dir.create("../turkey_predictions", showWarnings = FALSE)
for (lag_name in names(vintage_offsets)) {
  df_out <- data.frame(
    date = dates,
    actual = actuals,
    prediction = pred_dict[, lag_name]
  )
  write.csv(df_out, paste0("../turkey_predictions/bvar_tr_", lag_name, ".csv"), row.names = FALSE)
}

panels <- list(
  "Pre-crisis (2018-2019)" = c("2018-01-01", "2019-12-31"),
  "COVID      (2020-2021)" = c("2020-04-01", "2021-12-31"),
  "Post-COVID (2022-2025)" = c("2022-01-01", "2025-12-31"),
  "Full test  (2018-2025)" = c("2018-01-01", "2025-12-31")
)
rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
mae <- function(a, p) mean(abs(a - p), na.rm = TRUE)
d <- as.Date(dates)
cat("BVAR (Turkey) -- Evaluation by Panel & Vintage\n")
for (pn in names(panels)) {
  rng <- panels[[pn]]
  m <- d >= rng[1] & d <= rng[2]
  cat("---", pn, "---\n")
  for (lag_name in names(vintage_offsets)) {
    cat("  ", lag_name, "  RMSFE=",
        format(rmse(actuals[m], pred_dict[m, lag_name]), digits = 6),
        "  MAE=", format(mae(actuals[m], pred_dict[m, lag_name]), digits = 6),
        "  N=", sum(m), "\n")
  }
}
