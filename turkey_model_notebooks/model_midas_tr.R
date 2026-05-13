options(warn = -1)
library(tidyverse)
library(midasr)
library(forecast)
library(imputeTS)

source("../data/helpers.R")

metadata <- read_csv("../turkey_data/meta_data_tr.csv", show_col_types = FALSE)
data <- read_csv("../turkey_data/data_tf_monthly_tr.csv", show_col_types = FALSE) %>%
    arrange(date)
data_weekly <- read_csv("../turkey_data/data_tf_weekly_tr.csv", show_col_types = FALSE) %>%
    rename(date = Date) %>% arrange(date)

# Inf check on weekly data
for (col in colnames(data_weekly)) {
    if (col == "date") next
    if (sum(is.infinite(data_weekly[[col]])) > 0) {
        data_weekly[is.infinite(data_weekly[[col]]), col] <- NA
    }
}

# Cat3 features explicitly dropping monthly 'consu_i' and 'deposit_i'
cat3_features <- c(
  "altin_rezerv_var", "auto_prod", "bist100", "cpi_sa",
  "doviz_rezerv_var", "emp_rate", "empl_num", "fin_acc", "ipi_sa", "m3",
  "maden_ciro_endeksi_sa", "ppi", "reer", "resmi_rezerv_var", "tax",
  "total_prod", "tourist", "unemp_num", "unemp_rate", "usd_try_avg",
  "covid_2020q2", "covid_2020q3", "covid_2020q4"
)
weekly_vars <- c("consu_i_weekly", "deposit_i_weekly")

cat("MIDAS Turkey: ", length(weekly_vars), "weekly, ", length(cat3_features), "monthly\n")

target_variable <- "ngdprsaxdctrq"
train_start_date <- "2002-01-01"   # aligned with weekly data start
test_start_date  <- "2018-03-01"
test_end_date    <- "2025-12-01"
test_dates <- seq(as.Date(test_start_date), as.Date(test_end_date), by = "3 months")

test <- data %>%
    filter(date >= as.Date(train_start_date), date <= as.Date(test_end_date)) %>%
    data.frame()

for (col in colnames(test)) {
    if (sum(is.infinite(test[, col])) > 0) {
        test[is.infinite(test[, col]), col] <- NA
    }
}

vintage_offsets <- c(m1 = -2, m2 = -1, m3 = 0)
pred_dict <- data.frame(date = test_dates)
for (lag_name in names(vintage_offsets)) pred_dict[, lag_name] <- NA

# Helper: extract weekly series for a date range
get_weekly_series <- function(w_data, col, start_date, end_date, n_quarters, weeks_per_q = 13) {
    w_sub <- w_data %>%
        filter(date >= as.Date(start_date), date <= as.Date(end_date)) %>%
        pull(!!col)
    expected_len <- n_quarters * weeks_per_q
    n_actual <- length(w_sub)
    if (n_actual < expected_len) {
        w_sub <- c(rep(NA, expected_len - n_actual), w_sub)
    } else if (n_actual > expected_len) {
        w_sub <- w_sub[(n_actual - expected_len + 1):n_actual]
    }
    return(w_sub)
}

for (i in 1:length(test_dates)) {
    train <- test %>%
        filter(date <= seq(as.Date(test_dates[i]), by = "-3 months", length = 2)[2]) %>%
        na_mean()

    y <- train[substr(train$date, 6, 7) %in% c("03", "06", "09", "12"), target_variable]
    n_quarters <- length(y)
    models <- list()
    weights <- list()

    # Train on monthly Cat3 variables
    for (col in cat3_features) {
        if (col %in% colnames(train)) {
            x <- train[, col]
            # Monthly: 3 months per quarter, lag 0-3
            models[[col]] <- tryCatch(
                midas_r(y ~ mls(x, 0:3, 3, nealmon),
                        start = list(x = c(1, -0.5))),
                error = function(e) NULL)
        }
    }
    # Process weekly variables separately
    for (wcol in weekly_vars) {
        w_vec <- get_weekly_series(data_weekly, wcol,
            train$date[1], tail(train$date, 1), n_quarters, 13)
        models[[wcol]] <- tryCatch(
            midas_r(y ~ mls(w_vec, 0:12, 13, nealmon),
                    start = list(w_vec = c(1, -0.1))),
            error = function(e) NULL)
    }

    # RMSE-based weights
    for (col in names(models)) {
        if (is.null(models[[col]])) next
        fitted <- models[[col]]$fitted.values
        actual <- y[2:length(y)]
        weights[[col]] <- sqrt(mean((fitted - actual)^2, na.rm = TRUE))
    }
    adj <- abs(unlist(weights) - max(unlist(weights), na.rm = TRUE))
    weights <- adj / sum(adj, na.rm = TRUE)

    # Forecast per vintage
    for (lag_name in names(vintage_offsets)) {
        vintage_date <- shift_month(test_dates[i], vintage_offsets[[lag_name]])
        lagged_data <- gen_vintage_data(metadata, test, test_dates[i], vintage_date)
        lagged_data <- data.frame(lagged_data)
        lagged_data[lagged_data$date == test_dates[i], target_variable] <- NA
        lagged_data <- na_mean(lagged_data)

        preds <- list()
        for (col in cat3_features) {
            if (is.null(models[[col]])) next
            if (col %in% colnames(lagged_data)) {
                x <- lagged_data[, col]
                p <- tryCatch(forecast(models[[col]],
                    newdata = list(x = x))$mean, error = function(e) NA)
                preds[[col]] <- p[length(p)]
            }
        }
        for (wcol in weekly_vars) {
            if (is.null(models[[wcol]])) next
            w_vec_f <- get_weekly_series(data_weekly, wcol,
                lagged_data$date[1], vintage_date,
                n_quarters, 13)
            p <- tryCatch(forecast(models[[wcol]],
                newdata = list(w_vec = w_vec_f))$mean,
                error = function(e) NA)
            preds[[wcol]] <- p[length(p)]
        }

        w_preds <- unlist(preds)
        w <- weights[names(preds)]
        pred_dict[pred_dict$date == test_dates[i], lag_name] <-
            weighted.mean(w_preds, w, na.rm = TRUE)
    }
    if (i %% 4 == 0) print(paste(i, "/", length(test_dates)))
}

actuals <- test %>%
    filter(date >= as.Date(test_start_date)) %>%
    filter(substr(date, 6, 7) %in% c("03", "06", "09", "12")) %>%
    select(!!target_variable) %>% pull()

dir.create("../turkey_predictions", showWarnings = FALSE)
for (lag_name in names(vintage_offsets)) {
    df_out <- data.frame(date = test_dates, actual = actuals,
                         prediction = pred_dict[, lag_name])
    write.csv(df_out, paste0("../turkey_predictions/midas_tr_", lag_name, ".csv"), row.names = FALSE)
    cat("Saved midas_tr_", lag_name, ".csv (", nrow(df_out), " rows)\n", sep="")
}

panels <- list(
    "Pre-crisis (2018-2019)" = c("2018-01-01", "2019-12-31"),
    "COVID      (2020-2021)" = c("2020-04-01", "2021-12-31"),
    "Post-COVID (2022-2025)" = c("2022-01-01", "2025-12-31"),
    "Full test  (2018-2025)" = c("2018-01-01", "2025-12-31")
)
rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
mae  <- function(a, p) mean(abs(a - p), na.rm = TRUE)
d <- test_dates
cat("MIDAS (Turkey) — Evaluation by Panel & Vintage\n")
for (pn in names(panels)) {
    rng <- panels[[pn]]; m <- d >= rng[1] & d <= rng[2]
    cat("---", pn, "---\n")
    for (lag_name in names(vintage_offsets)) {
        cat("  ", lag_name, "  RMSFE=",
            format(rmse(actuals[m], pred_dict[m, lag_name]), digits=6), "  MAE=",
            format(mae(actuals[m], pred_dict[m, lag_name]), digits=6), "  N=", sum(m), "\n")
    }
}
