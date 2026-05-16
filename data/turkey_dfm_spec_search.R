.libPaths(c("C:/Users/asus/R/library", "C:/Users/asus/AppData/Local/R/win-library/4.6", .libPaths()))
options(warn = -1)
library(tidyverse)
library(nowcastDFM)

source("data/helpers.R")

metadata <- read_csv("turkey_data/meta_data_tr.csv", show_col_types = FALSE)
data <- read_csv("turkey_data/data_tf_monthly_tr.csv", show_col_types = FALSE) %>%
  dplyr::arrange(date)

target_variable <- "ngdprsaxdctrq"
train_start_date <- "1995-01-01"
validation_start_date <- "2012-03-01"
validation_end_date <- "2017-12-01"
test_start_date <- "2018-03-01"
test_end_date <- "2025-12-01"
vintage_offsets <- c(m1 = -2, m2 = -1, m3 = 0, post1 = 1, post2 = 2)
selection_vintages <- c("m1", "m2", "m3")

cat2_features <- c("ipi_sa", "usd_try_avg", "cpi_sa", "fin_acc")
selected10_features <- c(
  "altin_rezerv_var", "auto_prod", "cpi_sa", "empl_num", "ipi_sa",
  "maden_ciro_endeksi_sa", "reer", "total_prod", "unemp_rate", "usd_try_avg"
)
cat3_features <- c(
  "altin_rezerv_var", "auto_prod", "bist100", "consu_i", "cpi_sa",
  "deposit_i", "doviz_rezerv_var", "emp_rate", "empl_num", "fin_acc",
  "ipi_sa", "m3", "maden_ciro_endeksi_sa", "ppi", "reer",
  "resmi_rezerv_var", "tax", "total_prod", "tourist", "unemp_num",
  "unemp_rate", "usd_try_avg"
)

candidate_specs <- list(
  cat2 = cat2_features,
  selected10 = selected10_features,
  cat3 = cat3_features
)

patch_nowcast_dfm <- function() {
  local({
    fn <- nowcastDFM:::init_conds
    b <- body(fn)
    replace_cov <- function(node) {
      if (is.call(node)) {
        txt <- deparse(node)
        if (identical(txt, "cov(res[, idx_iM])")) {
          return(quote(cov(as.matrix(res[, idx_iM, drop = FALSE]))))
        }
        return(as.call(lapply(node, replace_cov)))
      }
      node
    }
    body(fn) <- replace_cov(b)
    assignInNamespace("init_conds", fn, ns = "nowcastDFM")
  })

  local({
    fn <- nowcastDFM::dfm
    b <- body(fn)
    replace_A <- function(node) {
      if (is.call(node)) {
        txt <- deparse(node)[1]
        if (grepl("A <- em_output\\$A_new", txt)) {
          return(quote({
            A <- em_output$A_new
            diag_A <- diag(A)
            diag_A <- pmin(pmax(diag_A, -0.95), 0.95)
            diag(A) <- diag_A
          }))
        }
        return(as.call(lapply(node, replace_A)))
      }
      node
    }
    body(fn) <- replace_A(b)
    assignInNamespace("dfm", fn, ns = "nowcastDFM")
  })
}

rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
mae <- function(a, p) mean(abs(a - p), na.rm = TRUE)

prepare_data <- function(end_date) {
  out <- data %>%
    dplyr::filter(date >= as.Date(train_start_date), date <= as.Date(end_date)) %>%
    data.frame()
  for (col in colnames(out)) {
    if (is.numeric(out[, col]) && sum(is.infinite(out[, col])) > 0) {
      out[is.infinite(out[, col]), col] <- NA
    }
  }
  out
}

make_dfm_objects <- function(features, train_end_date, data_end_date, max_iter = 500) {
  patch_nowcast_dfm()
  df <- prepare_data(data_end_date)
  col_names_dfm <- colnames(df)[2:ncol(df)]
  col_names_dfm <- col_names_dfm[col_names_dfm %in% c(target_variable, features)]
  col_names_dfm <- c(col_names_dfm[col_names_dfm != target_variable], target_variable)

  blocks <- metadata %>%
    dplyr::filter(series %in% col_names_dfm) %>%
    dplyr::filter(series %in% colnames(df))
  blocks <- blocks[match(col_names_dfm, blocks$series), ]
  blocks <- blocks %>%
    dplyr::select(starts_with("block_")) %>%
    select_if(~ sum(.) > 0) %>%
    data.frame()

  train_cols <- c("date", col_names_dfm)
  train <- df %>%
    dplyr::filter(date <= as.Date(train_end_date)) %>%
    dplyr::filter(date >= as.Date(train_start_date)) %>%
    data.frame()
  train <- train[, train_cols]
  train$date <- as.character(train$date)

  output_dfm <- dfm(data = train, blocks = blocks, max_iter = max_iter)
  list(data = df, cols = train_cols, blocks = blocks, output = output_dfm)
}

predict_spec <- function(objects, eval_dates) {
  pred_dict <- data.frame(date = eval_dates)
  for (lag_name in names(vintage_offsets)) pred_dict[, lag_name] <- NA

  for (i in seq_along(eval_dates)) {
    for (lag_name in names(vintage_offsets)) {
      vintage_date <- shift_month(eval_dates[i], vintage_offsets[[lag_name]])
      lagged_data <- gen_vintage_data(metadata, objects$data, eval_dates[i], vintage_date)
      lagged_data <- data.frame(lagged_data)
      lagged_data <- lagged_data[, objects$cols]
      lagged_data[lagged_data$date == eval_dates[i], target_variable] <- NA

      prediction <- tryCatch({
        predict_dfm(lagged_data, objects$output) %>%
          dplyr::filter(date == eval_dates[i]) %>%
          dplyr::select(!!target_variable) %>%
          pull()
      }, error = function(e) {
        cat("ERR", as.character(eval_dates[i]), lag_name, conditionMessage(e), "\n")
        NA
      })
      pred_dict[pred_dict$date == eval_dates[i], lag_name] <- prediction
    }
  }
  pred_dict
}

score_predictions <- function(pred_dict, df, label) {
  actuals <- df %>%
    dplyr::filter(date %in% as.Date(pred_dict$date)) %>%
    dplyr::select(!!target_variable) %>%
    pull()

  rows <- list()
  for (lag_name in names(vintage_offsets)) {
    p <- pred_dict[, lag_name]
    rows[[lag_name]] <- data.frame(
      sample = label,
      vintage = lag_name,
      n_obs = length(actuals),
      RMSFE = rmse(actuals, p),
      MAE = mae(actuals, p)
    )
  }
  bind_rows(rows)
}

out_dir <- "archive/logs/turkey_dfm_validation_selection"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

validation_dates <- seq(as.Date(validation_start_date), as.Date(validation_end_date), by = "3 months")
validation_results <- list()

for (spec_name in names(candidate_specs)) {
  cat("VALIDATION SPEC", spec_name, "\n")
  objects <- make_dfm_objects(
    features = candidate_specs[[spec_name]],
    train_end_date = "2011-12-31",
    data_end_date = validation_end_date,
    max_iter = 500
  )
  preds <- predict_spec(objects, validation_dates)
  scores <- score_predictions(preds, objects$data, "validation")
  scores$spec <- spec_name
  scores$selection_RMSFE <- mean(scores$RMSFE[scores$vintage %in% selection_vintages], na.rm = TRUE)
  validation_results[[spec_name]] <- scores
  write.csv(preds, file.path(out_dir, paste0("validation_predictions_", spec_name, ".csv")), row.names = FALSE)
}

validation_summary <- bind_rows(validation_results) %>%
  dplyr::select(spec, everything()) %>%
  dplyr::arrange(selection_RMSFE, vintage)
write.csv(validation_summary, file.path(out_dir, "validation_summary.csv"), row.names = FALSE)
print(validation_summary)

selection_table <- validation_summary %>%
  dplyr::filter(vintage %in% selection_vintages) %>%
  group_by(spec) %>%
  summarise(selection_RMSFE = mean(RMSFE, na.rm = TRUE), .groups = "drop") %>%
  dplyr::arrange(selection_RMSFE)
selected_spec <- selection_table$spec[1]
cat("SELECTED_DFM_SPEC", selected_spec, "\n")
write.csv(selection_table, file.path(out_dir, "selection_table.csv"), row.names = FALSE)

cat("FINAL TEST SPEC", selected_spec, "\n")
test_dates <- seq(as.Date(test_start_date), as.Date(test_end_date), by = "3 months")
final_objects <- make_dfm_objects(
  features = candidate_specs[[selected_spec]],
  train_end_date = "2017-12-31",
  data_end_date = test_end_date,
  max_iter = 500
)
test_preds <- predict_spec(final_objects, test_dates)
test_scores <- score_predictions(test_preds, final_objects$data, "test")
test_scores$spec <- selected_spec
write.csv(test_scores, file.path(out_dir, "test_summary_selected.csv"), row.names = FALSE)
write.csv(test_preds, file.path(out_dir, "test_predictions_selected.csv"), row.names = FALSE)

actuals <- final_objects$data %>%
  dplyr::filter(date %in% as.Date(test_dates)) %>%
  dplyr::select(!!target_variable) %>%
  pull()
dir.create("turkey_predictions", showWarnings = FALSE)
for (lag_name in names(vintage_offsets)) {
  write.csv(
    data.frame(date = test_dates, actual = actuals, prediction = test_preds[, lag_name]),
    paste0("turkey_predictions/dfm_tr_", lag_name, ".csv"),
    row.names = FALSE
  )
}
print(test_scores)
