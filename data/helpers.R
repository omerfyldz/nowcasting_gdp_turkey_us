# helpers.R — single source of truth for all R model notebooks.
#
# Source this in every R notebook:
#   source("../data/helpers.R")
#
# Functions:
#   gen_lagged_data — ragged-edge publication-lag mask

library(tidyverse)

gen_lagged_data <- function(metadata, data, last_date, lag) {
  # Only go up to the last date
  lagged_data <- data %>%
    dplyr::filter(date <= last_date)

  for (col in colnames(lagged_data)[2:length(colnames(lagged_data))]) {
    pub_lag <- metadata %>%
      dplyr::filter(series == col) %>%
      select(months_lag) %>%
      pull()

    # Go back as far as needed for the pub_lag of the variable,
    # then + the lag (so -2 for 2 months back)
    if (length(pub_lag) == 0) next  # column not in metadata (e.g. target, COVID dummies)
    condition <- (nrow(lagged_data) - pub_lag + lag)
    # Only input NA if the lag is less than the latest row in the data
    if (condition <= nrow(lagged_data)) {
      lagged_data[condition:nrow(lagged_data), col] <- NA
    }
  }
  lagged_data <- lagged_data %>%
    dplyr::filter(!is.na(date))
  return(lagged_data)
}

shift_month <- function(date, offset) {
  # Base-R month shift used to avoid adding another package dependency.
  seq(as.Date(date), by = paste(offset, "months"), length.out = 2)[2]
}

gen_vintage_data <- function(metadata, data, target_date, vintage_date) {
  # Build a ragged-edge dataset for a target quarter while making the
  # information date explicit. Rows are kept through target_date so DFM/MIDAS
  # can predict the target quarter, but values released after vintage_date are
  # masked as unavailable. The cutoff mirrors gen_lagged_data(..., lag = 0):
  # months_lag = 0 still masks the current month.
  target_date <- as.Date(target_date)
  vintage_date <- as.Date(vintage_date)

  vintage_data <- data %>%
    dplyr::filter(date <= target_date)

  for (col in colnames(vintage_data)[2:length(colnames(vintage_data))]) {
    pub_lag <- metadata %>%
      dplyr::filter(series == col) %>%
      select(months_lag) %>%
      pull()

    if (length(pub_lag) == 0) next

    available_through <- shift_month(vintage_date, -(pub_lag + 1))
    vintage_data[vintage_data$date > available_through, col] <- NA
  }

  vintage_data <- vintage_data %>%
    dplyr::filter(!is.na(date))
  return(vintage_data)
}
