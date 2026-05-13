setwd("C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/model_notebooks")
.libPaths(c('C:/Users/asus/R/library', .libPaths()))
cat("nowcastDFM:", as.character(packageVersion("nowcastDFM")), "\n")

library(readr)
data_raw <- read_csv("../data/data_tf_monthly.csv", show_col_types=FALSE)

# Check COVID dummies
covid_cols <- c("covid_2020q2", "covid_2020q3", "covid_2020q4")
for (col in covid_cols) {
  x <- data_raw[[col]]
  non_na <- x[!is.na(x)]
  cat(col, "- class:", class(x), "  non-NA count:", length(non_na), "  values:", unique(non_na), "\n")
}

# Check the 3 logical columns
logical_cols <- c("dtwexbgs_monthly_avg", "pcec96", "ppifis")
for (col in logical_cols) {
  x <- data_raw[[col]]
  cat(col, "- class:", class(x), "  non-NA:", sum(!is.na(x)), "\n")
}

# Check what cat3 features would be all-NA in 1959-2007
cat3_features <- tolower(c(
  "a014re1q156nbea","acogno","ahetpix","amdmuox","andenox","awotman",
  "busloans","ce16ov","ces1021000001","ces2000000008","ces9091000001",
  "ces9092000001","clf16ov","compapff","cusr0000sas","ddurrg3m086sbea",
  "dhlcrg3q086sbea","difsrg3q086sbea","dodgrg3q086sbea","dongrg3q086sbea",
  "dspic96","expgsc1","fpix","gcec1","gpdic1","houstne","housts",
  "hwiuratio","hwiuratiox","invest","ipdcongd","liabpix","lns13023705",
  "m2sl","manemp","mortg10yrx","nonrevsl","ophpbs","outbs","outnfb",
  "permitw","realln","slcex","spcs10rsa","tlbsnncbbdix","uemp15t26",
  "uemp27ov","uemplt5","ulcbs","ulcnfb","unrate","usgovt","usserv",
  "covid_2020q2","covid_2020q3","covid_2020q4"
))

train_start_date <- "1959-01-01"
train <- data_raw %>% dplyr::filter(date >= as.Date(train_start_date), date <= as.Date("2007-12-31"))
cat("\nCat3 features with all-NA in 1959-2007 training period:\n")
for (col in cat3_features) {
  if (col %in% colnames(train)) {
    x <- train[[col]]
    if (all(is.na(x))) cat(" ", col, "\n")
  }
}
