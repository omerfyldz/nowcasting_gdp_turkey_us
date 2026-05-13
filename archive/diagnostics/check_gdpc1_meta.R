.libPaths(c('C:/Users/asus/R/library', .libPaths()))
library(readr)
m <- read_csv('../data/meta_data.csv', show_col_types=FALSE)
cat('gdpc1 in metadata:', 'gdpc1' %in% m$series, '\n')
if ('gdpc1' %in% m$series) {
  print(m[m$series == 'gdpc1', ])
}
cat('Total metadata rows:', nrow(m), '\n')
cat('First 5 rows:\n')
print(head(m, 5))
