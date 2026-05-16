$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main'
$env:PATH = 'C:\Program Files\R\R-4.6.0\bin;' + $env:PATH
$notebooks = @(
  'model_rf.ipynb',
  'model_xgboost.ipynb',
  'model_gb.ipynb',
  'model_dt.ipynb',
  'model_mlp.ipynb',
  'model_lstm.ipynb',
  'model_deepvar.ipynb',
  'model_bvar.ipynb',
  'model_midas.ipynb',
  'model_midasml.ipynb',
  'model_dfm.ipynb'
)
$status = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\us_notebook_run_20260514_rf_onward' 'status.csv'
$summary = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\us_notebook_run_20260514_rf_onward' 'summary.txt'
'notebook,status,start,end,exit_code' | Set-Content -Path $status -Encoding UTF8
foreach ($nb in $notebooks) {
  $start = Get-Date -Format o
  "START $nb $start" | Add-Content -Path $summary -Encoding UTF8
  $log = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\us_notebook_run_20260514_rf_onward' ($nb + '.log')
  & jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 (Join-Path 'model_notebooks' $nb) *> $log
  $code = $LASTEXITCODE
  $end = Get-Date -Format o
  if ($code -eq 0) {
    "$nb,ok,$start,$end,$code" | Add-Content -Path $status -Encoding UTF8
    "OK $nb $end" | Add-Content -Path $summary -Encoding UTF8
  } else {
    "$nb,failed,$start,$end,$code" | Add-Content -Path $status -Encoding UTF8
    "FAILED $nb $end exit=$code" | Add-Content -Path $summary -Encoding UTF8
    exit $code
  }
}
"ALL_OK $(Get-Date -Format o)" | Add-Content -Path $summary -Encoding UTF8
