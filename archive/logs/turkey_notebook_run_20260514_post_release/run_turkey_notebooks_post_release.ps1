$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main'
$env:PATH = 'C:\Program Files\R\R-4.6.0\bin;' + $env:PATH
$notebooks = @(
  'model_arma_tr.ipynb',
  'model_ols_tr.ipynb',
  'model_var_tr.ipynb',
  'model_lasso_tr.ipynb',
  'model_ridge_tr.ipynb',
  'model_elasticnet_tr.ipynb',
  'model_rf_tr.ipynb',
  'model_xgboost_tr.ipynb',
  'model_gb_tr.ipynb',
  'model_dt_tr.ipynb',
  'model_mlp_tr.ipynb',
  'model_lstm_tr.ipynb',
  'model_deepvar_tr.ipynb',
  'model_bvar_tr.ipynb',
  'model_midas_tr.ipynb',
  'model_midasml_tr.ipynb',
  'model_dfm_tr.ipynb'
)
$status = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\turkey_notebook_run_20260514_post_release' 'status.csv'
$summary = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\turkey_notebook_run_20260514_post_release' 'summary.txt'
'notebook,status,start,end,exit_code' | Set-Content -Path $status -Encoding UTF8
foreach ($nb in $notebooks) {
  $start = Get-Date -Format o
  "START $nb $start" | Add-Content -Path $summary -Encoding UTF8
  $log = Join-Path 'C:\Users\asus\Desktop\nowcasting_benchmark-main\nowcasting_benchmark-main\logs\turkey_notebook_run_20260514_post_release' ($nb + '.log')
  & jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 (Join-Path 'turkey_model_notebooks' $nb) *> $log
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
