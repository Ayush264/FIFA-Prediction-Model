# PowerShell helper to run MLflow UI with project's sqlite DB and artifacts
param(
    [int]$Port = 5000,
    [string]$BindHost = '127.0.0.1'
)
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)
if (-not (Test-Path -Path venv\Scripts\Activate.ps1)) {
    Write-Error "Activate script not found. Ensure virtualenv is created in ./venv"
    exit 1
}
. .\venv\Scripts\Activate.ps1
$env:MLFLOW_ALLOW_FILE_STORE = 'true'
if (-not $env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
}
Write-Host "Starting MLflow UI on http://$BindHost`:$Port using backend $env:MLFLOW_TRACKING_URI"
python -m mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI --default-artifact-root ./mlruns --host $BindHost --port $Port
