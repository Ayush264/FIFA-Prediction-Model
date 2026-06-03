@echo off
REM Batch helper to run MLflow UI (Windows)
cd /d %~dp0
call venv\Scripts\activate.bat
set MLFLOW_ALLOW_FILE_STORE=true
if not defined MLFLOW_TRACKING_URI (
  set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
)
echo Starting MLflow UI on http://127.0.0.1:5000 using backend %MLFLOW_TRACKING_URI%
python -m mlflow ui --backend-store-uri %MLFLOW_TRACKING_URI% --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
pause
