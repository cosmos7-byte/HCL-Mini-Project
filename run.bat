@echo off
title MediPredict AI - Early Disease Prediction System
echo ======================================================================
echo          MEDIPREDICT AI - EARLY DISEASE PREDICTION SYSTEM
echo               Machine Learning Clinical Risk Predictor
echo ======================================================================
echo.

set "PY_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"

if not exist "%PY_CMD%" (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=python"
    ) else (
        echo [ERROR] Python not found. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo [1/3] Checking dataset files...
if not exist "data\early_disease_screening.csv" (
    echo Generating clinical datasets...
    "%PY_CMD%" "data\generate_datasets.py"
) else (
    echo Datasets verified.
)

echo [2/3] Checking trained models...
if not exist "models\saved\multi_disease_model.joblib" (
    echo Training machine learning models...
    "%PY_CMD%" "models\train_models.py"
) else (
    echo Trained models verified.
)

echo.
echo [3/3] Starting MediPredict AI Web Server on http://127.0.0.1:8000 ...
echo Opening your web browser...
start http://127.0.0.1:8000

"%PY_CMD%" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
