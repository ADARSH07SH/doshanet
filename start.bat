@echo off
echo ============================================
echo  DoshaNet - Starting Local Environment
echo ============================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8

cd /d %~dp0

echo [1/3] Checking dataset...
if not exist "dataset\data.json" (
    echo Generating dataset...
    python dataset\generate_dataset.py
) else (
    echo Dataset found - skipping generation.
)

echo.
echo [2/3] Checking trained model...
if not exist "model\saved\dosha_model.pt" (
    echo Training model - this may take ~30 seconds...
    python run_train.py
) else (
    echo Trained model found - skipping training.
)

echo.
echo [3/3] Starting FastAPI backend on http://localhost:8000 ...
echo.
echo ==============================================
echo  Frontend: open frontend\index.html in browser
echo  API Docs: http://localhost:8000/docs
echo  Health:   http://localhost:8000/health
echo ==============================================
echo.
echo Press Ctrl+C to stop the server.
echo.

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
