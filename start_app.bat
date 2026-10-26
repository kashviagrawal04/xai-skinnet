@echo off
setlocal
cd /d "%~dp0"
title XAI-SkinNet launcher

if not exist "runs\xai_skinnet_v2_meta\best.pt" (
  echo.
  echo  [!] Model file not found: runs\xai_skinnet_v2_meta\best.pt
  echo      Download best.pt and temperature.json from your Kaggle notebook's Output tab
  echo      and put them in runs\xai_skinnet_v2_meta\  then run this again.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up Python environment, this takes a few minutes...
  python -m venv .venv || (echo Python 3.10+ is required & pause & exit /b 1)
  call .venv\Scripts\activate
  pip install -r requirements.txt
  pip install -r backend\requirements.txt
) else (
  call .venv\Scripts\activate
)

if not exist "frontend\node_modules" (
  echo First run: installing web app packages...
  pushd frontend & call npm install & popd
)

set XAI_CONFIG_PATH=configs/config.yaml
set XAI_CHECKPOINT_PATH=runs/xai_skinnet_v2_meta/best.pt

echo Starting model server...
start "XAI-SkinNet model server" cmd /k "call .venv\Scripts\activate && uvicorn backend.app:app --port 8000"
echo Starting web app...
start "XAI-SkinNet web app" cmd /k "cd frontend && npm run dev -- --open"
echo.
echo Wait ~30 seconds for the model to load, then drop an image on the page.
echo Close the two new windows to stop.
