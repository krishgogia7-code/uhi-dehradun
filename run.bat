@echo off
title UHI - Dehradun Dashboard
cd /d "%~dp0"

rem ---- find Python (python, or py if python is not on PATH) ----
set PY=python
where python >nul 2>nul || set PY=py

rem ---- decide whether the data/models need to be (re)built ----
set NEEDBUILD=0
if not exist "outputs\metrics.json" set NEEDBUILD=1
if "%NEEDBUILD%"=="0" powershell -NoProfile -Command "if (Get-ChildItem 'data\*.csv' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Item 'outputs\metrics.json' -ErrorAction SilentlyContinue).LastWriteTime }) { exit 1 } else { exit 0 }"
if "%NEEDBUILD%"=="0" if errorlevel 1 set NEEDBUILD=1

if "%NEEDBUILD%"=="1" (
  if not exist "data\*.csv" %PY% make_sample_data.py
  echo.
  echo Preparing data and training models - about a minute, please wait ...
  %PY% step1_prepare_data.py
  %PY% step2_train_models.py
)

echo.
echo ============================================================
echo   URBAN HEAT ISLAND - DEHRADUN dashboard is starting...
echo.
echo   Your browser will open at  http://localhost:8501
echo   KEEP THIS WINDOW OPEN while you use the dashboard.
echo   To stop:  close this window  (or press Ctrl+C here).
echo ============================================================
echo.

rem open the browser once the server is up
start "UHI browser opener" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:8501"

%PY% -m streamlit run app.py

pause
