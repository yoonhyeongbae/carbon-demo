@echo off
setlocal
cd /d "%~dp0"

set "TARGET=%~1"
if "%TARGET%"=="" (
  echo.
  echo EV Carbon Optimizer v0.5.7 UI/Map patch
  echo ---------------------------------------
  echo Drag the EV_Carbon_Optimizer_v0.5.6_DEV_READY folder onto this BAT,
  echo or run:
  echo   APPLY_V057_MAP_UI_PATCH.bat C:\EV_Carbon_Optimizer_v0.5.6_DEV_READY
  echo.
  set /p TARGET=EVCO v0.5.6 DEV_READY folder path: 
)

if exist "%TARGET%\.venv-web\Scripts\python.exe" (
  "%TARGET%\.venv-web\Scripts\python.exe" "%~dp0APPLY_V057_MAP_UI_PATCH.py" "%TARGET%"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 "%~dp0APPLY_V057_MAP_UI_PATCH.py" "%TARGET%"
  ) else (
    python "%~dp0APPLY_V057_MAP_UI_PATCH.py" "%TARGET%"
  )
)

if errorlevel 1 (
  echo.
  echo PATCH FAILED. Read the error above.
  pause
  exit /b 1
)

echo.
echo PATCH COMPLETE.
echo Now run START_DEV.bat inside the patched EVCO folder.
pause
