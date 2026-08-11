@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo EV Carbon Optimizer v0.5.8 Update
echo - Stage/material result-map filters
echo - Natural Earth local vector basemap
echo - Map performance improvements
echo ============================================================
echo.

set "PYEXE="
if exist ".venv-web\Scripts\python.exe" set "PYEXE=.venv-web\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"

if defined PYEXE goto run_patch

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYEXE=py -3"
  goto run_patch
)

where python >nul 2>nul
if %errorlevel%==0 (
  set "PYEXE=python"
  goto run_patch
)

echo [ERROR] Python을 찾을 수 없습니다.
echo 기존 EV_Carbon_Optimizer_v0.5.6_DEV_READY 폴더의 START_DEV.bat을
echo 먼저 한 번 실행하여 portable Python/venv를 준비한 뒤 다시 실행하세요.
pause
exit /b 1

:run_patch
echo Using Python: %PYEXE%
%PYEXE% APPLY_V058_NATURAL_EARTH_PATCH.py
if errorlevel 1 (
  echo.
  echo [ERROR] v0.5.8 업데이트 적용에 실패했습니다.
  echo 위 오류 메시지를 확인하세요.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo v0.5.8 업데이트 완료
echo 이제 START_DEV.bat을 더블클릭하여 실행하세요.
echo ============================================================
pause
exit /b 0
