@echo off
setlocal
cd /d %~dp0\..

if not exist .venv (
  echo Environment not found. Please run scripts\bootstrap_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto fail
v1xdata update
if errorlevel 1 goto fail
v1xdata scan
if errorlevel 1 goto fail
v1xdata doctor
if errorlevel 1 goto fail

pause
exit /b 0

:fail
echo.
echo [ERROR] Daily V1.X pipeline stopped at the failed step above.
echo No stale scan should be published for this run.
pause
exit /b 1
