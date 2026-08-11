@echo off
setlocal
cd /d "%~dp0\.."

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher ^"py^" was not found.
  echo Please install Python 3.11 64-bit first, then run this file again.
  echo Recommended: Python 3.11.9 Windows installer (64-bit).
  pause
  exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.11 was not found by the Python launcher.
  echo Please install Python 3.11 64-bit first, then run this file again.
  pause
  exit /b 1
)

if not exist .venv (
  echo [1/6] Creating Python virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto :fail
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto :fail

echo [2/6] Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/6] Installing V1.X dependencies...
python -m pip install -e .
if errorlevel 1 goto :fail

echo [4/6] Downloading current A-share market snapshot...
v1xdata update
if errorlevel 1 goto :fail

echo [5/6] Backfilling historical daily bars from 2020. This can take a long time and is resumable...
v1xdata bootstrap --start 20200101 --resume
if errorlevel 1 goto :fail

echo [6/6] Running V1.X scan and database checks...
v1xdata scan
if errorlevel 1 goto :fail
v1xdata doctor
if errorlevel 1 goto :fail

echo.
echo [SUCCESS] V1.X data environment initialized successfully.
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup stopped at the failed step above. Take a screenshot and send it for troubleshooting.
pause
exit /b 1
