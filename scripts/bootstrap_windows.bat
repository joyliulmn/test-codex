@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 goto fail

echo [0/6] Checking Python 3.11...
where py >nul 2>nul
if errorlevel 1 goto no_py

py -3.11 --version
if errorlevel 1 goto no_py311

if exist .venv goto venv_ready
echo [1/6] Creating Python virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 goto fail
goto venv_ready

:venv_ready
echo [1/6] Python virtual environment is ready.
call .venv\Scripts\activate.bat
if errorlevel 1 goto fail

echo [2/6] Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 goto fail

echo [3/6] Installing V1.X dependencies...
python -m pip install -e .
if errorlevel 1 goto fail

echo [4/6] Downloading current A-share market snapshot...
v1xdata update
if errorlevel 1 goto fail

echo [5/6] Backfilling historical daily bars from 2020. This can take a long time and is resumable...
v1xdata bootstrap --start 20200101 --resume
if errorlevel 1 goto fail

echo [6/6] Running V1.X scan and database checks...
v1xdata scan
if errorlevel 1 goto fail
v1xdata doctor
if errorlevel 1 goto fail

echo.
echo [SUCCESS] V1.X data environment initialized successfully.
pause
exit /b 0

:no_py
echo.
echo [ERROR] Python launcher py was not found.
echo Install Python 3.11 64-bit first and run this file again.
pause
exit /b 1

:no_py311
echo.
echo [ERROR] Python 3.11 was not found by the Python launcher.
echo Install Python 3.11 64-bit first and run this file again.
pause
exit /b 1

:fail
echo.
echo [ERROR] Setup stopped at the failed step above.
echo Take a screenshot and send it for troubleshooting.
pause
exit /b 1
