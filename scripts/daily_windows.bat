@echo off
setlocal
cd /d %~dp0\..

if not exist .venv (
  echo Environment not found. Please run scripts\bootstrap_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
v1xdata update
v1xdata scan
v1xdata doctor

pause
