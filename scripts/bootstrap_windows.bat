@echo off
setlocal
cd /d %~dp0\..

if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .

v1xdata update
v1xdata bootstrap --start 20200101 --resume
v1xdata scan
v1xdata doctor

pause
