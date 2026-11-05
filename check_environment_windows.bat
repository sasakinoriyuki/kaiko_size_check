@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto app_environment
py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 goto default_python
py -3.12 -u check_environment.py
goto done
:default_python
py -u check_environment.py
goto done
:app_environment
".venv\Scripts\python.exe" -u check_environment.py
:done
echo.
echo Report folder: data\diagnostics
echo This check does not open a camera or download files.
pause
