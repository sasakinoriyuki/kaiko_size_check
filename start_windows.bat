@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup_needed
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
exit /b
:setup_needed
echo Run setup_windows.bat first.
pause
exit /b 1
