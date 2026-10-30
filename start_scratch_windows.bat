@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup_needed
".venv\Scripts\python.exe" start_scratch_app.py
if errorlevel 1 pause
exit /b
:setup_needed
echo Copy this update into the folder containing your existing app.py and .venv.
echo For a new installation, run setup_scratch_windows.bat first.
pause
exit /b 1
