@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" prepare_training_images.py
if errorlevel 1 pause
exit /b
:missing
echo The existing Python environment was not found.
echo Copy this BAT and prepare_training_images.py beside the app.py you already use.
pause
exit /b 1
