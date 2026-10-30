@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto environment_ready
py -3.12 -m venv .venv
if errorlevel 1 goto python_needed
:environment_ready
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
echo Setup complete. Run start_scratch_windows.bat. No pretrained model is downloaded.
pause
exit /b 0
:python_needed
echo Python 3.12 with the Python launcher is required.
pause
exit /b 1
:failed
echo Setup failed. Please share the error message above.
pause
exit /b 1
