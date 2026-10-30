@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto environment_ready
py -3.12 -m venv .venv
if errorlevel 1 goto python_needed
:environment_ready
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch; print(torch.__version__)"
if not errorlevel 1 goto ready
".venv\Scripts\python.exe" -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
:ready
echo Setup complete. Run start_windows.bat. No pretrained model is downloaded.
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
