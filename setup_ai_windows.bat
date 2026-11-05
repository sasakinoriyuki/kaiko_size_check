@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto environment_ready
py -3.12 -m venv .venv
if errorlevel 1 goto python_needed
:environment_ready
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installing optional local AI support for CPU. This may take several minutes.
".venv\Scripts\python.exe" -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_ai_model.py
if errorlevel 1 goto failed
echo AI setup complete. Run start_windows.bat.
pause
exit /b 0
:python_needed
echo Could not create the environment. Install Python 3.12 with the Python launcher first.
pause
exit /b 1
:failed
echo AI setup failed. Please share the error message above.
pause
exit /b 1
