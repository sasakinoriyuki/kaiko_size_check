@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto environment_ready
py -3.12 -m venv .venv
if errorlevel 1 goto python_needed
:environment_ready
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Optional NVIDIA GPU setup. Requires a compatible NVIDIA driver for CUDA 12.8.
".venv\Scripts\python.exe" -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_ai_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch; print('GPU available:', torch.cuda.is_available())"
echo AI setup complete. Run start_windows.bat.
pause
exit /b 0
:python_needed
echo Could not create the environment. Install Python 3.12 with the Python launcher first.
pause
exit /b 1
:failed
echo GPU setup failed. Share the error message, or use setup_ai_windows.bat for CPU.
pause
exit /b 1
