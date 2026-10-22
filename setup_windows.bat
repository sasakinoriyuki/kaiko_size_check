@echo off
setlocal
cd /d "%~dp0"
echo Creating a Python 3.12 environment...
py -3.12 -m venv .venv
if errorlevel 1 goto python_error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error
echo Setup complete. Double-click start_windows.bat to start.
pause
exit /b 0
:python_error
echo Python 3.12 was not found. Install Python 3.12 from python.org with Tcl/Tk and the Python launcher.
pause
exit /b 1
:install_error
echo Installation failed. Check the internet connection and the error above.
pause
exit /b 1
