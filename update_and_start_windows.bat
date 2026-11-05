@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import tkinter" >nul 2>&1
if not errorlevel 1 goto python312
py -c "import tkinter" >nul 2>&1
if not errorlevel 1 goto launcher_python
python -c "import tkinter" >nul 2>&1
if not errorlevel 1 goto path_python
echo Python with Tkinter was not found.
echo You can copy the update files into your existing app folder and run start_scratch_windows.bat.
pause
exit /b 1
:python312
py -3.12 update_and_start.py
set "KAIKO_UPDATE_EXIT=%errorlevel%"
goto finish
:launcher_python
py update_and_start.py
set "KAIKO_UPDATE_EXIT=%errorlevel%"
goto finish
:path_python
python update_and_start.py
set "KAIKO_UPDATE_EXIT=%errorlevel%"
:finish
if not "%KAIKO_UPDATE_EXIT%"=="0" pause
exit /b %KAIKO_UPDATE_EXIT%
