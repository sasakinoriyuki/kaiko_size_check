@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import tkinter" >nul 2>&1
if errorlevel 1 goto default_python
py -3.12 update_and_start.py
if errorlevel 1 pause
exit /b
:default_python
py update_and_start.py
if errorlevel 1 pause
