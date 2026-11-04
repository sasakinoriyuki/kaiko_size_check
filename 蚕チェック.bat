@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0_internal\bootstrap_windows.ps1"
set "KAIKO_EXIT_CODE=%ERRORLEVEL%"
if not "%KAIKO_EXIT_CODE%"=="0" pause
exit /b %KAIKO_EXIT_CODE%
