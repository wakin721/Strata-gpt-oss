@echo off
setlocal
chcp 65001 >nul
python -u "%~dp0api_server.py" %*
if errorlevel 1 pause
