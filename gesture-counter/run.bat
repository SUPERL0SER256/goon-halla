@echo off
title Gesture Counter
color 0A
echo ================================================
echo          GESTURE COUNTER LAUNCHER
echo ================================================
echo.
echo [1/3] Installing required packages...
pip install -r requirements.txt >nul 2>&1
echo       Done!
echo.
echo [2/3] Starting server...
start /B python app.py
echo       Server starting...
echo.
echo [3/3] Opening browser...
timeout /t 3 /nobreak >nul
start http://localhost:5000
echo       Browser opened!
echo.
echo ================================================
echo   Application is running!
echo   Close this window to stop the server.
echo ================================================
echo.
pause