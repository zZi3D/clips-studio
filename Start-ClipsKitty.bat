@echo off
title Clips Kitty Launcher
echo ===================================================
echo           Starting Clips Kitty (AI Studio)
echo ===================================================
echo.

cd /d "C:\Users\Zahied\Kitty"

echo [1/2] Starting Python FastAPI Backend on port 8765...
start "Clips Kitty - Backend" cmd /k "title Clips Kitty - Backend && cd /d C:\Users\Zahied\Kitty && .\.venv\Scripts\activate.bat && python main.py serve"

echo Waiting for backend to initialize...
ping 127.0.0.1 -n 3 >nul

echo [2/2] Starting Electron Desktop UI...
start "Clips Kitty - Desktop UI" cmd /k "title Clips Kitty - Desktop UI && cd /d C:\Users\Zahied\Kitty\ui && npm run dev"

echo.
echo ===================================================
echo Clips Kitty is starting up!
echo - Desktop App (Electron): Opent automatisch
echo - Web Dashboard (Browser): http://localhost:5173/
echo - Backend API Docs:       http://127.0.0.1:8765/docs
echo ===================================================
