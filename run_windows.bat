@echo off
setlocal
cd /d "%~dp0"
echo.
echo  Neural Playground - Local Learning Lab
echo  Keep this window open while using the app.
echo  Press Ctrl+C to stop the local server.
echo.
start "Neural Playground" http://127.0.0.1:8080
where py >nul 2>nul
if %errorlevel%==0 (
  py -m http.server 8080
) else (
  python -m http.server 8080
)
if errorlevel 1 (
  echo.
  echo Could not start the server. Check Python is installed and port 8080 is free.
  pause
)
endlocal
