@echo off
echo ==============================================
echo CosyVoice3 Server
echo ==============================================
echo.

cd /d "%~dp0"

echo.
echo Starting server...
echo.
echo Server will be available at http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.

python_embeded\python.exe api_server.py

pause