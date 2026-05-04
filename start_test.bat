@echo off
echo ==============================================
echo CosyVoice3 Test client
echo ==============================================
echo.

cd /d "%~dp0"

echo.
echo Starting client...
echo.
echo Press Ctrl+C to stop
echo.

python_embeded\python.exe api_test.py

pause