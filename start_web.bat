@echo off
echo ==============================================
echo CosyVoice3 Web Interface Launcher
echo ==============================================
echo.

cd /d "%~dp0"

echo Checking dependencies...
python_embeded\python.exe -m pip install gradio soundfile numpy --no-warn-script-location

echo.
echo Starting web server...
echo.
echo Server will be available at http://127.0.0.1:7860
echo Press Ctrl+C to stop
echo.

python_embeded\python.exe web_app.py

pause