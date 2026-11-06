@echo off
title Overtake Safety - webcam + Arduino LEDs
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please double-click 1_SETUP.bat first.
    pause
    exit /b 1
)
if not exist "models\best.pt" (
    echo models\best.pt is missing. Make sure the whole "models" folder was copied.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Starting webcam with Arduino. Plug the Arduino in BEFORE running this.
echo Close the Arduino IDE Serial Monitor if it is open.
echo Click the video window and press Q to stop.
echo.
python realtime.py --port auto
echo.
echo Finished.
pause
