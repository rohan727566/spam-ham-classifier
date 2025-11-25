@echo off
REM Train spam classifier model

echo.
echo ========================================
echo   Training Spam Classifier
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Starting training...
python -m src.spam_classifier.train

echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause
