@echo off
REM Launch FastAPI server for Windows

echo.
echo ========================================
echo   Spam Classifier API Server
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting server...
echo.
echo Server will be available at:
echo   - Local:   http://localhost:8000
echo   - Network: http://0.0.0.0:8000
echo   - API Docs: http://localhost:8000/api/docs
echo.

python -m src.spam_classifier.server

pause
