@echo off
echo.
echo  ======================================
echo   Titanic Survival Explorer
echo  ======================================
echo.

:: Check if pip packages are installed
python -m streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo  Starting app at http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
streamlit run app.py
pause
