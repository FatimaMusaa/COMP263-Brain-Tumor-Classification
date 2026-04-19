@echo off
setlocal

cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate
streamlit run app.py
