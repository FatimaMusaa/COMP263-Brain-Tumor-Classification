@echo off
setlocal

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete.
echo You can now run run_app.bat to start the Streamlit app.
echo.
pause
