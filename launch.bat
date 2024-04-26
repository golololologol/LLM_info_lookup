@echo off
CALL :checkPython

set "VENV_DIR=venv"
set "REQUIREMENTS=requirements.txt"

IF NOT EXIST "%VENV_DIR%" (
    echo Making the virtual environment, hold your horses...
    python -m venv %VENV_DIR%
    IF ERRORLEVEL 1 (
        echo Could not create virtual environment. Make sure Python is installed correctly.
	echo Recommended version of Python: 3.10.11
        goto :error
    )
)

echo Activating the virtual environment...
CALL %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment. Did you move the damn files around? Delete the venv folder and try again.
    goto :error
)

echo Checking if you already got what you need...
pip freeze > venv\current_requirements.txt
fc %REQUIREMENTS% venv\current_requirements.txt > venv\difference.txt
IF ERRORLEVEL 1 (
    echo Installing the damn dependencies from requirements.txt...
    pip install -r %REQUIREMENTS%
    IF ERRORLEVEL 1 (
        echo Something's fucked up with pip. Are you sure it's installed?
        goto :error
    )
) ELSE (
    echo All dependencies are already installed, lucky you.
)

echo Running the script...
python info_lookup_gradio.py
IF ERRORLEVEL 1 (
    echo The script got an issue, might fix later.
    goto :error
)

echo Done with the dirty work. Go get a drink or something.
goto :end

:checkPython
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python is not recognized as an internal or external command, operable program or batch file. Fix it, genius.
    goto :error
)
goto :eof

:error
echo Some error happened. Womp womp.
pause
exit /b

:end
pause
exit /b
