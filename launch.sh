#!/bin/bash

checkPython() {
    python --version &>/dev/null
    if [ $? -ne 0 ]; then
        echo "Python is not installed or it's not in your PATH, Einstein. Fix it!"
        exit 1
    fi
}

VENV_DIR="venv"
REQUIREMENTS="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Making the virtual environment, hold your horses..."
    python -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Could not create virtual environment. Make sure Python is installed correctly."
        echo "Recommended version of Python: 3.10.11"
        exit 1
    fi
fi

echo "Activating the virtual environment..."
source $VENV_DIR/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Did you move the damn files around? Delete the venv folder and try again."
    exit 1
fi

echo "Checking each damn dependency manually..."
while read requirement; do
    pip show $requirement &>/dev/null
    if [ $? -ne 0 ]; then
        echo "Missing $requirement, installing now..."
        pip install $requirement
        if [ $? -ne 0 ]; then
            echo "Failed to install $requirement. Your setup is probably cursed. Throw your computer out of the window."
            echo "Or deleting the venv folder might also suffice."
            exit 1
        fi
    else
        echo "$requirement is already installed..."
    fi
done < $REQUIREMENTS

echo "All dependencies are installed and verified. Aren't you lucky?"

echo "Running the script..."
python info_lookup_gradio.py
if [ $? -ne 0 ]; then
    echo "The script got an issue, might fix later."
    exit 1
fi

echo "Done with the dirty work. Go get a drink or something."
