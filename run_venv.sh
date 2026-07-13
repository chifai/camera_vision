#!/bin/bash
# Script to set up and run Python scripts inside a virtual environment (venv)

VENV_DIR=".venv"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# 2. Activate the virtual environment
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# 3. Upgrade pip and install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing/checking dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        exit 1
    fi
fi

# 4. Execute passed commands, if any
if [ $# -gt 0 ]; then
    echo "Executing in venv: $@"
    exec "$@"
else
    echo "==========================================================="
    echo " Virtual Environment setup complete & activated."
    echo " Usage:"
    echo "   ./run_venv.sh <command>"
    echo ""
    echo " Examples:"
    echo "   ./run_venv.sh python3 detect_charuco.py raw/std_charuco_10mm_rot1.png --crop"
    echo "   ./run_venv.sh python3 undistort_rectify.py raw/std_charuco_10mm_rot1.png --rectify"
    echo "==========================================================="
fi
