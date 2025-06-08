#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

VENV_DIR="./venv"

# deactivate

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists in $VENV_DIR"
fi

echo "Activating virtual environment..."

source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To activate the environment, run:"
echo "source $VENV_DIR/bin/activate"
