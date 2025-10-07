#!/bin/bash

# GAN Stress Testing Environment Setup Script
# This script helps set up and activate the virtual environment

set -e  # Exit on any error

PROJECT_DIR="/Users/jiapivialiu/Documents/Study/Github_repos/jiapivialiu-GAN-hub/GAN-for-stress-testing"
VENV_DIR="$PROJECT_DIR/gan-stress-env"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

echo "GAN Stress Testing Environment Setup"
echo "========================================"

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv gan-stress-env
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source gan-stress-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install/update requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo " requirements.txt not found, installing core packages..."
    pip install torch torchvision numpy pandas scikit-learn scipy matplotlib seaborn PyYAML jupyter notebook plotly ipywidgets tqdm
fi

# Verify installation
echo "Verifying installation..."
python verify_environment.py

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "cd $PROJECT_DIR && source gan-stress-env/bin/activate"
echo ""
echo "To start the demo notebook:"
echo "jupyter notebook notebooks/demo_stress_testing.ipynb"
