#!/bin/bash

# Setup script for Sentence JEPA

set -e  # Exit on error

echo "=========================================="
echo "Setting up Sentence JEPA environment"
echo "=========================================="

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Optional: Install spaCy model
echo ""
read -p "Install spaCy English model (recommended for better sentence splitting)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing spaCy model..."
    python -m spacy download en_core_web_sm
    echo "✓ spaCy model installed"
else
    echo "Skipping spaCy model (will use regex fallback)"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "    source .venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "    python test_pipeline.py"
echo ""
echo "To start training, run:"
echo "    python scripts/train.py --create-sample-data"
echo "    python scripts/train.py"
echo ""
