#!/bin/bash

# Build and Upload Script for torchCompactRadius
# This script builds source distributions and uploads to PyPI

set -e  # Exit on any error

echo "=========================================="
echo "Building and Uploading torchCompactRadius"
echo "=========================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check if we have the required tools
echo "Checking build tools..."
if ! command -v python -m build &> /dev/null; then
    echo "Installing build tools..."
    pip install build twine
fi

# Build source distribution only (no wheels for extensions)
echo "Building source distribution..."
python -m build --sdist

# Check the built package
echo "Checking built package..."
python -m twine check dist/*

# List what was built
echo "Built packages:"
ls -la dist/

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="

# Ask user if they want to upload
read -p "Do you want to upload to PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to PyPI..."
    
    # Upload to PyPI (will prompt for credentials)
    python -m twine upload dist/*
    
    echo "Upload completed!"
else
    echo "Upload skipped. To upload later, run:"
    echo "python -m twine upload dist/*"
fi

echo "=========================================="
echo "Process completed!"
echo "=========================================="
