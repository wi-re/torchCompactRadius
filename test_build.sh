#!/bin/bash

# Test Build Script - validates package before upload
# This script tests the build process without uploading

set -e

echo "========================================"
echo "Testing torchCompactRadius Build Process"
echo "========================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ test_env/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Build the package
echo "Building source distribution..."
python -m build --sdist

# Check the package
echo "Checking package validity..."
python -m twine check dist/*

# Create test environment
echo "Creating test environment..."
python -m venv test_env
source test_env/bin/activate

# Install the built package
echo "Installing built package in test environment..."
pip install dist/torchcompactradius-*.tar.gz

# Test import
echo "Testing package import..."
python -c "
import torchCompactRadius
import torch
print('✓ Package imported successfully')
print('✓ Available functions:', len([x for x in dir(torchCompactRadius) if not x.startswith('_')]))
print('✓ CUDA extension available:', hasattr(torch.ops, 'torchCompactRadius_cuda'))
if hasattr(torch.ops, 'torchCompactRadius_cuda'):
    print('✓ CUDA operations:', len([x for x in dir(torch.ops.torchCompactRadius_cuda) if not x.startswith('_')]))
print('✓ All tests passed!')
"

# Cleanup
deactivate
rm -rf test_env/

echo "========================================"
echo "Build test completed successfully!"
echo "The package is ready for upload."
echo "========================================"
