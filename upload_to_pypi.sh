#!/bin/bash
# upload_to_pypi.sh - Script to upload only source distribution to PyPI

set -e

echo "Building source distribution only for PyPI..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build only source distribution (no wheels)
python -m build --sdist

echo "Built distributions:"
ls -la dist/

# Upload to PyPI (only .tar.gz files)
echo "Uploading source distribution to PyPI..."
twine upload dist/*.tar.gz

echo "✅ Source distribution uploaded to PyPI successfully!"
echo "Users will now build from source when installing via 'pip install torchCompactRadius'"
