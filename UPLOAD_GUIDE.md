# PyPI Upload Instructions for torchCompactRadius

## Prerequisites

1. **Install upload tools:**
   ```bash
   pip install build twine
   ```

2. **Create PyPI account:**
   - Go to https://pypi.org/account/register/
   - Verify your email address

3. **Create API token (recommended):**
   - Go to https://pypi.org/manage/account/
   - Scroll to "API tokens" section
   - Click "Add API token"
   - Give it a name (e.g., "torchCompactRadius")
   - Save the token (starts with `pypi-`)

## Quick Upload (Using Script)

Simply run the provided script:
```bash
./build_and_upload.sh
```

This will:
- Clean previous builds
- Build source distribution
- Check the package
- Ask if you want to upload to PyPI

## Manual Upload Process

### 1. Clean and Build
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build source distribution
python -m build --sdist
```

### 2. Check Package
```bash
# Verify the package is valid
python -m twine check dist/*
```

### 3. Upload to PyPI

**Option A: Using API token (recommended)**
```bash
python -m twine upload dist/* --username __token__ --password pypi-YOUR_TOKEN_HERE
```

**Option B: Using username/password**
```bash
python -m twine upload dist/*
# Will prompt for username and password
```

## Test Installation

After uploading, test the installation:
```bash
# In a fresh environment
pip install torchCompactRadius

# Test import
python -c "import torchCompactRadius; print('Success!')"
```

## Upload to Test PyPI (Optional)

Before uploading to the main PyPI, you can test with Test PyPI:

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Install from Test PyPI to test
pip install --index-url https://test.pypi.org/simple/ torchCompactRadius
```

## Version Management

- Update version in `setup.py` before each upload
- PyPI doesn't allow re-uploading the same version
- Use semantic versioning (e.g., 0.5.1, 0.6.0, etc.)

## Important Notes

1. **Source-only distribution**: We're building source distributions only (no wheels) because users need to compile against their specific PyTorch/CUDA versions.

2. **Extension compilation**: When users run `pip install torchCompactRadius`, the extensions will be compiled during installation using their local PyTorch and CUDA setup.

3. **Build dependencies**: The `pyproject.toml` ensures that torch and other build dependencies are available during installation.

4. **CUDA support**: The package will automatically detect and build with CUDA if available in the user's environment.

## Troubleshooting

- **"File already exists" error**: You need to increment the version number
- **Authentication failed**: Check your API token or username/password
- **Build errors**: Ensure all source files are included in the package

## Example Complete Workflow

```bash
# 1. Update version in setup.py
# 2. Test locally
python -m build --sdist
pip install dist/torchcompactradius-*.tar.gz --force-reinstall

# 3. Upload
./build_and_upload.sh
```
