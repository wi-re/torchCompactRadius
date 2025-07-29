# Installation Guide for torchCompactRadius

## Standard Installation (Builds from Source)

For most users, simply install directly from PyPI:

```bash
pip install torchCompactRadius
```

This will **always build from source** to ensure compatibility with your specific PyTorch and CUDA versions.

## Pre-compiled Wheels (Advanced Users)

If you want to use pre-compiled wheels for faster installation, you can install from our wheel repository:

```bash
# For specific PyTorch and CUDA versions
pip install torchCompactRadius -f https://your-wheel-repo.com/torch-2.5.0+cu121/

# Or for CPU-only
pip install torchCompactRadius -f https://your-wheel-repo.com/torch-2.5.0+cpu/
```

## Build Requirements

When building from source, you need:

- PyTorch >= 1.9.0
- A compatible C++ compiler
- CUDA toolkit (for GPU support)
- Python development headers

### Linux/macOS
```bash
# Most dependencies are handled automatically
pip install torchCompactRadius
```

### Windows
```bash
# Ensure Visual Studio Build Tools are installed
pip install torchCompactRadius
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related build errors:

```bash
# Force CPU-only build
FORCE_ONLY_CPU=1 pip install torchCompactRadius

# Force CUDA build
FORCE_CUDA=1 pip install torchCompactRadius
```

### Build from Local Source
```bash
git clone https://github.com/wi-re/torchCompactRadius.git
cd torchCompactRadius
pip install -e .
```

## Why Build from Source?

This package contains compiled C++/CUDA extensions that are highly dependent on:
- Your specific PyTorch version
- Your CUDA version and architecture
- Your system's C++ ABI

Building from source ensures maximum compatibility and performance.
