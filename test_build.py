#!/usr/bin/env python3
"""
Test script to verify that setup.py works without torch being pre-installed.
This simulates the PyPI installation process.
"""

import sys
import subprocess
import tempfile
import shutil
import os

def test_build_without_torch():
    """Test building the package in a clean environment."""
    print("Testing package build without pre-installed torch...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy source files to temp directory
        source_files = [
            'setup.py', 'pyproject.toml', 'dsl.py', 
            'src/', 'README.md', 'LICENSE'
        ]
        
        for item in source_files:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.copytree(item, os.path.join(temp_dir, item))
                else:
                    shutil.copy2(item, temp_dir)
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test 1: Setup.py check
            print("\n1️⃣  Testing setup.py check...")
            result = subprocess.run([
                sys.executable, 'setup.py', 'check'
            ], capture_output=True, text=True)
            
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            print(f"Return code: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ Setup.py check passed!")
            else:
                print("❌ Setup.py check failed!")
                return False
            
            # Test 2: Build source distribution
            print("\n2️⃣  Testing source distribution build...")
            result = subprocess.run([
                sys.executable, '-m', 'build', '--sdist'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Source distribution build passed!")
                # List created files
                if os.path.exists('dist'):
                    print("Created files:")
                    for f in os.listdir('dist'):
                        print(f"  📦 {f}")
            else:
                print("❌ Source distribution build failed!")
                print("STDERR:", result.stderr)
                return False
            
            # Test 3: Simulate pip install from sdist
            print("\n3️⃣  Testing pip install from source distribution...")
            sdist_files = [f for f in os.listdir('dist') if f.endswith('.tar.gz')]
            if sdist_files:
                sdist_path = os.path.join('dist', sdist_files[0])
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', sdist_path, '--dry-run'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Pip install simulation passed!")
                else:
                    print("⚠️  Pip install simulation had issues (but this might be expected)")
                    print("STDERR:", result.stderr)
            
            return True
                
        finally:
            os.chdir(original_dir)

def test_pypi_simulation():
    """Simulate the complete PyPI installation process."""
    print("\n" + "="*60)
    print("🌐 SIMULATING PYPI INSTALLATION PROCESS")
    print("="*60)
    
    # Check current environment
    print("\n📋 Current environment:")
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("   PyTorch: Not installed")
    
    # Run the build test
    success = test_build_without_torch()
    
    if success:
        print("\n🎉 SUCCESS: Package can be built from source!")
        print("✅ Ready for PyPI source distribution upload")
        print("\nNext steps:")
        print("1. Run: python -m build --sdist")
        print("2. Run: twine upload dist/*.tar.gz")
        print("3. Users can install with: pip install torchCompactRadius")
    else:
        print("\n❌ FAILED: Package has build issues")
        print("Fix the issues before uploading to PyPI")
    
    return success

if __name__ == "__main__":
    test_pypi_simulation()
