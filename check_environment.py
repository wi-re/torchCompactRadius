#!/usr/bin/env python3
"""
Environment checker for torchCompactRadius build requirements.
Run this before installing to verify your environment is ready.
"""

import sys
import os
import subprocess

def check_environment():
    """Check if the environment is ready for building torchCompactRadius."""
    print("🔍 Checking build environment for torchCompactRadius...")
    print("=" * 60)
    
    # Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # PyTorch check
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   Device {i}: {props.name} (Compute {props.major}.{props.minor})")
        else:
            print("ℹ️  CUDA not available - will build CPU-only version")
    except ImportError:
        print("❌ PyTorch not found - install PyTorch first!")
        return False
    
    # CUDA_HOME check
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME:
            print(f"✅ CUDA_HOME: {CUDA_HOME}")
        else:
            print("ℹ️  CUDA_HOME not set - CPU-only build")
    except ImportError:
        pass
    
    # Compiler check
    print("\n🔧 Checking compilers...")
    
    # C++ compiler
    try:
        result = subprocess.run(['g++', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ g++: {version_line}")
        else:
            print("❌ g++ not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            result = subprocess.run(['clang++', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ clang++: {version_line}")
            else:
                print("❌ No C++ compiler found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ No C++ compiler found")
    
    # NVCC check (if CUDA available)
    if torch.cuda.is_available():
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        print(f"✅ NVCC: {line.strip()}")
                        break
            else:
                print("⚠️  nvcc not found - CUDA builds may fail")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  nvcc not found - CUDA builds may fail")
    
    print("\n" + "=" * 60)
    print("🚀 Environment check complete!")
    print("\nTo install torchCompactRadius:")
    print("   pip install torchCompactRadius")
    print("\nTo force CPU-only build:")
    print("   FORCE_ONLY_CPU=1 pip install torchCompactRadius")
    
    return True

if __name__ == "__main__":
    check_environment()
