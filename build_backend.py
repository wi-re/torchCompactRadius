"""
Custom build backend to ensure C++/CUDA extensions are built during PyPI installation.
"""

import os
import sys
from setuptools import build_meta as _orig
from setuptools.build_meta import *

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Custom build_wheel that ensures extensions are compiled."""
    print("=" * 60)
    print("CUSTOM BUILD: Building wheel with C++/CUDA extensions")
    print("=" * 60)
    
    # Ensure setup.py is properly executed
    result = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    
    print("=" * 60)
    print("CUSTOM BUILD: Wheel build completed")
    print("=" * 60)
    
    return result

def build_sdist(sdist_directory, config_settings=None):
    """Custom build_sdist."""
    print("CUSTOM BUILD: Building source distribution")
    return _orig.build_sdist(sdist_directory, config_settings)
