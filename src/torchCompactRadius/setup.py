from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='torchCompactRadiusCPP',
      ext_modules=[CUDAExtension.CppExtension('torchCompactRadiusCPP', 
    ['cppSrc/neighborhoodDynamic.cpp', 'cppSrc/neighborhoodDynamic.cu', 
     'cppSrc/neighborhoodFixed.cpp', 'cppSrc/neighborhoodFixed.cu',
     'cppSrc/hashing.cpp', 'cppSrc/hashing.cu',
     'cppSrc/neighborhoodSmall.cpp', 'cppSrc/neighborhoodSmall.cu',
     'cppSrc/cppWrapper.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
