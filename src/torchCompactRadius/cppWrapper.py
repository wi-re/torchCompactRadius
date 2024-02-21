from torchCompactRadius.compiler import compileSourceFiles
from typing import Optional, List
import torch
# from torch.profiler import record_function

neighborSearch_cpp = compileSourceFiles(
    ['cppSrc/neighborhoodDynamic.cpp', 'cppSrc/neighborhoodDynamic.cu', 
     'cppSrc/neighborhoodFixed.cpp', 'cppSrc/neighborhoodFixed.cu',
     'cppSrc/hashing.cpp', 'cppSrc/hashing.cu',
     'cppSrc/neighborhoodSmall.cpp', 'cppSrc/neighborhoodSmall.cu',
     'cppSrc/cppWrapper.cpp'], module_name = 'neighborSearch', verbose = False, openMP = True, verboseCuda = False, cuda_arch = None)

countNeighbors_cpp = neighborSearch_cpp.countNeighbors
buildNeighborList_cpp = neighborSearch_cpp.buildNeighborList
countNeighborsFixed_cpp = neighborSearch_cpp.countNeighborsFixed
buildNeighborListFixed_cpp = neighborSearch_cpp.buildNeighborListFixed
hashCells_cpp = neighborSearch_cpp.computeHashIndices
neighborSearchSmall = neighborSearch_cpp.neighborSearchSmall
neighborSearchSmallFixed = neighborSearch_cpp.neighborSearchSmallFixed