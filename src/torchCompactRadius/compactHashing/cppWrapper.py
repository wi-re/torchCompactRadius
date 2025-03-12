from torchCompactRadius.compiler import compileSourceFiles
from typing import Optional, List
import torch
import warnings
# from torch.profiler import record_function

hasPrecompiledCPU = False
hasPrecompiledGPU = False

try:
    import torchCompactRadius_cuda
    neighborSearch_cpp = torchCompactRadius_cuda
    if not torch.cuda.is_available():
        warnings.warn('Precompiled GPU version of the neighbor search is being used even though CUDA is not available!')
    hasPrecompiledGPU = True
except ImportError:
    # warnings.warn('No cuda version of the neighbor search is available.')
    pass
if not hasPrecompiledGPU:
    try:
        import torchCompactRadius_cpu
        neighborSearch_cpp = torchCompactRadius_cpu
        if torch.cuda.is_available():
            warnings.warn('Precompiled CPU version of the neighbor search is being used even though CUDA is available!')
        hasPrecompiledCPU = True
    except ImportError:
        # warnings.warn('No cpu version of the neighbor search is available.')
        pass

if not hasPrecompiledCPU and not hasPrecompiledGPU:
    warnings.warn('No precompiled version of the neighbor search is available.')
    neighborSearch_cpp = compileSourceFiles(
        ['cppSrc/neighborhoodDynamic_cpu.cpp',  'cppSrc/neighborhoodDynamic_cuda.cu', 
        # 'cppSrc/neighborhoodFixed_cpu.cpp',     'cppSrc/neighborhoodFixed_cuda.cu',
        'cppSrc/neighborhood_mlm_cpu.cpp',     'cppSrc/neighborhood_mlm_cuda.cu',
        'cppSrc/hashing_cpu.cpp',               'cppSrc/hashing_cuda.cu',
        # 'cppSrc/neighborhoodSmall_cpu.cpp',     'cppSrc/neighborhoodSmall_cuda.cu',
        'cppSrc/cppWrapper.cpp'], module_name = 'torchCompactRadius_jit', verbose = False, openMP = True, verboseCuda = False, cuda_arch = None)

countNeighbors_cpp = neighborSearch_cpp.countNeighbors
buildNeighborList_cpp = neighborSearch_cpp.buildNeighborList
# countNeighborsFixed_cpp = neighborSearch_cpp.countNeighborsFixed
# buildNeighborListFixed_cpp = neighborSearch_cpp.buildNeighborListFixed
hashCells_cpp = neighborSearch_cpp.computeHashIndices
# neighborSearchSmall = neighborSearch_cpp.neighborSearchSmall
# neighborSearchSmallFixed = neighborSearch_cpp.neighborSearchSmallFixed

# countNeighbors_cpp = None
# buildNeighborList_cpp = None
countNeighborsFixed_cpp = None
buildNeighborListFixed_cpp = None
neighborSearchSmall = None
neighborSearchSmallFixed = None