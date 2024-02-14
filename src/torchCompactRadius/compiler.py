
from pathlib import Path
import subprocess
import sys
import glob
import os
from typing import Optional
import platform
import torch
from torch.utils.cpp_extension import load

directory = Path(__file__).resolve().parent

def find_cuda_home():
    """
    Finds the CUDA home directory by checking various possible locations.
    Based on the original script from PyTorch
    
    Returns:
        str: The path to the CUDA home directory, or None if it is not found.
    """
    IS_WINDOWS = sys.platform == 'win32'

    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
#             print('.', which)
            nvcc = subprocess.check_output(
                [which, 'nvcc'], env = dict(PATH='%s:%s/bin' % (os.environ['PATH'], sys.exec_prefix))).decode().rstrip('\r\n')
#             print(nvcc)
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    if cuda_home is not None:
        os.environ['CUDA_HOME'] = cuda_home
    # print('Cuda compiler:', cuda_home)
    return cuda_home

find_cuda_home()

def getComputeCapability(device):
    """
    Get the compute capability of the specified CUDA device.

    Args:
        device (int): The index of the CUDA device.

    Returns:
        int: The compute capability of the device.

    """
    return int(''.join([str(s) for s in torch.cuda.get_device_capability(device)]))


def build_cpp_standard_arg(cpp_standard):
    """
    Build the argument for the C++ standard based on the given cpp_standard.
    Arguments are in the form of 'c++17'.

    Args:
        cpp_standard (str): The desired C++ standard.

    Returns:
        str: The argument for the C++ standard based on the platform.
    """
    if platform.system() == "Windows":
        return "/std:" + cpp_standard
    else:
        return "-std=" + cpp_standard

def compileSourceFiles(sourceFiles, module_name, directory: Optional[str] = None, 
                verbose = True, additionalFlags = [""], 
                openMP : bool = False, tbb : bool = False,
                verboseCuda : bool = False, cpp_standard : str = "c++17", cuda_arch : Optional[int] = None):
    """
    Compiles the given source files into a module.

    Args:
        sourceFiles (List[str]): List of source file paths.
        module_name (str): Name of the module.
        directory (Optional[str], optional): Directory path where the source files are located. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        additionalFlags (List[str], optional): Additional compilation flags. Defaults to [""].
        openMP (bool, optional): Whether to enable OpenMP support. Defaults to False.
        verboseCuda (bool, optional): Whether to print verbose output for CUDA. Defaults to False.
        cpp_standard (str, optional): C++ standard version. Defaults to "c++17".
        cuda_arch (Optional[int], optional): CUDA architecture version. Defaults to None.

    Returns:
        torch.utils.cpp_extension.CppExtension: Compiled module.
    """
    cpp_standard_arg = build_cpp_standard_arg(cpp_standard)

    hostFlags = [cpp_standard_arg, "-fPIC", "-O3", "-fopenmp"] if openMP else [cpp_standard_arg, "-fPIC", "-O3"]
    cudaFlags = [cpp_standard_arg,'-O3']
    
    if torch.cuda.is_available():
        computeCapability = getComputeCapability(torch.cuda.current_device()) if cuda_arch is None else cuda_arch
        if verbose:
            print('computeCapability:', computeCapability)
        smFlag = '-gencode=arch=compute_%d,code=sm_%d' % (computeCapability, computeCapability)
        cudaFlags.append(smFlag)
        if verbose:
            print('smFlag:', smFlag)
        cudaFlags.append('--use_fast_math')

        cudaFlags.append('-DCUDA_VERSION')
        hostFlags.append('-DCUDA_VERSION')
    
    # ldFlags = ['openmp'] if openMP else []
    ldFlags = []
    if verboseCuda:
        cudaFlags.append('--ptxas-options="-v "')

    if verbose:
        print('hostFlags:', hostFlags)
        print('cudaFlags:', cudaFlags)
    if tbb:
        cudaFlags.append('-DTBB_VERSION')
        hostFlags.append('-DTBB_VERSION')

    if openMP:
        # clang under macos does not support fopenmp so check for existence of clang via homebrew
        # will fail if no clang is found
        if platform.system() == "Darwin":

            os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'] if 'LDFLAGS' in os.environ else '', '-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++')
            os.environ['PATH'] = '%s %s' % ('/opt/homebrew/opt/llvm/bin:$PATH"', os.environ['PATH'])
            os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'] if 'LDFLAGS' in os.environ else '', "-L/opt/homebrew/opt/libomp/lib")
            os.environ['CPPFLAGS'] = '%s %s' % (os.environ['CPPFLAGS'] if 'CPPFLAGS' in os.environ else '', "-I/opt/homebrew/opt/llvm/include")
            os.environ['LDFLAGS'] = '%s %s' % (os.environ['LDFLAGS'], "-L/opt/homebrew/opt/libomp/lib")
            os.environ['CPPFLAGS'] = '%s %s' % (os.environ['CPPFLAGS'], "-I/opt/homebrew/opt/llvm/include:-I/opt/homebrew/opt/libomp/include")
            os.environ['CC'] = '/opt/homebrew/opt/llvm/bin/clang'
            os.environ['CXX'] = '/opt/homebrew/opt/llvm/bin/clang'
            nvcc = subprocess.check_output(
                ['which', 'clang'], env = dict(PATH='%s:%s/bin' % (os.environ['PATH'], sys.exec_prefix))).decode().rstrip('\r\n')
        
        cudaFlags.append('-DOMP_VERSION')
        hostFlags.append('-DOMP_VERSION')
    if directory is None:
        directory = Path(__file__).resolve().parent
    if verbose:
        print('directory:', directory)


    for sourceFile in sourceFiles:
        if verbose:
            print('sourceFile:', sourceFile)
        if os.path.exists(sourceFile) or os.path.exists(os.path.join(directory, sourceFile)):
            if verbose:
                print('source file exists:', sourceFile)
            continue
        else:
            raise RuntimeError('source file does not exist:', sourceFile)
    sourceFiles = [os.path.abspath(os.path.join(directory, sourceFile)) if os.path.exists(os.path.join(directory, sourceFile)) else sourceFile for sourceFile in sourceFiles]
    cppFiles = [sourceFile for sourceFile in sourceFiles if sourceFile.endswith('.cpp')]
    cuFiles = ['"%s"' % (sourceFile) for sourceFile in sourceFiles if sourceFile.endswith('.cu')]

    if verbose:
        print('cppFiles:', cppFiles)
        print('cuFiles:', cuFiles)

    if torch.cuda.is_available():
        return load(name=module_name, 
            sources=sourceFiles, verbose=verbose, extra_cflags=hostFlags, extra_cuda_cflags=cudaFlags, extra_ldflags=ldFlags)
    else:
        return load(name=module_name, 
            sources=cppFiles, verbose=verbose, extra_cflags=hostFlags, extra_ldflags=ldFlags)
