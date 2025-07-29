import glob
import os
import os.path as osp
import platform
import sys
from itertools import product

from setuptools import find_packages, setup

# Defer torch and dsl imports until needed
# import torch
# import dsl

__version__ = '0.5.0'
URL = 'https://github.com/wi-re/torchCompactRadius'

def get_cuda_config():
    """Get CUDA configuration, importing torch only when needed."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        
        WITH_CUDA = False
        if torch.cuda.is_available():
            WITH_CUDA = CUDA_HOME is not None or torch.version.hip
        
        # Print build configuration for transparency
        print(f"PyTorch version: {torch.__version__}")
        if WITH_CUDA:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA_HOME: {CUDA_HOME}")
        else:
            print("Building CPU-only version")
        
        return WITH_CUDA, torch
    except ImportError:
        print("Warning: torch not available during setup, building CPU-only version")
        return False, None

WITH_CUDA, torch_module = get_cuda_config()

suffices = ['cpu', 'cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda', 'cpu']
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    suffices = ['cuda']
if os.getenv('FORCE_ONLY_CPU', '0') == '1' or os.getenv('FORCE_CUDA', '1') == '0':
    suffices = ['cpu']

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'

if 'cuda' not in suffices:
    print('CUDA is not available. Building without CUDA.')
    WITH_CUDA = False

def get_extensions():
    # Import torch extensions only when building
    try:
        from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                              CUDAExtension)
        import dsl
    except ImportError as e:
        print(f"Warning: Could not import required dependencies: {e}")
        return []
    
    extensions = []

    extensions_dir = osp.join('src/torchCompactRadius/cppSrc')
    main_files = []

    for subfolder in os.listdir(extensions_dir):
        if os.path.isdir(os.path.join(extensions_dir, subfolder)):
            main_files += glob.glob(osp.join(extensions_dir, subfolder, '*.cpp'))
            if 'cuda' in suffices:
                main_files += glob.glob(osp.join(extensions_dir, subfolder, '*.cu'))
            # remove generated 'hip' files, in case of rebuilds

    main_files += glob.glob(osp.join(extensions_dir, '*.cpp'))
    if 'cuda' in suffices:
        main_files += glob.glob(osp.join(extensions_dir, '*.cu'))
    # remove generated 'hip' files, in case of rebuilds
    # main_files = [path for path in main_files if 'hip' not in path]
    main_files = [path for path in main_files if 'hip' not in path]

    print(f'Found {len(main_files)} extension main files.')
    for main in main_files:
        print(f'Including {main}.')
    print(f'Building with CUDA: {WITH_CUDA}')
    print(f'Building suffices: {suffices}')

    header_files = glob.glob(osp.join(extensions_dir, '*.h'))
    for subfolder in os.listdir(extensions_dir):
        if os.path.isdir(os.path.join(extensions_dir, subfolder)):
            header_files += glob.glob(osp.join(extensions_dir, subfolder, '*.h'))
    for header in header_files:
        print(f'Processing {header}.')
        dsl.process(header)

    suffix = 'cuda' if 'cuda' in suffices else 'cpu'
    # if 'cuda' in suffices and not WITH_CUDA:
        # raise ValueError('CUDA is not available. Please install CUDA.')
    
    extra_compile_args = {'cxx': ['-O2', "-DPy_LIMITED_API=0x03090000"]}
    extra_link_args = ['-s']

    define_macros = []#[('CUDA_VERSION',None)] if WITH_CUDA else []
    if sys.platform == 'win32':
        extra_compile_args['cxx'] += ['/openmp']
    else:
        if sys.platform == 'darwin':
            extra_compile_args['cxx'] += ['-Wno-sign-compare', '-Wno-unused-variable']
        else:
            extra_compile_args['cxx'] += ['-fopenmp', '-Wno-sign-compare', '-Wno-unused-variable']

    if sys.platform == 'darwin':
        extra_compile_args['cxx'] += ['-D_LIBCPP_DISABLE_AVAILABILITY']
        if platform.machine == 'arm64':
            extra_compile_args['cxx'] += ['-arch', 'arm64']
            extra_link_args += ['-arch', 'arm64']

    if suffix == 'cuda':
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-O2']
        nvcc_flags += [f'-DCOMPILE_WITH_CUDA=1']
        nvcc_flags += [f'-extended-lambda']
        extra_compile_args['nvcc'] = nvcc_flags

    Extension = CppExtension if suffix == 'cpu' else CUDAExtension
    extension = Extension(
        f'torchCompactRadius_{suffix}',
        main_files,
        include_dirs=[os.path.abspath(extensions_dir)],
        define_macros=define_macros,
        # undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        
        py_limited_api=True,
    )
    extensions += [extension]
    print(f'Added extension: {extension.name}')

    return extensions


install_requires = [
    'scipy',
    'torch',
]

build_requires = [
    'pytest'
]

# work-around hipify abs paths
include_package_data = True
try:
    import torch
    if torch.cuda.is_available() and torch.version.hip:
        include_package_data = False
except ImportError:
    pass  # Default to True if torch not available

def get_build_extension():
    """Get BuildExtension class, importing only when needed."""
    try:
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    except ImportError:
        print("Warning: torch not available, using default build extension")
        return None

setup(
    name='torchCompactRadius',
    version=__version__,
    description=('PyTorch Extension Library for compact hash map based neighbor searching '
                 'Algorithms'),
    author='Rene Winchenbach',
    author_email='contact@fluids.dev',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'cluster-algorithms',
    ],
    # python_requires='>=3.8',
    install_requires=install_requires,
    # extras_require={
    #     'test': test_requires,
    # },
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext': get_build_extension()
    } if get_build_extension() is not None else {},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    packages=find_packages(),
    include_package_data=include_package_data,
)