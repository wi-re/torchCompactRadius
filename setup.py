import glob
import os
import os.path as osp
import platform
import sys
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

import dsl

__version__ = '0.5.0'
URL = 'https://github.com/wi-re/torchCompactRadius'

WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None or torch.version.hip

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

# test_requires = [
#     'pytest',
#     'pytest-cov',
# ]

# work-around hipify abs paths
include_package_data = True
if torch.cuda.is_available() and torch.version.hip:
    include_package_data = False

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
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    packages=find_packages(),
    include_package_data=include_package_data,
)