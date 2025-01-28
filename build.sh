#!/bin/bash

# rm -rf src/*.so
# rm -rf src/*.egg-info
# rm -rf src/torchCompactRadius/__pycache__
# rm -rf wheels/
# rm -rf dist/
# rm -rf build/

export TORCH_VERSION=$1
export CUDA_VERSION=$2

eval "$(conda shell.bash hook)"
source ~/anaconda3/bin/activate dev_py311_torch-${TORCH_VERSION}_${CUDA_VERSION}
echo "activated dev_py311_torch-${TORCH_VERSION}_${CUDA_VERSION}"

rm -rf src/*.so
rm -rf src/*.egg-info
rm -rf build/

git restore setup.py
git restore src/torchCompactRadius/__init__.py

VERSION=`sed -n "s/^__version__ = '\(.*\)'/\1/p" setup.py`
PYTORCH_VERSION=`python -c "import torch; print(torch.__version__)"`
TORCH_VERSION=`echo "pt$PYTORCH_VERSION" | sed "s/..$//" | sed "s/\.//g"`
if [ "${CUDA_VERSION}" = "cpu" ]; then
    CUDA_VERSION="cpu"
else
    CUDA_VERSION=`python -c "import torch; print(torch.version.cuda)" | sed "s/\.//g"`
fi

echo "Building torch-$PYTORCH_VERSION+cu$CUDA_VERSION"
if [ "${CUDA_VERSION}" = "cpu" ]; then
    VERSION_TAG=`echo "$VERSION+${TORCH_VERSION}cpu"`
else
    VERSION_TAG=`echo "$VERSION+${TORCH_VERSION}cu${CUDA_VERSION}"`
fi

echo "Version tag: $VERSION_TAG"
sed -i "s/$VERSION/$VERSION_TAG/" setup.py
sed -i "s/$VERSION/$VERSION_TAGN/" src/torchCompactRadius/__init__.py
if [ "${CUDA_VERSION}" = "cpu" ]; then
    export FORCE_CUDA=0
    export TORCH_CUDA_ARCH_LIST=""
    FORCE_ONLY_CPU=1 python setup.py develop
else
    # echo "Sourcing .github/workflows/cuda/cu${CUDA_VERSION}-Linux-env.sh"
    source .github/workflows/cuda/cu${CUDA_VERSION}-Linux-env.sh
    python setup.py develop
fi

python setup.py bdist_wheel --dist-dir=dist

if [ "${CUDA_VERSION}" = "cpu" ]; then
    mkdir -p wheels/torch-$PYTORCH_VERSION+cpu
    mv dist/* wheels/torch-$PYTORCH_VERSION+cpu
else
    mkdir -p wheels/torch-$PYTORCH_VERSION+cu$CUDA_VERSION
    mv dist/* wheels/torch-$PYTORCH_VERSION+cu$CUDA_VERSION
fi

conda/torchCompactRadius/build_conda.sh