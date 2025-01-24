#!/bin/bash

source .github/workflows/cuda/cu124-Linux-env.sh
conda/torchCompactRadius/build_conda.sh 3.11 2.5.0 cu124
conda/torchCompactRadius/build_conda.sh 3.11 2.5.1 cu124
source .github/workflows/cuda/cu121-Linux-env.sh
conda/torchCompactRadius/build_conda.sh 3.11 2.5.0 cu121
conda/torchCompactRadius/build_conda.sh 3.11 2.5.1 cu121
source .github/workflows/cuda/cu116-Linux-env.sh
conda/torchCompactRadius/build_conda.sh 3.11 2.5.0 cu116
conda/torchCompactRadius/build_conda.sh 3.11 2.5.1 cu116