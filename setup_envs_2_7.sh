conda config --add channels defaults

conda create -n dev_py311_torch-2.7.1_cu118 python=3.11.0
conda activate dev_py311_torch-2.7.1_cu118
pip install numpy scipy
conda install cuda -c nvidia/label/cuda-11.8.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

conda create -y -n dev_py311_torch-2.7.1_cu126 python=3.11.0
conda activate dev_py311_torch-2.7.1_cu126
pip install numpy scipy
conda install cuda -c nvidia/label/cuda-12.6.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


conda create -y -n dev_py311_torch-2.7.1_cu128 python=3.11.0
conda activate dev_py311_torch-2.7.1_cu128
pip install numpy scipy
conda install cuda -c nvidia/label/cuda-12.8.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
