conda config --add channels defaults

conda create -n dev_py311_torch-2.5.0_cu124 python=3.11.0
conda activate dev_py311_torch-2.5.0_cu124
pip install numpy scipy
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

conda create -y -n dev_py311_torch-2.5.0_cu121 python=3.11.0
conda activate dev_py311_torch-2.5.0_cu121
pip install numpy scipy
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda create -y -n dev_py311_torch-2.5.0_cu118 python=3.11.0
conda activate dev_py311_torch-2.5.0_cu118
pip install numpy scipy
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.6 -c pytorch -c nvidia

conda create -y -n dev_py311_torch-2.5.0_cpu python=3.11.0
conda activate dev_py311_torch-2.5.0_cpu
pip install numpy scipy
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch 