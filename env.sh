conda create env --name torch_22 python=3.11
conda activate torch_22
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-toolkit -c pytorch -c nvidia
pip install tqdm seaborn pandas numpy scipy scikit-learn matplotlib
conda install pytorch-cluster -c pyg
