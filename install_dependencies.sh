#!/bin/bash 
set -euo pipefail

# make conda callable in a bash script
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
PS1=

# Create conda environment
conda create --yes --name pose-refinement python=3.6 ffmpeg
conda activate pose-refinement
conda install --yes opencv3 -c menpo
conda install --yes pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
conda install --yes cudatoolkit-dev=10.1 -c conda-forge
python -m pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

# Set up HR-Net
git clone https://github.com/HRNet/HRNet-Human-Pose-Estimation.git hrnet 
cd hrnet/lib
make
cd ..

python -m pip install scipy h5py json_tricks

# Download hrnet model from google drive. Original link: 
# https://drive.google.com/file/d/1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38/view?usp=sharing
LINK=`wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'`
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$LINK&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38" -O pose_hrnet_w32_256x192.pth 
rm -rf /tmp/cookies.txt