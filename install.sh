#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/megablocks
source .env/bin/activate

# pip version up
pip install --upgrade pip

# pip install requirements
pip install -r requirements.txt


git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
cd ..

pip install zarr

pip install tensorstore
pip install megablocks[all]