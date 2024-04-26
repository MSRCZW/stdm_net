#!/bin/bash
#JSUB -q aiai
#JSUB -gpgpu 1 
#JSUB -m gpu08
#JSUB -app default
#JSUB -n 1
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate dsta
#unset PYTHONPATH
fuser -V /dev/nvidia*
nvidia-smi > log6.txt
#python ./train_val_test/train.py 
#python -c "import torch;print(torch.version.cuda)"> log.txt
#
