#!/bin/bash
#JSUB -q aiai
#JSUB -gpgpu 4 
#JSUB -app default
#JSUB -m gpu08
#JSUB -n 5
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate dsta
#unset PYTHONPATH
nvidia-smi > log.txt
python ./train_val_test/train.py 
#python -c "import torch;print(torch.version.cuda)"> log.txt
#
