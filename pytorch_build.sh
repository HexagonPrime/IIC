#!/bin/bash -l
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
#SBATCH --mem=30000
"#SBATCH --constrain=v100"

conda activate myenv
python setup.py install

