#!/bin/bash -l
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100

cat /proc/meminfo
