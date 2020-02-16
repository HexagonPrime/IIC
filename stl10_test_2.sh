#!/bin/bash -l
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --ntasks=1
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
#SBATCH --mem=25000
#SBATCH --constrain=v100
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1

conda activate myenv
module load libs/cuda

export CUDA_VISIBLE_DEVICES=0 && python -m code.scripts.cluster.cluster_sobel_twohead --model_ind 569 --arch ClusterNet5gTwoHead --mode IID --dataset STL10 --dataset_root /users/k1763920/IIC/datasets/stl10 --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 700 --num_dataloaders 5 --num_sub_heads 5 --mix_train --crop_orig --rand_crop_sz 64 --input_sz 64 --head_A_first --double_eval --batchnorm_track