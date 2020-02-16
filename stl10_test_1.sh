#!/bin/bash -l
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
#SBATCH --mem=30000
#SBATCH --constrain=v100

conda activate py2
#module load libs/cuda

#export CUDA_VISIBLE_DEVICES=0 && python -m code.scripts.cluster.cluster_sobel_twohead --model_ind 569 --arch ClusterNet5gTwoHead --mode IID --#dataset STL10 --dataset_root /users/k1763920/IIC/datasets/stl10 --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --#num_epochs 2000 --batch_sz 700 --num_dataloaders 5 --num_sub_heads 5 --mix_train --crop_orig --rand_crop_sz 64 --input_sz 64 --head_A_first --#double_eval --batchnorm_track

export CUDA_VISIBLE_DEVICES=2 && python -m code.scripts.cluster.YT_BB_script --model_ind 640  --arch ClusterNet5gTwoHead --mode IID --dataset YT_BB --dataset_root /users/k1763920/yt_bb_util --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --num_epochs 4 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 5 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs
