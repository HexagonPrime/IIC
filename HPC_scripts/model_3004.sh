#!/bin/bash -l
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --mem=30000
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100
#SBATCH --time=4-0:00

conda activate py2
#module load libs/cuda

export CUDA_VISIBLE_DEVICES=0 && python -m code.scripts.cluster.YT_BB_script --model_ind 3004 --arch ClusterNet5gTwoHead --mode IID --dataset YT_BB --dataset_root "/users/k1763920/yt_bb" --out_root "/users/k1763920/out/" --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 5 --input_sz 32 --crop_orig --rand_crop --rand_crop_sz 20 --head_A_first --head_B_epochs 2 --base_frame 0 --base_interval 1 --base_num 10 --interval 4 --crop_by_bb --train_partition 'train' --test_partition 'test' --test_on_all_frame
