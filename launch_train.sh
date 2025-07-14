#!/bin/bash

#SBATCH --mem=320G
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 5-00:00
#SBATCH -c 8
#SBATCH -o logs/%J.%N-%x.out
#SBATCH -J train_segmentation_net

config_file=$1
exp_name=$2


python src/train.py \
    -c $config_file \
    -e $exp_name