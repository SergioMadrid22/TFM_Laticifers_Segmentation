#!/bin/bash

#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 5-00:00
#SBATCH -c 4
#SBATCH -o logs/slurm/%J.%N-%x.out
#SBATCH -J test_tta

config_file=$1
predictions_dir=$2
ae_dir=$3
output_dir=$4

python src/refine_ae_test.py \
    -c $config_file \
    -p $predictions_dir \
    --ae_dir $ae_dir \
    -o $output_dir