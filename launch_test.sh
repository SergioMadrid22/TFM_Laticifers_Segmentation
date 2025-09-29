#!/bin/bash

#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 5-00:00
#SBATCH -c 4
#SBATCH -o logs/slurm/%J.%N-%x.out
#SBATCH -J test_tta

config_file=$1
model_path=$2

python src/postprocess_final.py \
    -c $config_file \
    -d $model_path \
    -o predictions/final_experiments_hysteresis_400_branch5