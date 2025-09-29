#!/bin/bash

# SCRIPT FOR LAUNCHING A JOB ON SLURM WITHOUT GPU, JUST CPUS

#SBATCH --mem=320G
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -c 12
#SBATCH -o logs/%J.%N-%x.out
#SBATCH -J sato_images

python utils/precompute_curriculum_masks.py