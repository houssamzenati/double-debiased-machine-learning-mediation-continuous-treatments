#!/bin/bash

#
#SBATCH --job-name=baselines_uniform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --output=log/baselines_uniform_job_%A_%a.txt
#SBATCH --error log/error_baselines_uniform_%A_%a.out
#
#SBATCH --array=0-199

# Specify the path to the config file
# change to your own path
INPUT_FILE=/scratch/hzenati/double-debiased-machine-learning-mediation-continuous-treatments/baselines_experiment_parameters.csv

VALUES=({1001..1200})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in estimator sample_size random_seed 
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done

python experiment_baselines_uniform.py --run --estimator $estimator --n_samples $sample_size --random_seed $random_seed
