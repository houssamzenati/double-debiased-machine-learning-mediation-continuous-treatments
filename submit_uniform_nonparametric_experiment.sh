#!/bin/bash

#
#SBATCH --job-name=baselines_uniform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --output=log/baselines_uniform_job_%A_%a.txt
#SBATCH --error log/error_baselines_uniform_%A_%a.out
#
#SBATCH --array=0-999

# Specify the path to the config file
# change to your own path
INPUT_FILE=/scratch/hzenati/double-debiased-machine-learning-mediation-continuous-treatments/nonparametric_experiment_parameters.csv

VALUES=({1..1000})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in estimator sample_size density kernel random_seed 
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done

python experiment_nonparametric_uniform.py --run --estimator $estimator --n_samples $sample_size --density $density --kernel $kernel --random_seed $random_seed
