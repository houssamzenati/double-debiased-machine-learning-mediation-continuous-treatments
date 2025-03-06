#!/bin/bash

#
#SBATCH --job-name=bandwidth_uniform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --output=log/bandwidth_uniform_job_%A_%a.txt
#SBATCH --error log/error_bandwidth_uniform_%A_%a.out
#
#SBATCH --array=0-599

# change to your own path
INPUT_FILE=/scratch/hzenati/double-debiased-machine-learning-mediation-continuous-treatments/experiment_parameters/bandwidth_experiment_parameters.csv

VALUES=({1..600})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in expname epsilon mode sample_size random_seed 
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done


python experiment_bandwidth_uniform.py --run --expname $expname --epsilon $epsilon --estimator kme_dml --bandwidth_mode $mode --n_samples $sample_size --random_seed $random_seed
