#!/bin/bash

#
#SBATCH --job-name=sani_binomial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --output=log/sani_binomialm_job_%A_%a.txt
#SBATCH --error log/error_sani_binomial_%A_%a.out
#
#SBATCH --array=0-499

# Specify the path to the config file
# change to your own path
INPUT_FILE=/scratch/hzenati/double-debiased-machine-learning-mediation-continuous-treatments/experiment_parameters/sani_experiment_binomial_parameters.csv

VALUES=({1001..1500})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in estimator sample_size random_seed 
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done

python experiment_sani_binomial.py --run --estimator $estimator --n_samples $sample_size --random_seed $random_seed
