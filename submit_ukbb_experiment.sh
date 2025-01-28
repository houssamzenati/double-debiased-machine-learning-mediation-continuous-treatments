#!/bin/bash

#
#SBATCH --job-name=mediation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
#SBATCH --output=log/output_ukbb_%A_%a.txt
#SBATCH --error log/error_ukbb_%A_%a.out
#
#SBATCH --array=0-999

# Specify the path to the config file
# change to your own path
INPUT_FILE=/data/work/ukbb_parameters_hb1ac_ukbb.csv

VALUES=({1..1000})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in estimator d random_seed 
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done

# export PATH=/home/mind/hzenati/.local/miniconda3/condabin:$PATH
# conda activate mind
# export PYTHONPATH=/home/mind/hzenati/.local/

python experiment_ukbb.py --run --estimator $estimator --d $d --random_seed $random_seed
