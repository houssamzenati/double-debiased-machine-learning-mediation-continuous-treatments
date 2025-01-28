# Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments

This is the code for the DML mediation project with continuous treatments. 

## Prerequisites and environment setup, create your conda environment (python 3.12) and run

```
pip install -r requirements.txt
```
## Reproduce Hsu et al, 2020 experiments

### Baselines experiment 

#### If you use a slurm cluster 
``` 
python experiment_baselines_uniform.py --get_parameters_experiment
```
```
sbatch submit_uniform_baselines_experiment.sh 
```
#### Otherwise
```
python experiment_baselines_uniform.py --sequential_run
```
#### Get Figure 2, Tables 1, 2, 3
```
python experiment_baselines_uniform.py --results
```
### Bandwidth experiment 

#### If you use a slurm cluster  
```
python experiment_bandwidth_uniform.py --get_parameters_experiment
```
```
sbatch submit_uniform_bandwidth_experiment.sh 
```
#### Otherwise

python experiment_bandwidth_uniform.py --sequential_run

#### Get Figure 3
```
python experiment_bandwidth_uniform.py --results
```

### Coverage experiment 

#### If you use a slurm cluster  
```
python experiment_coverage_uniform.py --get_parameters_experiment
```
```
sbatch submit_uniform_coverage_experiment.sh 
```
#### Otherwise
```
python experiment_coverage_uniform.py --sequential_run
```
#### Get Table 4
```
python experiment_coverage_uniform.py --results
```
## Reproduce UKBB application

#### If you use a slurm cluster  
```
python experiment_ukbb.py --get_parameters_experiment
```
```
sbatch submit_ukbb_experiment.sh 
```
#### Otherwise
```
python experiment_ukbb.py --sequential_run
```
#### Get Figure 5, 6
```
python experiment_ukbb.py --results
```

Reference:

A large part of the conditional density estimation code comes from the library
https://github.com/freelunchtheorem/Conditional_Density_Estimation. 