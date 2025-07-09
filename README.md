# Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments

This is the code for our [DML mediation paper with continuous treatments.](https://arxiv.org/abs/2503.06156)

Please cite our work if you find it useful for your research and work:
```
@misc{zenati2025doubledebiasedmachinelearning,
      title={Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments}, 
      author={Houssam Zenati and Judith Abécassis and Julie Josse and Bertrand Thirion},
      year={2025},
      eprint={2503.06156},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2503.06156}, 
}
```

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
sbatch submit_jobs/submit_uniform_baselines_experiment.sh 
```
#### Otherwise
```
python experiment_baselines_uniform.py --sequential_run
```
#### Get Figure 2, Table 3
```
python experiment_baselines_uniform.py --results
```
### Bandwidth experiment 

#### If you use a slurm cluster  
```
python experiment_bandwidth_uniform.py --get_parameters_experiment
```
```
sbatch submit_jobs/submit_uniform_bandwidth_experiment.sh 
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
sbatch submit_jobs/submit_uniform_coverage_experiment.sh 
```
#### Otherwise
```
python experiment_coverage_uniform.py --sequential_run
```
#### Get Table 4
```
python experiment_coverage_uniform.py --results
```

### Parametric/nonparametric experiment 

#### If you use a slurm cluster  
```
python experiment_nonparametric_uniform.py --get_parameters_experiment
```
```
sbatch submit_jobs/submit_uniform_nonparametric_experiment.sh 
```
#### Otherwise
```
python experiment_nonparametric_uniform.py --sequential_run
```
#### Get Table 5
```
python experiment_nonparametric_uniform.py --results
```

## Experiments to compare with (Sani et al, 2024)

#### If you use a slurm cluster  
```
python experiment_sani_uniform.py --get_parameters_experiment
python experiment_sani_binomial.py --get_parameters_experiment

```
```
sbatch submit_jobs/submit_uniform_sani_experiment.sh 
sbatch submit_jobs/submit_binomial_sani_experiment.sh 
```
#### Otherwise
```
python experiment_sani_uniform.py --sequential_run
python experiment_sani_binomial.py --sequential_run

```
#### Get Tables 1, 2
```
python experiment_sani_uniform.py --results
```
#### Get Table 6
```
python experiment_sani_binomial.py --results
```

## Reproduce UKBB application

#### If you use a slurm cluster  
```
python experiment_ukbb.py --get_parameters_experiment
```
```
sbatch submit_jobs/submit_ukbb_experiment.sh 
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
