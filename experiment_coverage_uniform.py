import os
import sys
import csv

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

import numpy as np
import time
import os
from datetime import date
today = date.today()
from tqdm import tqdm
import argparse
import pandas as pd
from utils.utils import str_to_bool
import seaborn as sns
import matplotlib.pyplot as plt

from utils.parse import get_df, preprocess, save_result
from utils.loader import get_estimator_by_name
from utils.utils import display_experiment_configuration
from data.uniform import UniformEnv

param_settings = {
    'estimator': None,
    'regularization': True,
    'sample_splits': False,
    'reg_lambda': 1e-2,
    'reg_lambda_tilde': 1e-2,
    'kernel': 'gauss',
    'density': 'gaussian',
    'bandwidth': 1,
    'bandwidth_mode': 'auto',
    'epsilon': 0.3,
    'normalized': True
}

data_settings = {
    'expname': None,
    'data': 'uniform',
    'n_samples': None,
    'mediator_dimension': 1,
    'covariate_dimension': 1,
    'alpha': None,
    'beta': None,
    'stochasticity': None,
    'gamma': None,
    'noise': None,
}

metrics = {
        'bias_direct': None,
        'rmse_direct': None,
        'bias_indirect': None,
        'rmse_indirect': None,
        'bias_total': None,
        'rmse_total': None,
        'bias_mr': None,
        'rmse_mr': None,
        'coverage': None,
    }

task_pattern = ''
columns = []

for key in data_settings.keys():
    task_pattern += '{}:(.*)\|'.format(key)
    columns.append(key)

for key in param_settings.keys():
    task_pattern += '{}:(.*)\|'.format(key)
    columns.append(key)

metrics_pattern = ''
for key in metrics.keys():
    metrics_pattern += '{}:(.*)\|'.format(key)
    columns.append(key)

regex_pattern = '{} {}\n'.format(task_pattern, metrics_pattern)

EXPNAME = 'coverage_experiment'

def experiment(args):
    """
    Script to launch the experiment 

    """

    param_settings = {
        'estimator': args.estimator,
        'regularization': True,
        'sample_splits': 1,
        'reg_lambda': 1e-3,
        'reg_lambda_tilde': 1e-3,
        'kernel': 'gauss',
        'density': 'gaussian',
        'bandwidth': 0.3,
        'bandwidth_mode': args.bandwidth_mode,
        'epsilon': args.epsilon,
        'normalized': True,
        'random_seed': args.random_seed
    }

    data_settings = {
        'expname': EXPNAME,
        'data': 'uniform',
        'n_samples': args.n_samples,
        'mediator_dimension': 1,
        'covariate_dimension': 1,
        'alpha': args.alpha,
        'beta': args.beta,
        'stochasticity': args.stochasticity,
        'gamma': args.gamma,
        'noise': args.noise,
    }

    ### Print configuration
    display_experiment_configuration(data_settings, param_settings)

    ### Load causal environment
    causal_env = UniformEnv(data_settings)

    ### Load data
    causal_data = causal_env.generate_causal_data(data_settings, random_state=param_settings['random_seed'])
    x, t, m, y, params = causal_data

    estimator = get_estimator_by_name(param_settings)(param_settings, False)

    ### Fit nuisance parameters
    estimator.fit(t, m, x, y)

    ### Estimate causal effects
    causal_results = causal_env.causal_experiment(estimator, t, m, x, y, data_settings, params)

    ### Write results
    save_result(param_settings, causal_results, data_settings)

def sequential_experiments(args):
    """
    Script to launch the experiment 

    """

    sample_sizes = [500, 1000, 5000]

    for n_samples in sample_sizes:
        for rd in range(100):
            args.n_samples = n_samples
            args.random_seed = rd     
            experiment(args)

def get_tables(args):
    
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    df = preprocess(fd)

    # Group the dataframe by 'n_samples' and calculate the mean of the 'coverage' column
    average_coverage = df.groupby('n_samples')['coverage'].mean().reset_index()

    # Print the result to see the average coverage for each 'n_samples' value
    print(average_coverage)
     

def get_parameters_experiment(args):

    # Define additional parameters
    sample_sizes = [500, 1000, 5000]
    random_seeds = list(range(100))  # Random seeds 

    # Open or create a CSV file to write the values
    with open('coverage_experiment_parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each line with its number 
        line_number = 1
        for sample_size in sample_sizes:
            for seed in random_seeds:
                writer.writerow([line_number, sample_size, seed])
                line_number += 1


    print("The CSV file 'coverage_experiment_parameters.csv' has been created.")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run scripts for the '
                                                 'evaluation of causal '
                                                 'mediation methods')

    parser.add_argument('--sequential_run', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--get_parameters_experiment', action='store_true')
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.5)
    parser.add_argument('--beta', nargs="?", type=float, default=0.25)
    parser.add_argument('--stochasticity', nargs="?", type=float, default=1)
    parser.add_argument('--gamma', nargs="?", type=float, default=0.3)
    parser.add_argument('--bandwidth_mode', nargs="?",
                        default='heuristic', help='bandwidth method for kernel')
    parser.add_argument('--epsilon', nargs="?", type=float,
                        default=0.3, help='bandwidth method for kernel')
    parser.add_argument('--estimator', nargs="?", default='kme_dml',
                        choices=['ipw',
                                 'linear',
                                 'kme_g_computation',
                                 'kme_dml'], help='name of estimator')
    parser.add_argument('--n_samples', nargs="?", type=int,
                        default=1000, help='Number of samples')
    parser.add_argument('--random_seed', nargs="?", type=int,
                        default=42, help='random seed')
    parser.add_argument('--noise', nargs="?", default='uniform', choices=['uniform', 'gaussian'], help='nature of noise')

    args = parser.parse_args()

    if args.run:
        experiment(args)

    if args.sequential_run:
        sequential_experiments(args)

    if args.results:
        get_tables(args)
        
    if args.get_parameters_experiment:
        get_parameters_experiment(args)