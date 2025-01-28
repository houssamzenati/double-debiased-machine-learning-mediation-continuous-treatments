import os
import sys
import csv

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

import numpy as np
import os
from datetime import date
today = date.today()
import argparse
import pandas as pd
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
        # 'coverage': None,
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

EXPNAME = 'huber_experiment'

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

    epsilon_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] 
    bandwidth_modes = ['heuristic', 'amse']  
    # Define additional parameters
    sample_sizes = [500, 1000, 5000, 10000]


    for epsilon in epsilon_values:
        for bandwdith in bandwidth_modes:
            for n_samples in sample_sizes:
                for rd in range(100):
                    args.epsilon = epsilon
                    args.bandwidth_mode = bandwdith
                    args.n_samples = n_samples
                    args.random_seed = rd     
                    experiment(args)

def get_boxtplot(args):
    """
    Script to launch the experiment 

    """
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    fd = preprocess(fd)

    epsilon_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]   # Example

    for epsilon in epsilon_values:

        df = fd.copy()
        df = df[df.estimator == 'kme_dml']
        mask = ((df['bandwidth_mode'] == 'amse') & (abs(df['epsilon'] - epsilon) < 1e-2)) | (df['bandwidth_mode'] == 'heuristic')
        df = df[mask]
        mask = (df['n_samples'] == 500) | (df['n_samples'] == 1000) | (df['n_samples'] == 5000) | (df['n_samples'] == 10000)
        df = df[mask]
        
        figures_dir = 'boxplots/noise/bandwidths/'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        # Define font sizes
        title_fontsize = 50
        label_fontsize = 40
        legend_fontsize = 40
        tick_fontsize = 40

        # Set figure size and style
        plt.figure(figsize=(16, 10))  # Increased size for better readability
        sns.set_style("whitegrid") 

        # Boxplot for Mediated Response with custom options
        boxplot = sns.boxplot(x='n_samples', y='rmse_mr', hue='bandwidth_mode', data=df)

        # Optional: Use log scale for better visualization
        plt.yscale('log')

        # Title, labels, and legend adjustments
        plt.title('Mediated Response', fontsize=title_fontsize, weight='bold')
        plt.ylabel('Root Mean Squared Error', fontsize=label_fontsize, weight='bold')
        plt.xlabel('Training Sample Size', fontsize=label_fontsize, weight='bold')

        # Customizing ticks
        plt.xticks(fontsize=tick_fontsize, weight='bold')
        plt.yticks(fontsize=tick_fontsize)

        # Add vertical dashed lines between each n_samples category
        # Get unique n_samples values from the data
        n_samples_values = sorted(df['n_samples'].unique())
        for i in range(1, len(n_samples_values)):
            plt.axvline(x=i - 0.5, color='lightgrey', linestyle='--', lw=2)  # Lighter grey dashed lines

        # Add horizontal lines at 1e-1 and 1e-2
        plt.axhline(y=1e-1, color='lightgrey', linestyle='--', lw=2)
        plt.axhline(y=1e-2, color='lightgrey', linestyle='--', lw=2)

        # Move the legend to the bottom
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), 
                ncol=4, title='Estimator', title_fontsize=legend_fontsize, fontsize=legend_fontsize, frameon=True)

        # Optional: Adding a reference horizontal line
        plt.axhline(y=0, color="black", lw=4)

        # Adjust layout to remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

        # Save the figure with tight bounding box to remove whitespace
        plt.savefig('boxplots/noise/bandwidths/mr_rmse_boxplot_epsilon_{}.png'.format(epsilon), bbox_inches='tight', pad_inches=0)

        # Set figure size and style
        plt.figure(figsize=(16, 10))  # Increased size for better readability
        sns.set_style("whitegrid") 

        # Boxplot for Mediated Response with custom options
        boxplot = sns.boxplot(x='n_samples', y='bias_mr', hue='bandwidth_mode', data=df)

        # Optional: Use log scale for better visualization
        plt.yscale('log')

        # Title, labels, and legend adjustments
        plt.title('Mediated Response', fontsize=title_fontsize, weight='bold')
        plt.ylabel('Bias', fontsize=label_fontsize, weight='bold')
        plt.xlabel('Training Sample Size', fontsize=label_fontsize, weight='bold')

        # Customizing ticks
        plt.xticks(fontsize=tick_fontsize, weight='bold')
        plt.yticks(fontsize=tick_fontsize)

        # Add vertical dashed lines between each n_samples category
        # Get unique n_samples values from the data
        n_samples_values = sorted(df['n_samples'].unique())
        for i in range(1, len(n_samples_values)):
            plt.axvline(x=i - 0.5, color='lightgrey', linestyle='--', lw=2)  # Lighter grey dashed lines

        # Add horizontal lines at 1e-1 and 1e-2
        plt.axhline(y=1e-1, color='lightgrey', linestyle='--', lw=2)
        plt.axhline(y=1e-2, color='lightgrey', linestyle='--', lw=2)

        # Move the legend to the bottom
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), 
                ncol=4, title='Estimator', title_fontsize=legend_fontsize, fontsize=legend_fontsize, frameon=True)

        # Optional: Adding a reference horizontal line
        plt.axhline(y=0, color="black", lw=4)

        # Adjust layout to remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

        # Save the figure with tight bounding box to remove whitespace
        plt.savefig('boxplots/noise/bandwidths/mr_bias_boxplot_epsilon_{}.png'.format(epsilon), bbox_inches='tight', pad_inches=0)
                                                                              


def get_parameters_experiment(args):

    epsilon_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]   # Example
    bandwidth_modes = ['heuristic', 'amse']   # Example
    # Define additional parameters
    sample_sizes = [500, 1000, 5000, 10000]
    random_seeds = list(range(100))  # Random seeds 
    experiment_name = 'bandwidth_experiment'

    # Open or create a CSV file to write the values
    with open('bandwidth_experiment_parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each line with its number 
        line_number = 1
        for sample_size in sample_sizes:
            for seed in random_seeds:
                for mode in bandwidth_modes:
                    if mode == 'amse':
                        for epsilon in epsilon_values:
                            writer.writerow([line_number, experiment_name, epsilon, mode, sample_size, seed])
                            line_number += 1
                    else:
                        writer.writerow([line_number, experiment_name, 0, mode, sample_size, seed])
                        line_number += 1


    print("The CSV file 'bandwidth_experiment_parameters.csv' has been created.")
    

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
                        default='amse', help='bandwidth method for kernel')
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
    parser.add_argument('--expname', nargs="?", default=EXPNAME, help='name of experiment')

    args = parser.parse_args()

    if args.run:
        experiment(args)

    if args.sequential_run:
        sequential_experiments(args)

    if args.results:
        get_boxtplot(args)
        
    if args.get_parameters_experiment:
        get_parameters_experiment(args)