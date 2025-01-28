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

from utils.parse import get_df, preprocess, plot_result, regex_pattern, columns, save_result
from run import single_run, experiment


EXPNAME = 'gaussian_huber'

def experiment(args):
    """
    Script to launch the experiment 

    """

    param_settings = {
        'estimator': args.estimator,
        'regularization': True,
        'sample_splitting': False,
        'reg_lambda': 1e-3,
        'reg_lambda_tilde': 1e-3,
        'kernel': 'gauss',
        'density': 'gaussian',
        'bandwidth': 0.3,
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

    single_run(param_settings, data_settings)


def sequential_experiments(args):
    """
    Script to launch the experiment 

    """

    estimators_list = ['ipw', 'linear', 'kme_g_computation', 'kme_dml']
    sample_sizes = [300, 1000, 3000]

    results = pd.DataFrame()

    for estimator in estimators_list:
        for n_samples in sample_sizes:

            param_settings = {
                'estimator': estimator,
                'regularization': True,
                'sample_splitting': False,
                'reg_lambda': 1e-3,
                'reg_lambda_tilde': 1e-3,
                'kernel': 'gauss',
                'density': 'gaussian',
                'bandwidth': 0.3,
                'normalized': True,
            }

            data_settings = {
                'expname': EXPNAME,
                'data': 'uniform',
                'n_samples': n_samples,
                'mediator_dimension': 1,
                'covariate_dimension': 1,
                'alpha': args.alpha,
                'beta': args.beta,
                'stochasticity': args.stochasticity,
                'gamma': args.gamma,
                'noise': args.noise,
            }

            args = None
            metrics = experiment(args, param_settings, data_settings, 20)
            results = pd.concat([results, metrics], axis=0)


    # Create a boxplot using seaborn
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='n_samples', y='rmse_direct', hue='estimator', data=results)
    plt.yscale('log')  # Optional: Use log scale for MSE to better visualize differences
    plt.title('Performance of Different Estimators by Training Sample Size')
    plt.ylabel('Root Mean Squared Error')
    plt.xlabel('Training Sample Size')
    plt.legend(title='Estimator')
    plt.savefig('boxplots/boxplot_alpha_{}_beta_{}.png') 

    return results 

def get_results(args):
    """
    Script to launch the experiment 

    """
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    fd = preprocess(fd)

    stochasticity_values = [0.1, 1]  # Example values from 0.0 to 0.9
    gamma_values = [0.3]   # Example values from 0.0 to 1.8
    noise_values = ['uniform', 'gaussian']   # Example values from 0.0 to 1.8

    for s in stochasticity_values:
        for gamma in gamma_values:
            for noise in noise_values:

                df = fd.copy()
                df = df[df.noise == noise]
                df = df[df.stochasticity == s]
                df = df[df.gamma == gamma]
                
                figures_dir = 'boxplots/{}/'.format(noise)
                if not os.path.exists(figures_dir):
                    os.makedirs(figures_dir)

                # Define font sizes
                title_fontsize = 18
                label_fontsize = 16
                legend_fontsize = 14

                # Plot for Mediated Response
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='n_samples', y='rmse_mr', hue='estimator', data=df)
                plt.yscale('log')  # Optional: Use log scale for better visualization
                plt.title('Mediated Response', fontsize=title_fontsize)
                plt.ylabel('Root Mean Squared Error', fontsize=label_fontsize)
                plt.xlabel('Training Sample Size', fontsize=label_fontsize)
                plt.legend(title='Estimator', title_fontsize=legend_fontsize, fontsize=legend_fontsize)
                plt.xticks(fontsize=label_fontsize)
                plt.yticks(fontsize=label_fontsize)
                plt.tight_layout(pad=0)  # Adjust layout to remove padding
                plt.savefig('boxplots/{}/mr_boxplot_s_{}_gamma_{}.png'.format(noise, s, gamma))

                # Plot for Direct Effect
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='n_samples', y='rmse_direct', hue='estimator', data=df)
                plt.yscale('log')  # Optional: Use log scale for better visualization
                plt.title('Direct Effect', fontsize=title_fontsize)
                plt.ylabel('Root Mean Squared Error', fontsize=label_fontsize)
                plt.xlabel('Training Sample Size', fontsize=label_fontsize)
                plt.legend(title='Estimator', title_fontsize=legend_fontsize, fontsize=legend_fontsize)
                plt.xticks(fontsize=label_fontsize)
                plt.yticks(fontsize=label_fontsize)
                plt.tight_layout(pad=0)  # Adjust layout to remove padding
                plt.savefig('boxplots/{}/direct_boxplot_s_{}_gamma_{}.png'.format(noise, s, gamma))

                # Plot for Indirect Effect
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='n_samples', y='rmse_indirect', hue='estimator', data=df)
                plt.yscale('log')  # Optional: Use log scale for better visualization
                plt.title('Indirect Effect', fontsize=title_fontsize)
                plt.ylabel('Root Mean Squared Error', fontsize=label_fontsize)
                plt.xlabel('Training Sample Size', fontsize=label_fontsize)
                plt.legend(title='Estimator', title_fontsize=legend_fontsize, fontsize=legend_fontsize)
                plt.xticks(fontsize=label_fontsize)
                plt.yticks(fontsize=label_fontsize)
                plt.tight_layout(pad=0)  # Adjust layout to remove padding
                plt.savefig('boxplots/{}/indirect_boxplot_s_{}_gamma_{}.png'.format(noise, s, gamma))


def get_parameters_arguments_csv(args):

    stochasticity_values = [0.1, 1]  # Example values from 0.0 to 0.9
    gamma_values = [0.3]   # Example values from 0.0 to 1.8
    noise_values = ['uniform']   # Example values from 0.0 to 1.8
    # Define additional parameters
    estimators_list = ['ipw', 'linear', 'kme_g_computation', 'kme_dml']
    sample_sizes = [300, 1000, 3000]
    random_seeds = list(range(20))  # Random seeds 

    # Open or create a CSV file to write the values
    with open('parameters_{}.csv'.format(EXPNAME), mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        # writer.writerow(['Line Number', 'stochasticity', 'gamma',
        # 'estimator', 'sample_size', 'random_seed', 'noise'])

        # Write each line with its number and corresponding alpha, beta values
        line_number = 1
        for s in stochasticity_values:
            for gamma in gamma_values:
                for estimator in estimators_list:
                    for noise in noise_values:
                        for sample_size in sample_sizes:
                            for seed in random_seeds:
                                writer.writerow([line_number, s, gamma, estimator, noise, sample_size, seed])
                                line_number += 1

    print("The CSV file 'parameters_{}.csv' has been filled with the number of lines and alpha, beta values.".format(EXPNAME))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run scripts for the '
                                                 'evaluation of causal '
                                                 'mediation methods')

    parser.add_argument('--sequential_run', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--generate_slurm_experiments', action='store_true')
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.5)
    parser.add_argument('--beta', nargs="?", type=float, default=0.25)
    parser.add_argument('--stochasticity', nargs="?", type=float, default=1)
    parser.add_argument('--gamma', nargs="?", type=float, default=0.3)
    parser.add_argument('--estimator', nargs="?", default='ipw',
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
        get_results(args)

    if args.generate_slurm_experiments:
        get_parameters_arguments_csv(args)
        