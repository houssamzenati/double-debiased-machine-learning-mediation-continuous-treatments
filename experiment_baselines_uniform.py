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

EXPNAME = 'baselines_experiment'

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

    estimators_list = ['ipw', 'linear', 'kme_g_computation', 'kme_dml']
    sample_sizes = [500, 1000, 5000]


    for estimator in estimators_list:
        for n_samples in sample_sizes:
            for rd in range(100):
                args.estimator = estimator
                args.n_samples = n_samples
                args.random_seed = rd     
                experiment(args)


def get_boxtplot(args):
    """
    Script to launch the experiment 

    """
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    fd = preprocess(fd)


    figures_dir = 'boxplots/uniform/baselines/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    df = fd.copy()
    
    # Define font sizes
    title_fontsize = 50
    label_fontsize = 40
    legend_fontsize = 40
    tick_fontsize = 40

    # Define a mapping for the estimator labels
    label_mapping = {
        "linear": "OLS",
        "ipw": "IPW",
        "kme_g_computation": "KME",
        "kme_dml": "DML"
    }

    # Map the estimator values to their new labels
    df['estimator'] = df['estimator'].map(label_mapping)

    # Set figure size and style
    plt.figure(figsize=(16, 10))  # Increased size for better readability
    sns.set_style("whitegrid") 

    # Boxplot for Mediated Response with custom options
    boxplot = sns.boxplot(x='n_samples', y='rmse_mr', hue='estimator', data=df)

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
    plt.savefig('boxplots/uniform/baselines/mr_rmse_boxplot.png', bbox_inches='tight', pad_inches=0)


    # Set figure size and style
    plt.figure(figsize=(16, 10))  # Increased size for better readability
    sns.set_style("whitegrid")  

    # Boxplot for Mediated Response with custom options
    boxplot = sns.boxplot(x='n_samples', y='bias_mr', hue='estimator', data=df)

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
    plt.savefig('boxplots/uniform/baselines/mr_bias_boxplot.png',
    bbox_inches='tight', pad_inches=0)

                                                                              
def get_tables(args):
    
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    df = preprocess(fd)

    figures_dir = 'tables/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Define the estimators and their corresponding labels
    estimators = {
        'OLS': 'linear',
        'IPW': 'ipw',
        'KME': 'kme_g_computation',
        'DML': 'kme_dml'
    }

    # Define the metrics to be averaged
    metrics = ['bias_direct', 'rmse_direct', 'bias_indirect', 'rmse_indirect', 'bias_total', 'rmse_total']

    # Initialize an empty list to store summary rows
    summary_rows = []

    # Iterate over each estimator
    for estimator, col in estimators.items():
        # Filter the DataFrame for the current estimator
        estimator_df = df[df['estimator'] == col]
        
        # Group by 'n_samples' and calculate the mean and std for the specified metrics across 'random_seed'
        mean_values = estimator_df.groupby('n_samples')[metrics].mean().reset_index()
        std_values = estimator_df.groupby('n_samples')[['bias_direct', 'bias_indirect', 'bias_total']].std().reset_index()
        
        # Merge mean and std values
        summary = pd.merge(mean_values, std_values, on='n_samples', suffixes=('', '_std'))
        
        # Rename the std columns
        summary.rename(columns={
            'bias_direct_std': 'std_direct',
            'bias_indirect_std': 'std_indirect',
            'bias_total_std': 'std_total'
        }, inplace=True)

        # Add the estimator label to the summary DataFrame
        summary.insert(0, 'Estimator', estimator)
        
        # Append the results to the summary_rows list
        summary_rows.append(summary)

    # Concatenate all summary DataFrames
    summary_df = pd.concat(summary_rows, ignore_index=True)

    # Reorder the columns as per your request: for each category, the order is bias, std, rmse
    ordered_columns = ['Estimator', 'n_samples', 
                    'bias_direct', 'std_direct', 'rmse_direct', 
                    'bias_indirect', 'std_indirect', 'rmse_indirect', 
                    'bias_total', 'std_total', 'rmse_total']

    # Reorder the columns in the DataFrame
    summary_df = summary_df[ordered_columns]

    # Get unique values of n_samples
    n_samples_values = summary_df['n_samples'].unique()

    # Separate into different DataFrames for each n_samples value
    summary_dfs = {n: summary_df[summary_df['n_samples'] == n].reset_index(drop=True) for n in n_samples_values}

    # Save each summary DataFrame to a LaTeX table
    for n_samples, df in summary_dfs.items():

        # Remove the 'n_samples' column before saving
        df = df.drop(columns=['n_samples'])
        df = df.round(4)

        latex_filename = f"tables/uniform_n_samples_{n_samples}.tex"
        df.to_latex(latex_filename, index=False)
        print(f"LaTeX table saved for n_samples = {n_samples}: {latex_filename}")
     
def get_parameters_experiment(args):

    # Define additional parameters
    estimators_list = ['ipw', 'linear', 'kme_g_computation', 'kme_dml']
    sample_sizes = [500, 1000, 5000]
    random_seeds = list(range(100))  # Random seeds 

    # Open or create a CSV file to write the values
    with open('experiment_parameters/baselines_experiment_parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each line with its number 
        line_number = 1

        for estimator in estimators_list:
            for sample_size in sample_sizes:
                for seed in random_seeds:
                    writer.writerow([line_number, estimator, sample_size, seed])
                    line_number += 1

    print("The CSV file 'experiment_parameters/baselines_experiment_parameters.csv' has been created.")
    

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
    parser.add_argument('--expname', nargs="?", default=EXPNAME, help='name of experiment')

    args = parser.parse_args()

    if args.run:
        experiment(args)

    if args.sequential_run:
        sequential_experiments(args)

    if args.results:
        get_boxtplot(args)
        get_tables(args)

    if args.get_parameters_experiment:
        get_parameters_experiment(args)
        