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

EXPNAME = 'nonparametric_experiment'

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
        'kernel': args.kernel,
        'density': args.density,
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

    pass


def get_tables_mr(parameters):


    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    df = preprocess(fd)

    figures_dir = 'tables/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Define the estimators and their corresponding labels
    estimators = {
        'IPW': 'ipw',
        'KME': 'kme_g_computation',
        'DML': 'kme_dml',
    }

    # Define the metrics to be averaged
    metrics = ['bias_mr', 'rmse_mr']

    # Initialize an empty list to store summary rows
    summary_rows = []

    # Iterate over each estimator
    for estimator, col in estimators.items():
        # Filter the DataFrame for the current estimator
        estimator_df = df[df['estimator'] == col]

        # Group by 'n_samples' and calculate the mean and std for the specified metrics across 'random_seed'
        mean_values = estimator_df.groupby('n_samples')[metrics].mean().reset_index()
        std_values = estimator_df.groupby('n_samples')['bias_mr'].std().reset_index()

        # Merge mean and std values
        summary = pd.merge(mean_values, std_values, on='n_samples', suffixes=('', '_std'))

        # Rename the std columns
        summary.rename(columns={'bias_mr_std': 'std_mr'}, inplace=True)

        # Add the estimator label to the summary DataFrame
        summary.insert(0, 'Estimator', estimator)

        # Append the results to the summary_rows list
        summary_rows.append(summary)

    # Concatenate all summary DataFrames
    summary_df = pd.concat(summary_rows, ignore_index=True)

    # Reorder the columns
    ordered_columns = ['Estimator', 'n_samples', 'bias_mr', 'std_mr', 'rmse_mr']

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

        latex_filename = f"tables/mr_uniform_n_samples_{n_samples}.tex"
        df.to_latex(latex_filename, index=False)
        print(f"LaTeX table saved for n_samples = {n_samples}: {latex_filename}")


def get_tables_mr_with_multicolumns(parameters):
    # Load and preprocess data
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    df = preprocess(fd)

    # Output directory for tables
    figures_dir = 'tables/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Define the estimators and their corresponding labels
    estimators = {
        'IPW': 'ipw',
        'DML': 'kme_dml',
        'KME': 'kme_g_computation',
    }

    # Define densities and kernels combinations
    densities = ["gaussian", "conditional_kernel"]
    kernels = ["gauss", "linear"]

    # Define the metrics to include
    metrics = ['bias_mr', 'rmse_mr']

    # Initialize a dictionary to store results for each combination
    tables = {}

    for density in densities:
        for kernel in kernels:
            # Filter the DataFrame for the current density and kernel
            filtered_df = df[(df['density'] == density) & (df['kernel'] == kernel)]

            # Initialize a list to store summary rows
            summary_rows = []

            for estimator, col in estimators.items():
                # Filter the DataFrame for the current estimator
                estimator_df = filtered_df[filtered_df['estimator'] == col]

                # Group by 'n_samples' and calculate mean and std for metrics
                mean_values = estimator_df.groupby('n_samples')[metrics].mean().reset_index()
                std_values = estimator_df.groupby('n_samples')['bias_mr'].std().reset_index()

                # Merge mean and std values
                summary = pd.merge(mean_values, std_values, on='n_samples', suffixes=('', '_std'))

                # Rename the std column
                summary.rename(columns={'bias_mr_std': 'std_mr'}, inplace=True)

                # Add the estimator label to the summary DataFrame
                summary.insert(0, 'Estimator', estimator)

                # Append the results to the summary_rows list
                summary_rows.append(summary)

            # Concatenate all summary DataFrames
            summary_df = pd.concat(summary_rows, ignore_index=True)

            # Reorder the columns
            ordered_columns = ['Estimator', 'n_samples', 'bias_mr', 'std_mr', 'rmse_mr']
            summary_df = summary_df[ordered_columns]

            # Store the table in the dictionary
            tables[(density, kernel)] = summary_df

    # Generate a single LaTeX table with multicolumns
    with open(f"{figures_dir}/mr_summary_table.tex", 'w') as latex_file:
        latex_file.write("\\begin{table}[h]\n")
        latex_file.write("    \\centering\n")
        latex_file.write("\\caption{Performance Summary by Density and Kernel}\n")
        latex_file.write("\\begin{tabular}{llcccccccc}\n")
        latex_file.write("\\toprule\n")
        latex_file.write("Estimator & $n_\\text{samples}$ & ")

        # Add multicolumn headers
        densities_titles = ["gaussian", "Conditional-Kernel"]
        for density in densities_titles:
            for kernel in kernels:
                latex_file.write(f"\\multicolumn{{2}}{{c}}{{{density.capitalize()}-{kernel.capitalize()}}} ")
                if not (density == densities_titles[-1] and kernel == kernels[-1]):
                    latex_file.write("& ")
        latex_file.write("\\\\\n")

        # Add subheaders for Bias (std) and RMSE
        latex_file.write(" & & Bias (std) & RMSE & Bias (std) & RMSE & Bias (std) & RMSE & Bias (std) & RMSE \\\ \n")
        latex_file.write("\\midrule\n")

        # Write the table rows for each estimator and sample size
        for n_samples in sorted(df['n_samples'].unique()):
            for estimator in estimators.keys():
                latex_file.write(f"{estimator} & {n_samples} & ")
                for density in densities:
                    for kernel in kernels:
                        sub_df = tables[(density, kernel)]
                        row = sub_df[(sub_df['Estimator'] == estimator) & (sub_df['n_samples'] == n_samples)]
                        if not row.empty:
                            row = row.iloc[0]
                            latex_file.write(f"{row['bias_mr']:.4f} ({row['std_mr']:.4f}) & {row['rmse_mr']:.4f} ")
                        else:
                            latex_file.write("- & - ")
                        if not (density == densities[-1] and kernel == kernels[-1]):
                            latex_file.write("& ")
                latex_file.write("\\\\\n")

        latex_file.write("\\bottomrule\n")
        latex_file.write("\\end{tabular}\n")
        latex_file.write("\\label{tab:mr_summary_table}\n")
        latex_file.write("\\end{table}\n")

    print("LaTeX summary table saved at 'tables/mr_summary_table.tex'")

def get_parameters_experiment(args):

    # Define additional parameters
    estimators_list = ['ipw', 'kme_dml', 'kme_g_computation']
    sample_sizes = [500, 1000, 5000]
    densities = ["gaussian", "conditional_kernel"]
    kernels = ["gauss", "linear"]

    random_seeds = list(range(100))  # Random seeds 

    # Open or create a CSV file to write the values
    with open('nonparametric_experiment_parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each line with its number 
        line_number = 1

        for estimator in estimators_list:
            for sample_size in sample_sizes:
                for density in densities:
                    for kernel in kernels:
                        for seed in random_seeds:
                            writer.writerow([line_number, estimator, sample_size, density, kernel, seed])
                            line_number += 1

    print("The CSV file 'nonparametric_experiment_parameters.csv' has been created.")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run scripts for the '
                                                 'evaluation of causal '
                                                 'mediation methods')

    parser.add_argument('--sequential_run', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--get_parameters_experiment', action='store_true')
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.)
    parser.add_argument('--beta', nargs="?", type=float, default=0.)
    parser.add_argument('--stochasticity', nargs="?", type=float, default=1)
    parser.add_argument('--gamma', nargs="?", type=float, default=0.3)
    parser.add_argument('--bandwidth_mode', nargs="?",
                        default='heuristic', help='bandwidth method for kernel')
    parser.add_argument('--epsilon', nargs="?", 
                        default=0.1, help='bandwidth method for kernel')
    parser.add_argument('--estimator', nargs="?", default='ipw',
                        choices=['ipw',
                                 'linear',
                                 'kme_g_computation',
                                 'kme_dml',
                                 'sani_dml'], help='name of estimator')
    parser.add_argument('--n_samples', nargs="?", type=int,
                        default=1000, help='Number of samples')
    parser.add_argument('--random_seed', nargs="?", type=int,
                        default=42, help='random seed')
    parser.add_argument('--noise', nargs="?", default='uniform', choices=['uniform', 'gaussian'], help='nature of noise')
    parser.add_argument('--expname', nargs="?", default=EXPNAME, help='name of experiment')
    parser.add_argument('--kernel', nargs="?", default='gauss',
                        help='kernel choice')
    parser.add_argument('--density', nargs="?", default='gaussian',
                        help='kernel choice')
    args = parser.parse_args()

    if args.run:
        experiment(args)

    if args.sequential_run:
        sequential_experiments(args)

    if args.results:
        get_tables_mr_with_multicolumns(args)

    if args.get_parameters_experiment:
        get_parameters_experiment(args)
        