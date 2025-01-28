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
import seaborn as sns
import matplotlib.pyplot as plt

from utils.loader import get_estimator_by_name
from utils.utils import display_experiment_configuration 
from utils.parse import get_df, preprocess, save_result
from data.ukbb import UKBBEnv

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
    'data': 'ukbb',
    'n_samples': None,
    'd': None,
    'd_prime': None,
    'random_state': None
}

metrics = {
        'total_effect': None,
        'direct_effect': None,
        'indirect_effect': None,
        'mediated_response': None,
        'variance': None,
        'margin_error': None
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

EXPNAME = 'hb1ac_ukbb'

def experiment(args, param_settings=None, data_settings=None):
    """
    Script to launch the experiment 

    """
    if args:
        param_settings = {
            'estimator': args.estimator,
            'regularization': True,
            'sample_splits': 2,
            'reg_lambda': 1e-3,
            'reg_lambda_tilde': 1e-3,
            'kernel': 'gauss',
            'density': 'gaussian',
            'bandwidth': 1,
            'bandwidth_mode': args.bandwidth_mode,
            'epsilon': args.epsilon,
            'normalized': True
        }

        data_settings = {
            'expname': EXPNAME,
            'data': 'ukbb',
            'n_samples': args.n_samples,
            'd': args.d,
            'd_prime': args.d_prime,
            'random_state': args.random_seed
        }


    ### Print configuration
    display_experiment_configuration(data_settings, param_settings)

    ### Load causal environment
    causal_env = UKBBEnv(data_settings)

    ### Load data
    causal_data = causal_env.generate_causal_data(data_settings, random_state=data_settings['random_state'], treatment_strategy='hb1ac')
    x, t, m, y, params = causal_data

    estimator = get_estimator_by_name(param_settings)(param_settings, False)

    ### Fit nuisance parameters
    estimator.fit(t, m, x, y)

    ### Estimate causal effects
    causal_results = causal_env.causal_experiment(estimator, t, m, x, y, data_settings)

    ### Write results
    save_result(param_settings, causal_results, data_settings)

    return causal_results

def sequential_experiments(args):
    """
    Script to launch the experiment 

    """
    estimators_list = ['ipw', 'linear', 'kme_g_computation', 'kme_dml']
    d_list = np.arange(20, 50, 2)


    for estimator in estimators_list:
        for d in d_list:
            for rd in range(100):
                args.estimator = estimator
                args.d = d
                args.random_seed = rd     
                experiment(args)



def get_results(args):
    """
    Script to launch the experiment 

    """
    fd = get_df(regex_pattern, columns, results_file=f'results/{EXPNAME}/')
    fd = preprocess(fd)

    figures_dir = 'plots/ukbb/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Define the metrics to plot
    metrics = ['total_effect', 'indirect_effect']

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
    df = fd.copy()

    # Map the estimator values to their new labels
    df['estimator'] = df['estimator'].map(label_mapping)

    # Custom titles for the metrics
    metric_titles = {
        'total_effect': 'Total Effect',
        'indirect_effect': 'Indirect Effect'
    }

    # Loop through only the selected metrics
    for metric in metrics:
        # Create a new figure for each metric
        # Set the context and style for the plot
        sns.set_style('whitegrid')  # Set style first
        plt.figure(figsize=(20, 12))   # Increased size for better readability
        
        for estimator in df['estimator'].unique():
            df_estimator = df[df['estimator'] == estimator]

            # Group by 'd' and calculate the mean and standard deviation across 'random_state'
            grouped = df_estimator.groupby('d').agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index()

            # Plot the mean with fill_between for the error bars (mean +/- std)
            plt.plot(grouped['d'], grouped['mean_metric'], label=f'{estimator}', lw=3)
            plt.fill_between(grouped['d'], 
                            grouped['mean_metric'] - grouped['std_metric'], 
                            grouped['mean_metric'] + grouped['std_metric'], 
                            alpha=0.2)

        # Add labels and titles with your font sizes and bold settings
        plt.xlabel('Treatment values', fontsize=label_fontsize, weight='bold')
        plt.ylabel(f'HbA1c (mmol/mol)', fontsize=label_fontsize, weight='bold')
        plt.title(f'{metric_titles[metric]} Across Estimators', fontsize=title_fontsize, weight='bold')
        
        # Customize the ticks
        plt.xticks(fontsize=tick_fontsize, weight='bold')
        plt.yticks(fontsize=tick_fontsize)

        # Customize the legend
        plt.legend(loc='upper left', fontsize=legend_fontsize, title='Estimator', title_fontsize=legend_fontsize, frameon=True)

        # Enable grid
        plt.grid(True)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.savefig('plots/ukbb/{}_plot.png'.format(metric), bbox_inches='tight', pad_inches=0)
        plt.show()

    # Set the context and style for the plot
    sns.set_style('whitegrid')  # Set style first
    # Create a larger figure for the combined plot
    plt.figure(figsize=(20, 12))  # Increased size for your grand visualization
    metric = 'mediated_response'
    df = fd.copy()
    # Filter the dataframe for the kme_dml estimator
    df_estimator = df[df['estimator'] == 'kme_dml']

    # Group by 'd' and calculate the mean, 2.5th percentile, 97.5th percentile, and margin error across 'random_state'
    grouped = df_estimator.groupby('d').agg(
        mean_metric=(metric, 'mean'),
        lower_percentile=(metric, lambda x: np.percentile(x, 2.5)),
        upper_percentile=(metric, lambda x: np.percentile(x, 97.5)),
        mean_margin_error=('margin_error', 'mean')
    ).reset_index()

    # Plot 1: Mediated Response with 95% Confidence Interval (2.5th and 97.5th percentiles)
    plt.plot(grouped['d'], grouped['mean_metric'], label=f'Bootstrap CI', color='blue', lw=3)
    plt.fill_between(grouped['d'], 
                    grouped['lower_percentile'], 
                    grouped['upper_percentile'], 
                    alpha=0.2, color='blue')

    # Plot 2: Mediated Response with Margin Error Bars
    plt.plot(grouped['d'], grouped['mean_metric'], label=f'Asymptotic CI', color='green', linestyle='--', lw=3)
    plt.fill_between(grouped['d'], 
                    grouped['mean_metric'] - grouped['mean_margin_error'], 
                    grouped['mean_metric'] + grouped['mean_margin_error'], 
                    alpha=0.2, color='green')

    # Add labels, legend, and title with larger font sizes and bold text
    plt.xlabel('Treatment values', fontsize=label_fontsize, weight='bold')
    plt.ylabel(f'Mediated Response values', fontsize=label_fontsize, weight='bold')
    plt.title(f'Mediated Response with 95% Confidence Interval (CI)', fontsize=title_fontsize, weight='bold')

    # Customize the ticks
    plt.xticks(fontsize=tick_fontsize, weight='bold')
    plt.yticks(fontsize=tick_fontsize)

    # Customize the legend
    plt.legend(loc='upper right', fontsize=legend_fontsize, title_fontsize=legend_fontsize, frameon=True)

    # Enable grid
    plt.grid(True)

    # Adjust layout and display the combined plot
    plt.tight_layout()
    plt.savefig('plots/ukbb/mr_confidence_intervals.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def get_parameters_experiment(args):

    # Define additional parameters
    estimators_list = ['linear', 'kme_g_computation', 'ipw', 'kme_dml']
    d_list = np.arange(20, 50, 2)
    random_states = list(range(100))  # Random seeds  


    # Open or create a CSV file to write the values
    name = 'ukbb_experiment_parameters.csv'
    with open(name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each line with its number and corresponding alpha, beta values
        line_number = 1
        for estimator in estimators_list:
            for d in d_list:
                for seed in random_states:
                    writer.writerow([line_number, estimator, d, seed])
                    line_number += 1
                    

    print("The CSV file {} has created".format(name))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run scripts for the '
                                                 'evaluation of causal '
                                                 'mediation methods')

    parser.add_argument('--sequential_run', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--get_parameters_experiment', action='store_true')
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--d', nargs="?", type=float, default=0.5)
    parser.add_argument('--d_prime', nargs="?", type=float, default=40)
    parser.add_argument('--estimator', nargs="?", default='ipw',
                        choices=['ipw',
                                 'linear',
                                 'kme_g_computation',
                                 'kme_dml'], help='name of estimator')
    parser.add_argument('--n_samples', nargs="?", type=int,
                        default=1000, help='Number of samples')
    parser.add_argument('--bandwidth_mode', nargs="?",
                        default='auto', help='bandwidth method for kernel')
    parser.add_argument('--epsilon', nargs="?", type=float,
                        default=0.1, help='bandwidth method for kernel')
    parser.add_argument('--random_seed', nargs="?", type=int,
                        default=42, help='random seed')
    args = parser.parse_args()

    if args.run:
        experiment(args)

    if args.sequential_run:
        sequential_experiments(args)

    if args.results:
        get_results(args)

    if args.get_parameters_experiment:
        get_parameters_experiment(args)
        