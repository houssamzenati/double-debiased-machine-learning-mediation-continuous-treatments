# Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import matplotlib.colors as mcolors
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import os
from os import listdir
from os.path import isfile, join
from os import walk
import pandas as pd
import re

from datetime import date
today = date.today()

def save_result(param_settings, metrics, data_settings):
    """
    Saves results in .txt file

    Parameters
    ----------
    param_settings : dictionary
        contains hyperparameter values
    
    metrics : dictionary 
        contains metrics to evaluate the experiment

    data_settings : dictionary
        contains data parameters
    """

    task_name = ""

    for key in data_settings.keys():
        task_name += f'{key}:{data_settings[key]}|'

    for key in param_settings.keys():
        task_name += f'{key}:{param_settings[key]}|'

    metrics_information = ''
    for key in metrics.keys():
        metrics_information += f'{key}:{metrics[key]}|'

    result = f'{task_name} {metrics_information}\n'
    results_dir = f"results/{data_settings['expname']}/{data_settings['data']}/{today.strftime('%d-%m-%Y')}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, 'metrics.txt')

    with open(fname, 'a') as file:
        file.write(result)

def preprocess(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df.dropna()

param_settings = {
    'estimator': None,
    'regularization': None,
    'sample_splitting': None,
    'reg_lambda': None,
    'reg_lambda_tilde':None,
    'kernel': None,
    'density': None,
    'bandwidth': None,
    'normalized': None,
    'random_seed': None
}

data_settings = {
    'expname': None,
    'data': None,
    'n_samples': None,
    'mediator_dimension': None,
    'covariate_dimension': None,
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
# %%

def create_df(fname, regex_pattern, columns):
    rgx = re.compile(regex_pattern, flags=re.M)
    with open(fname, 'r') as f:
        lines = rgx.findall(f.read())
        df = pd.DataFrame(lines, columns=columns)

        columns_str = ['algo']
        columns_flt_int = [item for item in columns if item not in columns_str]
        for col in columns_flt_int:
            try:
                df[col] = df[col].astype(np.int32)
            except ValueError:
                df[col] = df[col].astype(np.float32, errors='ignore')
    return df



def get_df(regex_pattern, columns, results_file='results', select_file=None):
    paths = []
    for (dirpath, dirnames, filenames) in walk(results_file):
        print(dirpath, dirnames, filenames)
        if select_file:
            condition = filenames != [] and select_file in dirpath
        else:
            condition = filenames != []
        condition = (filenames == ['metrics.txt'])
        if condition:
            paths.extend([os.path.join(dirpath, 'metrics.txt')])

    print(paths)
#     regex_pattern = 'algo:(.*)\|mu:(.*)\|lambda:(.*)\|C:(.*)\|bonus:(.*)\|rd:(.*)\|horizon:(.*) average_reward:(.*)\|total_time:(.*)'
#     columns = ['algo', 'mu', 'lambda', 'C', 'bonus', 'rd', 'horizon', 'average_reward', 'total_time']

    # regex_pattern = 'algo:(.*)\|mu:(.*)\|lambda:(.*)\|C:(.*)\|beta:(.*)\|rd:(.*)\|kernel:(.*)\|horizon:(.*)\|env:(.*) average_reward:(.*)\|regret:(.*)\|total_time:(.*)'
    # columns = ['algo', 'mu', 'lambda', 'C', 'beta', 'rd', 'kernel', 'horizon', 'env', 'average_reward', 'regret', 'total_time']

    list_dfs = []
    for path in paths:
        list_dfs += [create_df(path, regex_pattern, columns)]

    df = pd.concat(list_dfs, ignore_index=True)

    return df


# %%
labels=['g_computation', 'ipw', 'linear', 'dml', 'mr', 'kme_g_computation', 'kme_dml']


def plot_result(df, labels, mediator_setting, path=None):
    df['score'] = abs(df['direct_effect_treated']-df['true_direct_treated']) + abs(df['indirect_effect_control']-df['true_indirect_control'])

    l = list(param_settings.keys()) + list(data_settings.keys())
    l.remove('random_seed')
    l.remove('expname')
    l.remove('data')
    
    # table
    grouped = df.groupby(l).mean().reset_index()
    index_to_aggregate = ['n_samples', 'mediator_dimension', 'covariate_dimension', 'estimator', 'kernel']
    grouped_ = grouped.groupby(index_to_aggregate)
    
    min_index = grouped_['score'].idxmin()
    
    table = grouped.loc[min_index]

    
    list_params = list(param_settings.keys())
    list_params.remove('random_seed')
    list_params.remove('estimator')
    
    # Group by the hyperparameters
    grouped = df.groupby(l)

    # Calculate the mean and std for the metrics
    aggregated_df = grouped.agg({
        'error_total': ['mean', 'std'],
        'error_direct': ['mean', 'std'],
        'error_indirect': ['mean', 'std'],
        'relative_error_total': ['mean', 'std'],
        'relative_error_direct': ['mean', 'std'],
        'relative_error_indirect': ['mean', 'std'],
    }).reset_index()

    # Flatten the MultiIndex columns
    aggregated_df.columns = ['_'.join(col).strip('_') for col in aggregated_df.columns.values]
    
    # total effects
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].errorbar(aggregated_df['estimator'], aggregated_df['error_direct_mean'], 
                yerr=aggregated_df['error_direct_std'], fmt='o', capsize=5, marker='o')

    ax[0].set_xlabel('Estimator')
    ax[0].set_ylabel('Metric')
    ax[0].set_title('Mean Direct Effect Error')
    ax[0].set_xticks(range(len(aggregated_df['estimator'])))
    ax[0].set_xticklabels(aggregated_df['estimator'])

    ax[1].errorbar(aggregated_df['estimator'], aggregated_df['error_indirect_mean'], 
                yerr=aggregated_df['error_indirect_std'], fmt='o', capsize=5, marker='o')
    ax[1].set_xlabel('Estimator')
    ax[1].set_ylabel('Metric')
    ax[1].set_title('Mean Indirect Effect Error')
    ax[1].set_xticks(range(len(aggregated_df['estimator'])))
    ax[1].set_xticklabels(aggregated_df['estimator'])
    
    if path:
        plt.savefig(path)
    else:
        plt.savefig('figures/continous_mediators_{}_n10000.jpg'.format(mediator_setting))
    # plt.show()
    
    summary_df = table.groupby(['estimator', 'random_seed'])['score'].mean().reset_index()
    
    # Calculate the mean score per estimator across all random seeds
    summary_df = summary_df.groupby('estimator')['score'].mean().reset_index().sort_values(by='score', ascending=True)
    return summary_df
