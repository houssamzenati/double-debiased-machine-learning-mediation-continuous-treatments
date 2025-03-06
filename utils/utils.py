import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)


from sklearn.cluster import KMeans



from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np


def is_array_integer(array):
    if array.shape[1]>1:
        return False
    return all(list((array == array.astype(int)).squeeze()))

def str_to_bool(string):
    if bool(string) == string:
        return string
    elif string == 'True':
         return True
    elif string == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


def bucketize_mediators(m, n_buckets=10, random_state=42):
    kmeans = KMeans(n_clusters=n_buckets, random_state=random_state, n_init="auto").fit(m)
    return kmeans.predict(m)


def split_data(causal_data, nuisances, test_size=0.5, random_state=42):
    x, t, m, y = causal_data
    x_train, x_test, t_train, t_test, m_train, m_test, y_train, y_test = (
        train_test_split(x,
                         t,
                         m,
                         y,
                         test_size=test_size,
                         random_state=random_state))
    causal_data_nuisance = x_train, t_train, m_train, y_train
    causal_data_estimation = x_test, t_test, m_test, y_test

    return causal_data_nuisance, causal_data_estimation


# def _get_train_test_lists(crossfit, n, x):
#     """
#     Obtain train and test folds

#     Returns
#     -------
#     train_test_list : list
#         indexes with train and test indexes
#     """
#     if crossfit < 2:
#         train_test_list = [[np.arange(n), np.arange(n)]]
#     else:
#         kf = KFold(n_splits=crossfit)
#         train_test_list = list()
#         for train_index, test_index in kf.split(x):
#             train_test_list.append([train_index, test_index])
#     return train_test_list

def _get_interactions(interaction, *args):
    """
    this function provides interaction terms between different groups of
    variables (confounders, treatment, mediators)

    Parameters
    ----------
    interaction : boolean
                    whether to compute interaction terms

    *args : flexible, one or several arrays
                    blocks of variables between which interactions should be
                    computed


    Returns
    --------
    array_like
        interaction terms

    Examples
    --------
    >>> x = np.arange(6).reshape(3, 2)
    >>> t = np.ones((3, 1))
    >>> m = 2 * np.ones((3, 1))
    >>> get_interactions(False, x, t, m)
    array([[0., 1., 1., 2.],
           [2., 3., 1., 2.],
           [4., 5., 1., 2.]])
    >>> get_interactions(True, x, t, m)
    array([[ 0.,  1.,  1.,  2.,  0.,  1.,  0.,  2.,  2.],
           [ 2.,  3.,  1.,  2.,  2.,  3.,  4.,  6.,  2.],
           [ 4.,  5.,  1.,  2.,  4.,  5.,  8., 10.,  2.]])
    """
    variables = list(args)
    for index, var in enumerate(variables):
        if len(var.shape) == 1:
            variables[index] = var.reshape(-1,1)
    pre_inter_variables = np.hstack(variables)
    if not interaction:
        return pre_inter_variables
    new_cols = list()
    for i, var in enumerate(variables[:]):
        for j, var2 in enumerate(variables[i+1:]):
            for ii in range(var.shape[1]):
                for jj in range(var2.shape[1]):
                    new_cols.append((var[:, ii] * var2[:, jj]).reshape(-1, 1))
    new_vars = np.hstack(new_cols)
    result = np.hstack((pre_inter_variables, new_vars))
    return result



def display_experiment_configuration(data_settings, param_settings):
    experiment_message = 'Experiment: '
    for key in data_settings.keys():
        experiment_message += '{} : {}, '.format(key, data_settings[key])
    print(experiment_message)
    experiment_message = 'Running with '
    for key in param_settings.keys():
        experiment_message += '{} : {}, '.format(key, param_settings[key])
    print(experiment_message)

def display_experiment_results(metrics):
    
    for key in metrics.keys():
        print('{}: {}'.format(key, metrics[key]))

# def write_metrics(error_total, error_direct, error_indirect):

#     metrics = instantiate_metrics()
#     metrics['error_total'] = error_total
#     metrics['error_direct'] = error_direct
#     metrics['error_indirect'] = error_indirect
#     # metrics['true_total'] = true_total[0]
#     # metrics['true_direct'] = true_direct[0]
#     # metrics['true_indirect'] = true_indirect[0]
#     return metrics