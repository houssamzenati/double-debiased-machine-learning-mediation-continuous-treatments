import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
from scipy.special import expit, logit
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_continuous
from tqdm import tqdm

from data.base import CausalEnvironment
from utils.utils import display_experiment_configuration, display_experiment_results, write_metrics, split_data


path_to_file = '/data/work/tmp/20230721_fixed_ukbb_dataset_ncp2_bh1ac.csv'

class UKBBEnv(CausalEnvironment):

    def __init__(self, *args):
        """
        """
        super(UKBBEnv, self).__init__(*args)
        self.grid = np.arange(1, 25, 1)

    def generate_causal_data(self, data_settings, 
                            params=None,
                            random_state=42,
                            treatment_strategy='hb1ac'):
        """
        treatment_strategy: which treatment with which confounder
            either 'alcohol', or 'hb1ac' or 'hb1ac_without_diabetic'
            hb1ac_without_diabetic is when the treatment is the hb1ac level
            without including the dibaetic status as a confounder variable
            as it may be a collider or a mediator instead.
        """
        
        n_samples = data_settings['n_samples']
        rng = np.random.RandomState(random_state)
        mri_data_type = 'smri_dmri'
        
        ukbb_data = pd.read_csv(path_to_file, sep='\t')
        potential_mediator_cols = [c for c in ukbb_data.columns if 'mri' in c]
        unwanted_cols = [c for c in ukbb_data.columns if 'Unnamed' in c]
        ukbb_data.drop(columns=unwanted_cols + ['eid'], inplace=True)
        cols_to_remove = ['alcohol_above_nhs_reco'] + [c for c in ukbb_data.columns if "Alcohol_intake_frequency" in c]
        if treatment_strategy == 'alcohol':
            treatment_name = 'total_alcohol_weekly_units_2'
            self.grid = np.arange(1, 25, 1)
        elif treatment_strategy == 'hb1ac':
            treatment_name = 'hb1ac'
            self.grid = np.arange(1, 50, 2)
        elif treatment_strategy == 'hb1ac_without_diabetic':
            treatment_name = 'hb1ac'
            cols_to_remove += ['diabetic']
            self.grid = np.arange(1, 50, 2)
        else:
            raise(ValueError, 'wrong treatment strategy specified')
        mediator_columns = [c for c in ukbb_data.columns if '{}_pc'.format(mri_data_type) in c]
        outcome_columns = ['g_factor']
        treatment_columns = [treatment_name]

        confounder_columns = set(ukbb_data.columns).difference(potential_mediator_cols)\
                                                                      .difference(cols_to_remove)\
                                                                      .difference(outcome_columns)\
                                                                      .difference(treatment_columns)\
                                                                      .difference(['current_never_smoker', 'previous_never_smoker'])

        # i remove this confounder because of overlap issues
        confounder_columns.remove('Adopted_as_a_child-0.0_1.0')
        confounder_columns.remove('Adopted_as_a_child-0.0_0.0')
        # 'Country_of_birth_(UK/elsewhere)-0.0_5.0'
        # 'Comparative_height_size_at_age_10-0.0_-1.0'
        # 'Country_of_birth_(UK/elsewhere)-0.0_4.0'
        ukbb_data.dropna(subset=treatment_columns, inplace=True)
        ukbb_data = ukbb_data[ukbb_data.hb1ac<55]
        # ukbb_data = ukbb_data[ukbb_data.total_alcohol_weekly_units_2<=50]


        x = ukbb_data[list(confounder_columns)].values
        y = -ukbb_data[outcome_columns].values
        t = ukbb_data[treatment_columns].values
        m = ukbb_data[mediator_columns].values

        ok_idx = (~np.isnan(t)).ravel()
        x, y, t, m = x[ok_idx, :], y[ok_idx, :], t[ok_idx, :], m[ok_idx, :]


        # # extract some bootstrap sample to reduce the size
        # ind = rng.choice(len(y), n_samples, replace=True)


        # y_b, t_b, m_b, x_b = y[ind], t[ind], m[ind, :], x[ind, :]
        y_b, t_b, m_b, x_b = y, t, m, x
        print('Length of dataset {}'.format(y_b.shape[0]))
        x_b = x_b.astype(float)

        return x_b, t_b, m_b, y_b, params

    
    def causal_experiment(self, estimator, t, m, x, y, settings_params):

        n_samples = settings_params['n_samples']
        d, d_prime = settings_params['d'], settings_params['d_prime']

        rng = np.random.RandomState(settings_params['random_state'])
        ind = rng.choice(len(y), n_samples, replace=True)
        y_b, t_b, m_b, x_b = y[ind], t[ind], m[ind, :], x[ind, :]
        x_b = x_b.astype(float)
        
        # Estimate causal effects
        causal_effects = estimator.estimate(d, d_prime, t_b, m_b, x_b, y_b)

        metrics = causal_effects

        display_experiment_results(metrics)

        return metrics