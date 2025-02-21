import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy.special import expit, logit
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_continuous

from data.base import CausalEnvironment
from utils.utils import display_experiment_configuration, display_experiment_results, write_metrics, split_data

params = {
    'alpha':0.5,
    'beta':0.25,
    }


class BinomialEnv(CausalEnvironment):

    def __init__(self, *args):
        """IPW estimator

        Attributes:
            _clip (float):  clipping the propensities
            _trim (float): remove propensities which are below the trim threshold

        """
        super(BinomialEnv, self).__init__(*args)
        grid = np.arange(-1.5, 1.5, 0.1)
        self.grid = np.delete(grid, 15)
        self.name = 'Binomial'
        if self._settings:
            self.alpha = self._settings['alpha']
            self.beta = self._settings['beta']
        else:
            self.alpha = params['alpha']
            self.beta = params['beta']

        self.coeff = self._settings['gamma']
        self.stochasticity = self._settings['stochasticity']

        self.alpha = self._settings['alpha']
        self.beta = self._settings['beta']
        self.coeff = self._settings['gamma']
        self.stochasticity = self._settings['stochasticity']
        self.noise = self._settings['noise']

    def generate_causal_data(self, data_settings, 
                            params=params,
                            random_state=42,
                            mode='id'):
        
        n_samples = data_settings['n_samples']
        
        rng = np.random.RandomState(random_state)
        
    
        # Generate X from a multivariate normal distribution with specified variances
        X = rng.normal(0, np.sqrt([0.25, 0.1, 0.8]), size=(n_samples, 3))

        # Extract X1, X2, X3
        X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]

        # Generate A based on the specified distribution
        T = rng.normal(5 + X1 + 0.2 * X1**2, 1, size=n_samples)

        # Define the sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Calculate delta(A, X) using the sigmoid function
        delta = sigmoid(-5 + 5 * T + 2 * X2 + 10 * T * X3)

        # Generate M from a Bernoulli distribution with parameter delta
        M = rng.binomial(1, delta, size=n_samples)

        # Generate Y based on the specified distribution
        Y = rng.normal(-T + 20 * M + 5 * M * X1 + X2, 1, size=n_samples)

        M = M.reshape(-1, 1)

        return X, T, M, Y, params

    def get_causal_effects(self, X, t, t_prime, data_settings, 
                            params,
                            mode='id'):
        

        pass
    
    def get_mediated_response(self, X, t, t_prime, data_settings, 
                            params,
                            mode='id'):

        pass

    def causal_experiment(self, estimator, t, m, x, y, data_settings, params):


        d = 4.5
        d_prime = 6
        

        causal_effects = estimator.estimate(d, d_prime, t, m, x, y)
        mediated_response = 9.1
        
        # metrics
        average_absolute_bias = abs(causal_effects['mediated_response'] - mediated_response)

        metrics = {
            'bias_direct': 0,
            'rmse_direct': 0,
            'bias_indirect': 0,
            'rmse_indirect': 0,
            'bias_total': 0,
            'rmse_total': 0,
            'bias_mr': average_absolute_bias,
            'rmse_mr': average_absolute_bias,
            'coverage': 0
        }

        display_experiment_results(metrics)

        return metrics