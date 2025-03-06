import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.preprocessing import StandardScaler

import numpy as np

from data.base import CausalEnvironment
from utils.utils import display_experiment_results

params = {
    'alpha':0.5,
    'beta':0.25,
    }


class UniformEnv(CausalEnvironment):

    def __init__(self, *args):
        """IPW estimator

        Attributes:
            _clip (float):  clipping the propensities
            _trim (float): remove propensities which are below the trim threshold

        """
        super(UniformEnv, self).__init__(*args)
        grid = np.arange(-1.5, 1.5, 0.1)
        self.grid = np.delete(grid, 15)
        self.name = 'Uniform'
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
        self.mediator_dimension = self._settings['mediator_dimension']

    def generate_causal_data(self, data_settings, 
                            params=params,
                            random_state=42,
                            mode='id'):
        
        n_samples = data_settings['n_samples']
        
        rng = np.random.RandomState(random_state)
        
        if self.noise == 'uniform':
            X = rng.uniform(-1.5, 1.5, size=(n_samples,1))
            U = self.stochasticity*rng.uniform(-2, 2, size=(n_samples,1))
            V = self.stochasticity*rng.uniform(-2, 2, size=(n_samples,1))
            W = self.stochasticity*rng.uniform(-2, 2, size=(n_samples,1))
            
        elif self.noise == 'gaussian':

            X = rng.normal(size=(n_samples,1))
            U = self.stochasticity*rng.normal(size=(n_samples,1))
            V = self.stochasticity*rng.normal(size=(n_samples,1))
            W = self.stochasticity*rng.normal(size=(n_samples,1))
        

        T = self.coeff*X+W
        M = self.coeff*T+self.coeff*X+V
        Y = self.coeff*T+self.coeff*M+self.alpha*T*M+self.coeff*X+self.beta*T**3+U
        if self.mediator_dimension == 5:
           M =+ self.stochasticity*rng.uniform(-2, 2, size=(n_samples, 5))
        return X, T, M, Y, params

    def get_causal_effects(self, X, t, t_prime, data_settings, 
                            params,
                            mode='id'):
        

        theta = self.coeff*(t_prime-t) + self.coeff*self.alpha*t*(t_prime-t)+self.beta*(t_prime**3-t**3)
        delta = self.coeff**2*(t_prime-t) + self.coeff*self.alpha*t_prime*(t_prime-t)
        tau = theta + delta

        causal_effects = [tau, theta, delta]

        return causal_effects
    
    def get_mediated_response(self, X, t, t_prime, data_settings, 
                            params,
                            mode='id'):

        return self.coeff*t+self.coeff**2*t_prime+self.coeff*self.alpha*t_prime*t+self.beta*t**3

    def causal_experiment(self, estimator, t, m, x, y, data_settings, params):


        d_prime = 0
        
        direct_bias = []
        direct_rmse = []
        
        indirect_bias = []
        indirect_rmse = []
        
        total_bias = []
        total_rmse = []

        mr_bias = []
        mr_rmse = []
        coverages = []

        for d in self.grid:

            total, direct, indirect = self.get_causal_effects(x, d, d_prime, data_settings, params)
            causal_effects = estimator.estimate(d, d_prime, t, m, x, y)
            mediated_response = self.get_mediated_response(x, d, d_prime, data_settings, params)
            
            # metrics
            direct_bias.append(abs(causal_effects['direct_effect'] - direct))
            direct_rmse.append((causal_effects['direct_effect'] - direct)**2)
        
            indirect_bias.append(abs(causal_effects['indirect_effect'] - indirect))
            indirect_rmse.append((causal_effects['indirect_effect'] - indirect)**2)
        
            total_bias.append(abs(causal_effects['total_effect'] - total))
            total_rmse.append((causal_effects['total_effect'] - total)**2)

            mr_bias.append(abs(causal_effects['mediated_response'] - mediated_response))
            mr_rmse.append((causal_effects['mediated_response'] - mediated_response)**2)

            coverage = (
                (mediated_response >= causal_effects['mediated_response'] - causal_effects['margin_error']) &
                (mediated_response <= causal_effects['mediated_response'] + causal_effects['margin_error'])
                )
            
            coverages.append(coverage)

        
        bias_direct = np.mean(direct_bias)
        rmse_direct = np.sqrt(np.mean(direct_rmse))
        
        bias_indirect = np.mean(indirect_bias)
        rmse_indirect = np.sqrt(np.mean(indirect_rmse))
        
        bias_total = np.mean(total_bias)
        rmse_total = np.sqrt(np.mean(total_rmse))

        bias_mr = np.mean(mr_bias)
        rmse_mr = np.sqrt(np.mean(mr_rmse))

        coverage = np.mean(coverages)

        metrics = {
            'bias_direct': bias_direct,
            'rmse_direct': rmse_direct,
            'bias_indirect': bias_indirect,
            'rmse_indirect': rmse_indirect,
            'bias_total': bias_total,
            'rmse_total': rmse_total,
            'bias_mr': bias_mr,
            'rmse_mr': rmse_mr,
            'coverage': coverage
        }

        display_experiment_results(metrics)

        return metrics