from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy.special import expit, logit
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_continuous

import os
import sys
from abc import ABCMeta, abstractmethod


class CausalEnvironment:
    """General abstract class for causal environment
    """
    __metaclass__ = ABCMeta
    def __init__(self, settings=None, verbose=False, *args, **kwargs):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.rng = np.random.RandomState(123)
        self._settings = settings
        self._verbose = bool(verbose)

    
    @property
    def verbose(self):
        return self._verbose


    @abstractmethod
    def generate_causal_data(self, data_settings, 
                         data_parameters=None,
                         random_state=42,
                         mode='id'):
        """Fits nuisance parameters to data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        """
        pass

    @abstractmethod
    def get_causal_effects(self, X, t, t_prime, data_settings, 
                         params,
                         mode='id'):
        """Fits nuisance parameters to data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        """
        pass

    @abstractmethod
    def causal_experiment(self, estimator, t, m, x, y, data_settings, params):
        """Fits nuisance parameters to data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        """
        pass

