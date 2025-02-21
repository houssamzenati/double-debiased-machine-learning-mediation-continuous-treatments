import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from utils.decorators import fitted
from utils.utils import _get_interactions, split_data, is_array_integer
from nuisances.density import get_density_by_name


class Estimator:
    """General abstract class for causal mediation Estimator.

    Attributes
    ----------
    rng : RandomState
        Random number generator for reproducibility.
    _settings : dict
        Configuration settings for the estimator.
    _verbose : bool
        Whether to print verbose output.
    _fitted : bool
        Whether the estimator has been fitted.
    _regularize : bool
        Whether regularization is enabled.
    _bandwidth : float
        Bandwidth parameter for kernel density estimation.

    """
    __metaclass__ = ABCMeta

    def __init__(self, settings, verbose, *args, **kwargs):
        """Initializes the Estimator.

        Parameters
        ----------
        settings : dict
            Estimator configuration settings.
        verbose : bool
            Whether to enable verbose output.
        args, kwargs : optional
            Additional arguments.

        """
        self.rng = np.random.RandomState(123)
        self._settings = settings
        self._verbose = bool(verbose)
        self._fitted = False
        self._regularize = settings['regularization']
        self._bandwidth = settings['bandwidth']

    @property
    def verbose(self):
        """Returns whether verbose mode is enabled."""
        return self._verbose


    @abstractmethod
    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data.

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values, binary or continuous
        m : array-like, shape (n_samples,)
            Mediator values, binary and unidimensional.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates (potential confounders).
        y : array-like, shape (n_samples,)
            Outcome values, continuous.

        """
        pass


    @abstractmethod
    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data.

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values, binary.
        m : array-like, shape (n_samples,)
            Mediator values, binary and unidimensional.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates (potential confounders).
        y : array-like, shape (n_samples,)
            Outcome values, continuous.

        """
        pass

    def compute_causal_effects(self, causal_data, nuisances):
        """Computes causal effects by splitting data, fitting nuisances, and estimating effects.

        Parameters
        ----------
        causal_data : tuple
            Tuple containing (x, t, m, y).
        nuisances : dict
            Dictionary of nuisance parameters.

        """
        ### Split data
        causal_data_nuisance, causal_data_estimation = split_data(causal_data, nuisances)

        ### Fit nuisance parameters
        x, t, m, y = causal_data_nuisance
        self.fit(t, m, x, y)

        ### Estimate causal effects
        x, t, m, y = causal_data_estimation
        causal_effects = self.estimate(t, m, x, y)

    def fit_tx_density(self, t, x):
        """Fits the density function f(T|X) using the specified estimation method in
        settings dictionary

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates.

        """
        self._density_tx = get_density_by_name(self._settings)
        self._density_tx.fit(x, t.squeeze())

    def fit_txm_density(self, t, x, m):
        """Fits the joint density function f(T|X, M).

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates.
        m : array-like, shape (n_samples,)
            Mediator values.

        """
        self._density_txm = get_density_by_name(self._settings)
        xm = np.hstack((x, m))
        self._density_txm.fit(xm, t.squeeze())


    def fit_mediator_density(self, t, x, m):
        """Fits the density function f(M|T, X).

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates.
        m : array-like, shape (n_samples,)
            Mediator values.

        """
        self._density_m = get_density_by_name(self._settings)
        t_x = _get_interactions(False, t, x)
        self._density_m.fit(t_x, m.squeeze())


    def fit_bandwidth(self, t):
        """ Fits the bandwidth with the Scott heurisitic
    
        """
        std, n = np.std(t), t.shape[0]
        self._bandwidth = 1.06 * std * n ** (-1/5)