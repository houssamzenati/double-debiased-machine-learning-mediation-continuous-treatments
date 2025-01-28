import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from estimators.base import Estimator
from nuisances.kme import _kme_conditional_mean_outcome
from utils.decorators import fitted


class KMEGComputation(Estimator):
    """Implementation of Kernel Mean Embedding G computation

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, settings, verbose=0):
        super(KMEGComputation, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self.name = 'G_comp'

    def resize(self, t, m, x, y):
        """Resize data for the right shape

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
        if len(y) != len(y.ravel()):
            raise ValueError("Multidimensional y is not supported")
        if len(t) != len(t.ravel()):
            raise ValueError("Multidimensional t is not supported")

        n = len(y)
        if len(x.shape) == 1:
            x.reshape(n, 1)
        if len(m.shape) == 1:
            m.reshape(n, 1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError(
                "Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y

    def fit(self, t, m, x, y):
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
        # self.fit_score_nuisances(t, m, x, y)
        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")


    @fitted
    def estimate(self, d, d_prime, t, m, x, y):
        """Estimates causal effect on data

        """
        t, m, x, y = self.resize(t, m, x, y)

        eta_d_d, eta_d_d_prime, eta_d_prime_d, eta_d_prime_d_prime = (
            _kme_conditional_mean_outcome(d,
                                          d_prime,
                                          y,
                                          t,
                                          m,
                                          x,
                                          self._settings))

        direct_effect = eta_d_prime_d - eta_d_d
        indirect_effect = eta_d_prime_d_prime - eta_d_prime_d
        total_effect = direct_effect + indirect_effect

        causal_effects = {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'mediated_response': eta_d_d_prime,
        'variance': 0,
        'margin_error': 0
        }

        return causal_effects