import os
import sys

import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV, LinearRegression

from estimators.base import Estimator
from utils.config import ALPHAS, CV_FOLDS, TINY
from utils.decorators import fitted


class Linear(Estimator):

    def __init__(self, *args):
        """Coefficient product estimator

        Attributes:
            _clip (float):  clipping the propensities
            _trim (float): remove propensities which are below the trim threshold

        """
        super(Linear, self).__init__(*args)
        self._crossfit = 0
        self.name = 'OLS'

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
        # estimate mediator densities

        if self._regularize:
            alphas = ALPHAS
        else:
            alphas = [TINY]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(m.shape) == 1:
            m = m.reshape(-1, 1)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        self._coef_t_m = np.zeros(m.shape[1])
        self._coef_x_m = np.zeros((m.shape[1], x.shape[1]))

        for i in range(m.shape[1]):
            # m_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS) \
            #     .fit(np.hstack((x, t)), m[:, i])
            m_reg = LinearRegression().fit(np.hstack((x, t)), m[:, i])
            self._coef_t_m[i] = m_reg.coef_[-1]
            self._coef_x_m[i] = m_reg.coef_[:-1]
        # y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS) \
        #     .fit(np.hstack((x, t, m)), y.ravel())
        y_reg =  LinearRegression().fit(np.hstack((x, t, m)), y.ravel())

        self._coef_y = y_reg.coef_

        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")


    @fitted
    def estimate(self, d, d_prime, t, m, x, y):
        """Estimates causal effect on data

        """
        direct_effect = self._coef_y[x.shape[1]] * (d_prime - d)
        indirect_effect = sum(self._coef_y[x.shape[1] + 1:] * self._coef_t_m) * (d_prime - d)
        M_d_prime = np.mean(x.dot(self._coef_x_m.T)) + self._coef_t_m * d_prime
        mediated_response = np.mean(x.dot(self._coef_y[:x.shape[1]])) \
            + self._coef_y[x.shape[1]] * d + sum(M_d_prime * self._coef_t_m)

        causal_effects = {
            'total_effect': direct_effect+indirect_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediated_response': mediated_response,
            'variance': 0,
            'margin_error': 0
        }
        return causal_effects
