import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np

from estimators.base import Estimator
from utils.decorators import fitted
from sklearn.model_selection import KFold
from scipy.stats import norm
from sklearn.cluster import KMeans
from utils.utils import (bucketize_mediators,
                         is_array_integer)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from nuisances.kme import _kme_cross_conditional_mean_outcomes
from utils.utils import _get_interactions, is_array_integer
from nuisances.density import get_density_by_name


def gaussian_kernel(u):
    return np.exp(-0.5 * u**2 )/(np.sqrt(2*np.pi))
def gaussian_kernel_h(u,h_2):
    return (1/(np.sqrt(h_2)*np.sqrt(2*np.pi)))*np.exp((-0.5)/h_2 * (1.0*u)**2 )
def gaussian_k_bar(u):
    return (1/(np.sqrt(4*np.pi)))*np.exp(.25* np.linalg.norm(1.0*u)**2)
def epanechnikov_kernel(u):
    condition = np.abs(u) <=1
    return np.where(condition, 0.75*(1-u**2), 0) 

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5

class SaniDoubleMachineLearning(Estimator):
    """Implementation of double machine learning

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, settings, verbose=0):
        super(SaniDoubleMachineLearning, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self._normalized = self._settings['normalized']
        self._sample_splits = self._settings['sample_splits']
        self.kernel = gaussian_kernel
        self.name = 'DML_Sani'
        self._bandwidth_mode = self._settings['bandwidth_mode']
        self._epsilon = settings['epsilon']

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
            m = m.reshape(n, 1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError(
                "Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y
    
    def fit_conditional_mean_outcome(self, t, x, m, y):
        """Fits the joint density function f(T|X, M).

        Parameters
        ----------
        t : array-like, shape (n_samples,)
            Treatment values.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates.
        m : array-like, shape (n_samples,)
            Mediator values.
        y : array-like, shape (n_samples,)
            Outcome values.

        """
        regressor = RidgeCV(alphas=ALPHAS, cv=CV_FOLDS) 
        x_t_m = np.hstack([x, t.reshape(-1, 1), m])
        self._regressor_y = regressor.fit(x_t_m, y)

    def estimate_cross_conditional_mean_outcome_discrete(self, d, d_prime, m, x):
        """
        Estimate the conditional mean outcome,
        the cross conditional mean outcome

        Returns
        -------
        mu_m0x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M,X]
        mu_m1x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M,X]
        E_mu_t0_t0, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=0,X]
        E_mu_t0_t1, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t1_t0, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t1, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=1,X]
        """
        n = m.shape[0]

        # Initialisation
        (
            E_mu_d_prime_d,  # E[E[Y|T=d',M,X]|T=d,X]
            E_mu_d_prime_d_prime,  # E[E[Y|T=d',M,X]|T=d',X]
            E_mu_d_d_prime,  # E[E[Y|T=d,M,X]|T=d',X]
            E_mu_d_d,  # E[E[Y|T=d,M,X]|T=d,X]
            mu_d_prime_mx,
            mu_d_mx
        ) = [np.zeros(n) for _ in range(6)]

        # predict E[Y|T=t,M,X]
        x_d_m = np.hstack([x, d.reshape(-1, 1), m])
        x_d_prime_m = np.hstack([x, d_prime.reshape(-1, 1), m])
        mu_d_prime_mx = self._regressor_y.predict(
                x_d_prime_m)
        mu_d_mx = self._regressor_y.predict(
                x_d_m)
        

        for b in np.unique(m):

            # f(M=b|T=d,X)
            b = np.ones((n,1)) * b
            f_b_dx = self.predict_mediator_density(d, x, b)
            f_b_d_prime_x = self.predict_mediator_density(d_prime, x, b)
            x_d_b = np.hstack([x, d.reshape(-1, 1), b])
            x_d_prime_b = np.hstack([x, d_prime.reshape(-1, 1), b])

            # predict E[E[Y|T=d',M=m,X]|T=d,X]
            E_mu_d_prime_d += self._regressor_y.predict(
                x_d_prime_b) * f_b_dx
            # predict E[E[Y|T=d,M=m,X]|T=d',X]
            E_mu_d_d_prime += self._regressor_y.predict(
                x_d_b) * f_b_d_prime_x
            # predict E[E[Y|T=d',M=m,X]|T=d',X]
            E_mu_d_prime_d_prime += self._regressor_y.predict(
                x_d_prime_b) * f_b_d_prime_x
            # predict E[E[Y|T=d,M=m,X]|T=d,X]
            E_mu_d_d += self._regressor_y.predict(
                x_d_b) * f_b_dx


        return mu_d_mx, mu_d_prime_mx, E_mu_d_d, E_mu_d_d_prime, E_mu_d_prime_d, E_mu_d_prime_d_prime
    
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
        t_x = _get_interactions(False, t, x)
        if not is_array_integer(m):
            if m.shape[1]>1:
                self._settings['density'] = 'ls_conditional'
            self._density_m = get_density_by_name(self._settings)
            self._density_m.fit(t_x, m.squeeze())
        else:
            classifier_m = LogisticRegressionCV(random_state=42, 
                                                Cs=ALPHAS,
                                                cv=CV_FOLDS, 
                                                multi_class='auto',
                                                max_iter=1000)
            self._classifier_m = classifier_m.fit(t_x, m.squeeze())

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
        t, m, x, y = self.resize(t, m, x, y)
        if self._epsilon == 'kme':
            self.fit_mediator_density(t, x, m)
        else:
            if not is_array_integer(m):
                self._bucketizer = KMeans(n_clusters=10, random_state=self.rng,
                                n_init="auto").fit(m)
                m = np.expand_dims(self._bucketizer.predict(m), axis=-1)
        self.fit_mediator_density(t, x, m)

        self.fit_tx_density(t, x)
        
        self.fit_conditional_mean_outcome(t, m, x, y)
        self.fit_bandwidth(t)

        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")

    def predict_mediator_density(self, d, x, m):
        """Predicts the density function f(M|D, X).

        Parameters
        ----------
        d : array-like, shape (n_samples,)
            Treatment values.
        x : array-like, shape (n_samples, n_features_covariates)
            Covariates.
        m : array-like, shape (n_samples,)
            Mediator values.

        """
        d_x = _get_interactions(False, d, x)
        if self._epsilon == 'kme':
            return self._density_m.pdf(d_x, m)
        # if not is_array_integer(m):
        #     self._density_m.pdf(d_x, m)
        else:
            return self._classifier_m.predict_proba(d_x)[np.arange(m.shape[0]), m.ravel().astype(int)]


    def estimate(self,d, d_prime, t, m, x, y):
        return self._estimate(d, d_prime, t, m, x, y)


    def _estimate(self, d, d_prime, t, m, x, y):
        """Estimates causal effect on data

        """

        n = t.shape[0]

        # Create placeholders for cross-fitted predictions
        y_d_m_d = np.zeros(n)
        y_d_prime_m_d_prime = np.zeros(n)
        y_d_prime_m_d = np.zeros(n)
        y_d_m_d_prime = np.zeros(n)

        t, m, x, y = self.resize(t, m, x, y)

        if self._sample_splits == 1:
            d_ = d * np.ones_like(t)
            d_prime_ = d_prime * np.ones_like(t)
            f_d_x = self._density_tx.pdf(x, d_)
            f_d_prime_x = self._density_tx.pdf(x, d_prime_)

            f_m_xd = self.predict_mediator_density(d_, x, m)
            f_m_xd_prime = self.predict_mediator_density(d_prime_, x, m)

            # estimate conditional mean outcomes
            if self._epsilon == 'kme':
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                                _kme_cross_conditional_mean_outcomes(d_,
                                                                    d_prime_,
                                                                    y,
                                                                    t,
                                                                    m,
                                                                    x,
                                                                    self._settings))
            else:
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                    self.estimate_cross_conditional_mean_outcome_discrete(d_,
                                                        d_prime_,
                                                        m,
                                                        x))

            k_dt = self.kernel((d_-t)/self._bandwidth)/self._bandwidth
            k_d_prime_t = self.kernel((d_prime_-t)/self._bandwidth)/self._bandwidth

            k_dt = k_dt.squeeze()
            k_d_prime_t = k_d_prime_t.squeeze()
            y = y.squeeze()

            # score computing
            if self._normalized:
                sum_score_d_d = np.mean(k_dt / f_d_x)
                sum_score_d_prime_d_prime = np.mean(k_d_prime_t / f_d_prime_x)
                sum_score_d_prime_d = np.mean(k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x))
                sum_score_d_d_prime = np.mean(k_dt * f_m_xd_prime / (f_m_xd * f_d_x))

                y_d_m_d = (k_dt / f_d_x * (y - psi_d_d)) / sum_score_d_d + psi_d_d
                
                y_d_prime_m_d_prime = (k_d_prime_t / f_d_prime_x * (y - psi_d_prime_d_prime)) / sum_score_d_prime_d_prime + psi_d_prime_d_prime
                
                y_d_prime_m_d = (k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x) * (y-mu_d_prime)) / sum_score_d_prime_d + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d)/sum_score_d_d + psi_d_prime_d

                y_d_m_d_prime = (k_dt * f_m_xd_prime / (f_m_xd * f_d_x) * (y-mu_d)) / sum_score_d_d_prime + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime)/sum_score_d_prime_d_prime + psi_d_d_prime


            else:

                y_d_m_d = (k_dt / f_d_x * (y - psi_d_d)) + psi_d_d
                
                y_d_prime_m_d_prime = (k_d_prime_t / f_d_prime_x * (y - psi_d_prime_d_prime)) + psi_d_prime_d_prime
                
                y_d_prime_m_d = (k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x) * (y-mu_d_prime)) + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d) + psi_d_prime_d

                y_d_m_d_prime = (k_dt * f_m_xd_prime / (f_m_xd * f_d_x) * (y-mu_d)) + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime) + psi_d_d_prime

        else: 
            
            # Initialize KFold for sample splitting
            kf = KFold(n_splits=self._sample_splits)

            # Cross-Fitting
            for train_idx, test_idx in kf.split(x):

                # Train nuisance models on one split

                self.fit_tx_density(t[train_idx], x[train_idx])
                self.fit_mediator_density(t[train_idx], x[train_idx], m[train_idx])
                
                # Predict for the other split

                d_ = d * np.ones_like(t[test_idx])
                d_prime_ = d_prime * np.ones_like(t[test_idx])
                f_d_x = self._density_tx.pdf(x[test_idx], d_)
                f_d_prime_x = self._density_tx.pdf(x[test_idx], d_prime_)

                f_m_xd = self.predict_mediator_density(d_, x[test_idx], m[test_idx])
                f_m_xd_prime = self.predict_mediator_density(d_prime_, x[test_idx], m[test_idx])

                # estimate conditional mean outcomes
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                    self.estimate_cross_conditional_mean_outcome_discrete(d_,
                                                    d_prime_,
                                                    m[test_idx],
                                                    x[test_idx]))

                k_dt = self.kernel((d_-t[test_idx])/self._bandwidth)/self._bandwidth
                k_d_prime_t = self.kernel((d_prime_-t[test_idx])/self._bandwidth)/self._bandwidth

                k_dt = k_dt.squeeze()
                k_d_prime_t = k_d_prime_t.squeeze()
                y_ = y[test_idx].squeeze()

                # score computing
                if self._normalized:

                    sum_score_d_d = np.mean(k_dt / f_d_x)
                    sum_score_d_prime_d_prime = np.mean(k_d_prime_t / f_d_prime_x)
                    sum_score_d_prime_d = np.mean(k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x))
                    sum_score_d_d_prime = np.mean(k_dt * f_m_xd_prime / (f_m_xd * f_d_x))

                    y_d_m_d[test_idx] = (k_dt / f_d_x * (y_ - psi_d_d)) / sum_score_d_d + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = (k_d_prime_t / f_d_prime_x * (y_ - psi_d_prime_d_prime)) / sum_score_d_prime_d_prime + psi_d_prime_d_prime 
                    y_d_prime_m_d[test_idx] = (k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x) * (y_ - mu_d_prime)) / sum_score_d_prime_d + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d)/sum_score_d_d + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = (k_dt * f_m_xd_prime / (f_m_xd * f_d_x) * (y_ - mu_d)) / sum_score_d_d_prime + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime)/sum_score_d_prime_d_prime + psi_d_d_prime

                else:

                    y_d_m_d[test_idx] = (k_dt / f_d_x * (y_ - psi_d_d)) + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = (k_d_prime_t / f_d_prime_x * (y_ - psi_d_prime_d_prime)) + psi_d_prime_d_prime
                    y_d_prime_m_d[test_idx] = (k_d_prime_t * f_m_xd / (f_m_xd_prime * f_d_prime_x) * (y_ - mu_d_prime)) + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d) + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = (k_dt * f_m_xd_prime / (f_m_xd * f_d_x) * (y_ - mu_d)) + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime) + psi_d_d_prime

        # mean score computing
        my_dm_d = np.mean(y_d_m_d)
        my_d_prime_m_d_prime = np.mean(y_d_prime_m_d_prime)
        my_d_prime_m_d = np.mean(y_d_prime_m_d)
        my_d_m_d_prime = np.mean(y_d_m_d_prime)

        v_d_m_d_prime = self._bandwidth*np.mean((y_d_m_d_prime - my_d_m_d_prime)**2)

        confidence_level=0.95
        # Calculate the critical value (z_alpha/2) from the normal distribution
        z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Calculate margin of error
        margin_of_error = z_alpha_2 * np.sqrt(v_d_m_d_prime /(self._bandwidth * n))

        # effects computing
        total = my_d_prime_m_d_prime - my_dm_d
        direct = my_d_prime_m_d - my_dm_d
        indirect = my_d_prime_m_d_prime - my_d_prime_m_d

        causal_effects = {
            'total_effect': total,
            'direct_effect': direct,
            'indirect_effect': indirect,
            'mediated_response': my_d_m_d_prime,
            'variance': v_d_m_d_prime,
            'margin_error': margin_of_error
        }
        return causal_effects
    
