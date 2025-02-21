import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np

from estimators.base import Estimator
from nuisances.kme import _kme_cross_conditional_mean_outcomes
from utils.decorators import fitted
from sklearn.model_selection import KFold
from scipy.stats import norm

def db_exp_kernel(x1, x2, variance = 1):
    return exp(-1 * (np.linalg.norm(x1-x2)) / (2*variance))

def gram_matrix(xs):
    return rbf_kernel(xs, gamma=0.5)
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2 )/(np.sqrt(2*np.pi))
def gaussian_kernel_h(u,h_2):
    return (1/(np.sqrt(h_2)*np.sqrt(2*np.pi)))*np.exp((-0.5)/h_2 * (1.0*u)**2 )
def gaussian_k_bar(u):
    return (1/(np.sqrt(4*np.pi)))*np.exp(.25* np.linalg.norm(1.0*u)**2)
def epanechnikov_kernel(u):
    condition = np.abs(u) <=1
    return np.where(condition, 0.75*(1-u**2), 0) 

class KMEDoubleMachineLearning(Estimator):
    """Implementation of double machine learning

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, settings, verbose=0):
        super(KMEDoubleMachineLearning, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self._normalized = self._settings['normalized']
        self._sample_splits = self._settings['sample_splits']
        self.kernel = gaussian_kernel
        self.name = 'DML'
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

        self.fit_tx_density(t, x)
        self.fit_txm_density(t, x, m)
        self.fit_bandwidth(t)

        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")

    def estimate(self, d, d_prime, t, m, x, y):

        if self._bandwidth_mode == 'amse':
            self.fit_bandwidth(t)
            self.fit_amse_bandwidth(d, d_prime, t, m, x, y)
            return self._estimate(d, d_prime, t, m, x, y)
        
        else:
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
            d = d * np.ones_like(t)
            d_prime = d_prime * np.ones_like(t)
            f_d_x = self._density_tx.pdf(x, d)
            f_d_prime_x = self._density_tx.pdf(x, d_prime)

            xm = np.hstack((x, m))
            f_d_xm = self._density_txm.pdf(xm, d)
            f_d_prime_xm = self._density_txm.pdf(xm, d_prime)

            # estimate conditional mean outcomes
            # mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 
            mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                _kme_cross_conditional_mean_outcomes(d,
                                                    d_prime,
                                                    y,
                                                    t,
                                                    m,
                                                    x,
                                                    self._settings))

            k_dt = self.kernel((d-t)/self._bandwidth)/self._bandwidth
            k_d_prime_t = self.kernel((d_prime-t)/self._bandwidth)/self._bandwidth

            k_dt = k_dt.squeeze()
            k_d_prime_t = k_d_prime_t.squeeze()
            y = y.squeeze()

            # score computing
            if self._normalized:
                sum_score_d_d = np.mean(k_dt / f_d_x)
                sum_score_d_prime_d_prime = np.mean(k_d_prime_t / f_d_prime_x)
                sum_score_d_prime_d = np.mean(k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x))
                sum_score_d_d_prime = np.mean(k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x))

                y_d_m_d = (k_dt / f_d_x * (y - psi_d_d)) / sum_score_d_d + psi_d_d
                
                y_d_prime_m_d_prime = (k_d_prime_t / f_d_prime_x * (y - psi_d_prime_d_prime)) / sum_score_d_prime_d_prime + psi_d_prime_d_prime
                
                y_d_prime_m_d = (k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x) * (y-mu_d_prime)) / sum_score_d_prime_d + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d)/sum_score_d_d + psi_d_prime_d

                y_d_m_d_prime = (k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x) * (y-mu_d)) / sum_score_d_d_prime + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime)/sum_score_d_prime_d_prime + psi_d_d_prime


            else:

                y_d_m_d = (k_dt / f_d_x * (y - psi_d_d)) + psi_d_d
                
                y_d_prime_m_d_prime = (k_d_prime_t / f_d_prime_x * (y - psi_d_prime_d_prime)) + psi_d_prime_d_prime
                
                y_d_prime_m_d = (k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x) * (y-mu_d_prime)) + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d) + psi_d_prime_d

                y_d_m_d_prime = (k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x) * (y-mu_d)) + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime) + psi_d_d_prime

        else: 
            
            # Initialize KFold for sample splitting
            kf = KFold(n_splits=self._sample_splits)

            # Cross-Fitting
            for train_idx, test_idx in kf.split(x):

                # Train nuisance models on one split

                self.fit_tx_density(t[train_idx], x[train_idx])
                self.fit_txm_density(t[train_idx], x[train_idx], m[train_idx])
                
                # Predict for the other split

                d = d * np.ones_like(t[test_idx])
                d_prime = d_prime * np.ones_like(t[test_idx])
                f_d_x = self._density_tx.pdf(x[test_idx], d)
                f_d_prime_x = self._density_tx.pdf(x[test_idx], d_prime)

                xm = np.hstack((x[test_idx], m[test_idx]))
                f_d_xm = self._density_txm.pdf(xm, d)
                f_d_prime_xm = self._density_txm.pdf(xm, d_prime)

                # estimate conditional mean outcomes
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                    _kme_cross_conditional_mean_outcomes(d,
                                                        d_prime,
                                                        y[test_idx],
                                                        t[test_idx],
                                                        m[test_idx],
                                                        x[test_idx],
                                                        self._settings))

                k_dt = self.kernel((d-t[test_idx])/self._bandwidth)/self._bandwidth
                k_d_prime_t = self.kernel((d_prime-t[test_idx])/self._bandwidth)/self._bandwidth

                k_dt = k_dt.squeeze()
                k_d_prime_t = k_d_prime_t.squeeze()
                y_ = y[test_idx].squeeze()

                # score computing
                if self._normalized:

                    sum_score_d_d = np.mean(k_dt / f_d_x)
                    sum_score_d_prime_d_prime = np.mean(k_d_prime_t / f_d_prime_x)
                    sum_score_d_prime_d = np.mean(k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x))
                    sum_score_d_d_prime = np.mean(k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x))

                    y_d_m_d[test_idx] = (k_dt / f_d_x * (y_ - psi_d_d)) / sum_score_d_d + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = (k_d_prime_t / f_d_prime_x * (y_ - psi_d_prime_d_prime)) / sum_score_d_prime_d_prime + psi_d_prime_d_prime 
                    y_d_prime_m_d[test_idx] = (k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x) * (y_ - mu_d_prime)) / sum_score_d_prime_d + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d)/sum_score_d_d + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = (k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x) * (y_ - mu_d)) / sum_score_d_d_prime + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime)/sum_score_d_prime_d_prime + psi_d_d_prime

                else:

                    y_d_m_d[test_idx] = (k_dt / f_d_x * (y_ - psi_d_d)) + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = (k_d_prime_t / f_d_prime_x * (y_ - psi_d_prime_d_prime)) + psi_d_prime_d_prime
                    y_d_prime_m_d[test_idx] = (k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x) * (y_ - mu_d_prime)) + (k_dt / f_d_x) * (mu_d_prime-psi_d_prime_d) + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = (k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x) * (y_ - mu_d)) + (k_d_prime_t / f_d_prime_x) * (mu_d-psi_d_d_prime) + psi_d_d_prime

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

    def fit_amse_bandwidth(self, d, d_prime, t, m, x, y):
        """ Fits the bandwidth with the Scott heurisitic
    
        """        
        n = t.shape[0]
        effects = self._estimate(d, d_prime, t, m, x, y)
        v_d_m_d_prime = effects['variance']
        mr_d_m_d_prime = effects['mediated_response']

        self._bandwidth = self._epsilon * self._bandwidth 
        effects = self._estimate(d, d_prime, t, m, x, y)
        mr_d_m_d_prime_epsilon = effects['mediated_response']

        bias_d_m_d_prime = (mr_d_m_d_prime - mr_d_m_d_prime_epsilon)/((self._bandwidth/self._epsilon)**2*(1-self._epsilon)**2)

        self._bandwidth = (v_d_m_d_prime/(4*bias_d_m_d_prime)**2)**(1/5)*n**(-1/5)
