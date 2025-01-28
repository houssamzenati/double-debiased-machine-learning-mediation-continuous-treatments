
import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np

#
from estimators.base import Estimator
from utils.decorators import fitted
from utils.utils import _get_interactions


def gaussian_kernel(u):
    return np.exp(-0.5 * u**2 )/(np.sqrt(2*np.pi))
def gaussian_kernel_h(u,h_2):
    return (1/(np.sqrt(h_2)*np.sqrt(2*np.pi)))*np.exp((-0.5)/h_2 * (1.0*u)**2 )
def gaussian_k_bar(u):
    return (1/(np.sqrt(4*np.pi)))*np.exp(.25* np.linalg.norm(1.0*u)**2)
def epanechnikov_kernel(u):
    condition = np.abs(u) <=1
    return np.where(condition, 0.75*(1-u**2), 0) 

class ImportanceWeighting(Estimator):

  def __init__(self, *args):
    """IPW estimator

    Attributes:
        _clip (float):  clipping the propensities
        _trim (float): remove propensities which are below the trim threshold

    """
    super(ImportanceWeighting, self).__init__(*args)
    self._crossfit = 0
    self.kernel = epanechnikov_kernel
    self.EPS = 1e-5
    self.name = 'IPW'

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

    """
    # self.fit_score_nuisances(t, m, x, y)
    t, m, x, y = self.resize(t, m, x, y)

    self.fit_tx_density(t, x)
    self.fit_txm_density(t, x, m)
    # self.fit_mediator_density(t, x, m)
    self.fit_bandwidth(t)

    self._fitted = True

    if self.verbose:
        print(f"Nuisance models fitted")

  @fitted
  def estimate(self, d, d_prime, t, m, x, y):
    """Estimates causal effect on data

    """
    t, m, x, y = self.resize(t, m, x, y)
    d = d * np.ones_like(t)
    d_prime = d_prime * np.ones_like(t)
    
    f_d_x = self._density_tx.pdf(x, d)
    f_d_prime_x = self._density_tx.pdf(x, d_prime)

    xm = np.hstack((x, m))
    f_d_xm = self._density_txm.pdf(xm, d)
    f_d_prime_xm = self._density_txm.pdf(xm, d_prime)

    # d_x = _get_interactions(False, d, x)
    # d_prime_x = _get_interactions(False, d_prime, x)

    # f_m_dx = self._density_m.pdf(d_x, m)
    # f_m_dprime_x = self._density_m.pdf(d_prime_x, m)

    # f_m_dx = np.prod(f_m_dx, axis=1)
    # f_m_dprime_x = np.prod(f_m_dprime_x, axis=1)

    k_dt = self.kernel((d-t)/self._bandwidth)/self._bandwidth
    k_d_prime_t = self.kernel((d_prime-t)/self._bandwidth)/self._bandwidth
    k_dt = k_dt.squeeze()
    k_d_prime_t = k_d_prime_t.squeeze()
    y = y.squeeze()

    # importance weighting
    y_d_m_d = np.sum(y * k_dt / (f_d_x + self.EPS)) / np.sum(k_dt / (f_d_x + self.EPS))
    y_d_prime_m_d = np.sum(y * k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x + self.EPS)) /\
        np.sum(k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x + self.EPS))
    
    y_d_m_d_prime = np.sum(y * k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x + self.EPS)) /\
        np.sum(k_dt * f_d_prime_xm / (f_d_xm * f_d_prime_x + self.EPS))

    # y_d_prime_m_d = np.sum(y * k_d_prime_t * f_m_dx / (f_m_dprime_x * f_d_prime_x + self.EPS)) /\
    #     np.sum(k_d_prime_t * f_m_dx / (f_m_dprime_x * f_d_prime_x + self.EPS))
    # y_d_m_d_prime = np.sum(y * k_dt * f_m_dprime_x / (f_m_dx * f_d_x + self.EPS)) /\
    #     np.sum(k_dt * f_m_dprime_x / (f_m_dx * f_d_x + self.EPS))
    y_d_prime_m_d_prime = np.sum(y * k_d_prime_t / (f_d_prime_x + self.EPS)) /\
        np.sum(k_d_prime_t / (f_d_prime_x + self.EPS))
    
    # y_d_m_d = np.mean(y * k_dt / f_d_x)
    # y_d_prime_m_d = np.mean(y * k_d_prime_t * f_d_xm / (f_d_prime_xm * f_d_x)) 
    # y_d_prime_m_d_prime = np.mean(y * k_d_prime_t / f_d_prime_x) 
    
    direct_effect = y_d_prime_m_d - y_d_m_d
    indirect_effect = y_d_prime_m_d_prime - y_d_prime_m_d
    total_effect = direct_effect + indirect_effect

    causal_effects = {
      'total_effect': total_effect,
      'direct_effect': direct_effect,
      'indirect_effect': indirect_effect,
      'mediated_response': y_d_m_d_prime,
      'variance': 0,
      'margin_error': 0
    }
    return causal_effects
  
