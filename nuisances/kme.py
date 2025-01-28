import jax.numpy as jnp
import numpy as np

from utils.kernel import (Exponential, Gaussian, Linear, Polynomial,
                          RationalQuadratic)

settings = {
    'reg_lambda': 1e-5,
    'bandwidth': 1
}

def _get_kernel(settings):
    if settings['kernel'] == 'gauss':
        kernel = Gaussian(settings)
        return kernel
    elif settings['kernel'] == 'exp':
        kernel = Exponential(settings)
        return kernel
    elif settings['kernel'] == 'linear':
        return Linear(settings)
    elif settings['kernel'] == 'polynomial':
        return Polynomial(settings)
    elif settings['kernel'] == 'rq':
        return RationalQuadratic(settings)
    else:
        raise NotImplementedError

def _kme_cross_conditional_mean_outcomes(d, d_prime, y, t, m, x, settings):
    """
    Estimate the cross conditional mean outcome

    Parameters
    ----------
    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    settings dictionary
            parameters for the kernel

    Returns
    -------
    mu_0m : array-like, shape (n_samples)
            conditional mean outcome for control group

    mu_1m : array-like, shape (n_samples)
            conditional mean outcome for treatment group

    psi_t0t0 : array-like, shape (n_samples)
            cross conditional mean outcome 
    
    psi_t0t1 : array-like, shape (n_samples)
            cross conditional mean outcome 
    
    psi_t1t0 : array-like, shape (n_samples)
            cross conditional mean outcome 
    
    psi_t1t1 : array-like, shape (n_samples)
            cross conditional mean outcome 
    """
    n = x.shape[0]

    reg_lambda = settings['reg_lambda']
    reg_lambda_1 = settings['reg_lambda_tilde']

    kernel = _get_kernel(settings)

    K_T = kernel.gram_matrix(t)
    K_M = kernel.gram_matrix(m)
    K_X = kernel.gram_matrix(x)

    K_T_M_X = K_T * K_M * K_X + n * reg_lambda * jnp.eye(n)
    K_T_X = K_T * K_X + n * reg_lambda_1 * jnp.eye(n)

    inv_K_T_M_X = np.linalg.inv(K_T_M_X)
    inv_K_T_X = np.linalg.inv(K_T_X)

    K_M_inv_K_T_X = np.dot(K_M, inv_K_T_X)
    Y_inv_K_T_M_X = np.dot(y, inv_K_T_M_X)

    t1 = np.array([1])
    t0 = np.array([0])

    d = d * np.array([1])
    d_prime = d_prime * np.array([1])

    K_Td = kernel.evaluate(t, d)
    K_Td_prime = kernel.evaluate(t, d_prime)

    #
    half_d = np.dot(K_M_inv_K_T_X, K_Td * K_X) * K_X
    half_d_prime = np.dot(K_M_inv_K_T_X, K_Td_prime * K_X) * K_X

    # eta_tt
    psi_d_d = np.dot(Y_inv_K_T_M_X, K_Td * half_d)

    # eta_td
    psi_d_d_prime = np.dot(Y_inv_K_T_M_X, K_Td_prime * half_d)

    # eta_dt
    psi_d_prime_d = np.dot(Y_inv_K_T_M_X, K_Td * half_d_prime)

    # eta_dd
    psi_d_prime_d_prime = np.dot(Y_inv_K_T_M_X, K_Td_prime * half_d_prime)

    mu_d = np.dot(Y_inv_K_T_M_X, K_Td * K_M * K_X)
    mu_d_prime = np.dot(Y_inv_K_T_M_X, K_Td_prime * K_M * K_X)

    return mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime


def _kme_conditional_mean_outcome(d, d_prime, y, t, m, x, settings):
    """
    Estimate the mediated response curve

    Parameters
    ----------
    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    settings dictionary
            parameters for the kernel

    Returns
    -------
    eta_t0t0 : array-like, shape (n_samples)
            mediated response curve 
    
    eta_t0t1 : array-like, shape (n_samples)
            mediated response curve 
    
    eta_t1t0 : array-like, shape (n_samples)
            mediated response curve 
    
    eta_t1t1 : array-like, shape (n_samples)
            mediated response curve 
    """
    _, _, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
        _kme_cross_conditional_mean_outcomes(d,
                                             d_prime,
                                             y,
                                             t,
                                             m,
                                             x,
                                             settings))

    # eta_t1t1
    eta_d_d = np.mean(psi_d_d)

    # eta_t1t0
    eta_d_d_prime = np.mean(psi_d_d_prime)

    # eta_t0t1
    eta_d_prime_d = np.mean(psi_d_prime_d)

    # eta_t0t0
    eta_d_prime_d_prime = np.mean(psi_d_prime_d_prime)

    return eta_d_d, eta_d_d_prime, eta_d_prime_d, eta_d_prime_d_prime