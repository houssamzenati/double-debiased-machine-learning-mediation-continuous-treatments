from data.gaussian import logit_normal_pdf

import numpy as np

def instantiate_metrics():
    """
    Instantiates metrics dictionary with metric keys

    Returns
    -------
        dictionary
    """
    return {
        'error_total_effect': None,
        'error_direct_effect': None,
        'error_indirect_effect': None,
    }

def compute_errors(causal_effects, true_causal_effects):
    """
    Computes errors between true and estimated causal effects

    Parameters
    ----------
    true_causal_effects : dictionary

    causal_effects : dictionary
            estimated causal effect
    
    Returns
    -------
    float
            error on total effect
    float
            error on direct effect treated (\theta(1))
    float 
            error on direct effect control (\theta(0))
    float
            error on indirect effect treated (\delta(1))
    float 
            error on indirect effect control (\delta(0))
    """    

    true_total, true_direct, true_indirect = true_causal_effects
    error_total = abs(true_total-causal_effects['total_effect'])
    error_direct = abs(true_direct - causal_effects['direct_effect'])
    error_indirect = abs(true_indirect - causal_effects['indirect_effect'])
    relative_error_total = abs(error_total/true_total)
    relative_error_direct = abs(error_direct/true_direct)
    relative_error_indirect = abs(error_indirect/true_indirect)


    errors = {
        'error_total':error_total,
        'error_direct': error_direct,
        'error_indirect': error_indirect,
        'relative_error_total': relative_error_total, 
        'relative_error_direct': relative_error_direct,
        'relative_error_indirect': relative_error_indirect
    }
    return errors

def compute_score(s_direct, s_indirect):

    x_centers = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    y_centers = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    pdf_values = logit_normal_pdf(x_centers)[:, np.newaxis] * logit_normal_pdf(y_centers)[np.newaxis, :]

    score_direct = np.sum(pdf_values * s_direct) / np.sum(pdf_values)
    score_indirect = np.sum(pdf_values * s_indirect) / np.sum(pdf_values)

    return score_direct + score_indirect

def causal_experiment(estimator, t, m, x, y, data_settings, params):


    ### Estimate causal effects
    n_d = 6
    d = np.linspace(0, 1, n_d)
    # y = np.linspace(0, 1, ny)
    d_v, d_prime_v = np.meshgrid(d, d, indexing='ij')

    surface_total = np.zeros((n_d, n_d))
    surface_direct = np.zeros((n_d, n_d))
    surface_indirect = np.zeros((n_d, n_d))
    errors_total = np.zeros((n_d, n_d))
    errors_direct = np.zeros((n_d, n_d))
    errors_indirect = np.zeros((n_d, n_d))
    relative_errors_total = np.zeros((n_d, n_d))
    relative_errors_direct = np.zeros((n_d, n_d))
    relative_errors_indirect = np.zeros((n_d, n_d))

    x_centers = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    y_centers = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    covariate_dimension = x.shape[1]
    mean1 = np.zeros(covariate_dimension)
    cov1 = np.eye(covariate_dimension) 
    loc = (mean1.reshape(1,-1).dot(params['alpha_X'] * params['omega_X_1']) + params['alpha_0']).squeeze()
    var = (params['alpha_X'] * params['omega_X_1']).T.dot(cov1.dot(params['alpha_X'] * params['omega_X_1'])).squeeze() + SIGMA_T**2
    std_dev = np.sqrt(var)
    pdf_values = norm(loc=loc, scale=std_dev).pdf(x_centers)[:, np.newaxis] * norm(loc=loc, scale=std_dev).pdf(y_centers)[np.newaxis, :]
    # pdf_values = logit_normal_pdf(x_centers)[:, np.newaxis] * logit_normal_pdf(y_centers)[np.newaxis, :]

    for i in range(n_d):
        for j in range(n_d):
            if i != j:
                d = d_v[i,j]
                d_prime = d_prime_v[i,j]
                causal_effects = estimator.estimate(d, d_prime, t, m, x, y)
                true_causal_effects = get_causal_effects(x, d, d_prime, data_settings, 
                                params,
                                mode='id')
                errors = compute_errors(causal_effects, true_causal_effects)
                errors_total[i,j] = errors['error_total']
                errors_direct[i,j] = errors['error_direct']
                errors_indirect[i,j] = errors['error_indirect']
                relative_errors_total[i,j] = errors['relative_error_total']
                relative_errors_direct[i,j] = errors['relative_error_direct']
                relative_errors_indirect[i,j] = errors['relative_error_indirect']
                surface_total[i,j] = errors['error_total']
                surface_direct[i,j] = errors['error_direct']
                surface_indirect[i,j] = errors['error_indirect']

    metrics = {
        'error_total' : np.sum(pdf_values * errors_total) / np.sum(pdf_values),
        'error_direct' : np.sum(pdf_values * errors_direct) / np.sum(pdf_values),
        'error_indirect' : np.sum(pdf_values * errors_indirect) / np.sum(pdf_values),
        'relative_error_total' : np.sum(pdf_values * relative_errors_total) / np.sum(pdf_values),
        'relative_error_direct' : np.sum(pdf_values * relative_errors_direct) / np.sum(pdf_values),
        'relative_error_indirect' : np.sum(pdf_values * relative_errors_indirect) / np.sum(pdf_values),
    }
    display_experiment_results(metrics)
    surfaces = {
        'total': surface_total,
        'direct': surface_direct,
        'indirect': surface_indirect
    }

    return metrics, surfaces