"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np
import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from cde.density_estimator.LSCDE import LSConditionalDensityEstimation
from cde.density_estimator.CKDE import ConditionalKernelDensityEstimation
from cde.density_estimator.Gaussian import GaussianDensityEstimation


def get_density_by_name(settings):
    if settings['density'] == 'ls_conditional':
        return LSConditionalDensityEstimation(center_sampling_method='all', regularization=1, bandwidth=0.2)
    elif settings['density'] == 'conditional_kernel':
        return ConditionalKernelDensityEstimation(bandwidth='normal_reference')
    elif settings['density'] == 'gaussian':
        return GaussianDensityEstimation()
    else:
        raise NotImplementedError    
    
