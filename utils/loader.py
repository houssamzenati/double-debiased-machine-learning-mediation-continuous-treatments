import os
import sys
import jax.scipy as jsp
import jax.numpy as jnp

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from estimators.ipw import ImportanceWeighting
from estimators.linear import Linear
from estimators.kme_g_computation import KMEGComputation
from estimators.kme_dml import KMEDoubleMachineLearning
from estimators.sani_dml import SaniDoubleMachineLearning
from estimators.sani_kme_dml import SaniKMEDoubleMachineLearning

def get_estimator_by_name(settings):
    if settings['estimator'] == 'ipw':
        return ImportanceWeighting
    elif settings['estimator'] == 'linear':
        return Linear
    elif settings['estimator'] == 'kme_g_computation':
        return KMEGComputation
    elif settings['estimator'] == 'kme_dml':
        return KMEDoubleMachineLearning
    elif settings['estimator'] == 'sani_dml':
        return SaniDoubleMachineLearning
    elif settings['estimator'] == 'sani_kme_dml':
        return SaniKMEDoubleMachineLearning
    else:
        raise NotImplementedError


    