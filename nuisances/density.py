"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np
import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from sklearn.base import clone

from utils.utils import _get_train_test_lists, _get_interactions, is_array_integer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed


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
    

class ConditionalNearestNeighborsKDE(BaseEstimator):
    """Conditional Kernel Density Estimation using nearest neighbors.

    This class implements a Conditional Kernel Density Estimation by applying
    the Kernel Density Estimation algorithm after a nearest neighbors search.

    It allows the use of user-specified nearest neighbor and kernel density
    estimators or, if not provided, defaults will be used.

    Parameters
    ----------
    nn_estimator : NearestNeighbors instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.NearestNeighbors` class
        to use for finding nearest neighbors. If not specified, a
        `~sklearn.neighbors.NearestNeighbors` instance with `n_neighbors=100`
        will be used.

    kde_estimator : KernelDensity instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.KernelDensity` class
        to use for estimating the kernel density. If not specified, a
        `~sklearn.neighbors.KernelDensity` instance with `bandwidth="scott"`
        will be used.
    """

    def __init__(self, nn_estimator=None, kde_estimator=None):
        self.nn_estimator = nn_estimator
        self.kde_estimator = kde_estimator

    def fit(self, X, y=None):
        if self.nn_estimator is None:
            self.nn_estimator_ = NearestNeighbors(n_neighbors=100)
        else:
            self.nn_estimator_ = clone(self.nn_estimator)
        self.nn_estimator_.fit(X, y)
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the conditional density estimation of new samples.

        The predicted density of the target for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be estimated, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        kernel_density_list : list of len n_samples of KernelDensity instances
            Estimated conditional density estimations in the form of
            `~sklearn.neighbors.KernelDensity` instances.
        """
        _, ind_X = self.nn_estimator_.kneighbors(X)
        if self.kde_estimator is None:
            kernel_density_list = [
                KernelDensity(bandwidth="scott").fit(
                    self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        else:
            kernel_density_list = [
                clone(self.kde_estimator).fit(
                    self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        return kernel_density_list

    def pdf(self, x, y):

        ckde_preds = self.predict(x)

        def _evaluate_individual(y_, cde_pred):
            # The score_samples and score methods returns stuff on log scale,
            # so we have to exp it.
            expected_value = np.exp(cde_pred.score(y_.reshape(-1, 1)))
            return expected_value

        individual_predictions = Parallel(n_jobs=-1)(
            delayed(_evaluate_individual)(y_, cde_pred)
            for y_, cde_pred in zip(y, ckde_preds)
        )

        return np.array(individual_predictions)
