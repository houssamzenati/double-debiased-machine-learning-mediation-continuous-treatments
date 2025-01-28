import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import itertools
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp

from cde.utils.center_point_select import sample_center_points
from cde.utils.misc import norm_along_axis_1
from .BaseDensityEstimator import BaseDensityEstimator
from cde.utils.async_executor import execute_batch_async_pdf
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, rv_continuous

MULTIPROC_THRESHOLD = 10**4

class GaussianDensityEstimation(BaseDensityEstimator):
  """ Gaussian Density

  http://proceedings.mlr.press/v9/sugiyama10a.html

  Args:
      name: (str) name / identifier of estimator
      ndim_x: (int) dimensionality of x variable
      ndim_y: (int) dimensionality of y variable
      center_sampling_method: String that describes the method to use for finding kernel centers. Allowed values \
                            [all, random, distance, k_means, agglomerative]
      bandwidth: scale / bandwith of the gaussian kernels
      n_centers: Number of kernels to use in the output
      regularization: regularization / damping parameter for solving the least-squares problem
      keep_edges: if set to True, the extreme y values as centers are kept (for expressiveness)
      n_jobs: (int) number of jobs to launch for calls with large batch sizes
      random_seed: (optional) seed (int) of the random number generators used
    """

  def __init__(self, name='LSCDE', ndim_x=None, ndim_y=None, center_sampling_method='k_means',
               bandwidth=0.5, n_centers=500, regularization=1.0,
               keep_edges=True, n_jobs=-1, random_seed=None):

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    self.bandwidth = bandwidth
    self.regularization = regularization
    self.n_jobs = n_jobs

    self.fitted = False
    self.can_sample = False
    self.reg = LinearRegression()


  def fit(self, X, Y, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
    """
    # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
    self.reg.fit(X, Y)
    self.scale = np.sqrt(np.mean((Y - self.reg.predict(X))**2))
    self.fitted = True

  def representation(self, X):
    return self.reg.predict(X)

  def pdf(self, X, Y):
    """ Predicts the conditional density p(y|x). Requires the model to be fitted.

       Args:
         X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
         Y: numpy array of y targets - shape: (n_samples, n_dim_y)

       Returns:
          conditional probability density p(y|x) - numpy array of shape (n_query_samples, )

     """
    assert self.fitted, "model must be fitted for predictions"

    representation = self.representation(X)
    return norm(loc=representation, scale=self.scale).pdf(Y)

