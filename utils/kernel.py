import jax.scipy as jsp
import jax.numpy as jnp
from jax import grad, hessian
from jax import jit
import jax as jax
from functools import partial
import numpy as np
from scipy.spatial.distance import cdist

@jax.jit
def sqeuclidean_distance(x, y):
    return jnp.sum((x-y)**2)

# RBF Kernel
@jax.jit
def rbf_kernel(gamma, x, y):
    return jnp.exp( - gamma * sqeuclidean_distance(x, y))

# Exponential Kernel
@jax.jit
def exp_kernel(gamma, x, y):
    return jnp.exp( - gamma * jnp.sqrt(sqeuclidean_distance(x, y)))

@jax.jit
def linear_kernel(gamma, x, y):
    return gamma + jnp.dot(x, y)

@jax.jit
def polynomial_kernel(gamma, x, y):
    return (gamma + jnp.dot(x, y))**2

@jax.jit
def rational_quadratic_kernel(param, x, y):
    l, alpha = param
    gamma = 1/(2* alpha * l ** 2)
    return  jnp.power(1 + gamma * sqeuclidean_distance(x, y), alpha)

def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)

class Kernel:

    def __init__(self, settings):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.context_dimension = 1
        self._param = settings['bandwidth']

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, X1, X2):
        pass

    def fit(self, X):
        pass

class Gaussian(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Gaussian, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._sigma = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(rbf_kernel, 1/(2* self._sigma ** 2),X1,X2)
    
    def fit(self, X):
        """
        Args:
            X (np.ndarray)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        dists = cdist(X, X, 'sqeuclidean')
        self._sigma = max(np.median(dists), self._param)

class Exponential(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Exponential, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._alpha = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(exp_kernel, self._alpha, X1, X2)


class Linear(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Linear, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._sigma0 = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(linear_kernel, self._sigma0, X1, X2)

class Polynomial(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Polynomial, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._sigma0 = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(polynomial_kernel, self._sigma0, X1, X2)

class RationalQuadratic(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(RationalQuadratic, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._l = 1
        self._alpha = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        param = self._l, self._alpha
        return gram(rational_quadratic_kernel, param, X1, X2)