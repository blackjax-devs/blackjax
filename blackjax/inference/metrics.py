r"""Metric space in which the Hamiltonian dynamic is embedded.

An important particular case (and the most used in practice) of metric for the
position space in the Euclidean metric. It is defined by a definite positive
matrix :math:`M` with fixed value so that the kinetic energy of the hamiltonian
dynamic is independent of the position and only depends on the momentum
:math:`p` [1]_.

For a Newtonian hamiltonian dynamic the kinetic energy is given by:

.. math::
    K(p) = \frac{1}{2} p^T M^{-1} p

References
----------
.. [1]: Betancourt, Michael, et al. "The geometric foundations of hamiltonian
        monte carlo." Bernoulli 23.4A (2017): 2257-2298.

"""
from typing import Callable, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree

__all__ = ["gaussian_euclidean"]

PyTree = Union[Dict, List, Tuple]
EuclideanKineticEnergy = Callable[[PyTree], float]


def gaussian_euclidean(
    inverse_mass_matrix: jnp.DeviceArray,
) -> Tuple[Callable, EuclideanKineticEnergy]:
    r"""Hamiltonian dynamic on euclidean manifold with normally-distributed momentum.

    The gaussian euclidean metric is a euclidean metric further characterized
    by setting the conditional probability density :math:`\pi(momentum|position)`
    to follow a standard gaussian distribution. A Newtonian hamiltonian
    dynamics is assumed.

    Arguments
    ---------
    inverse_mass_matrix
        One or two-dimensional array corresponding respectively to a diagonal
        or dense mass matrix. The inverse mass matrix is multiplied to a
        flattened version of the Pytree in which the chain position is stored
        (the current value of the random variables). The order of the variables
        should thus match JAX's tree flattening order, and more specifically
        that of `ravel_pytree`.
        In particular, JAX sorts dictionaries by key when flattening them. The
        value of each variables will appear in the flattened Pytree following
        the order given by `sort(keys)`.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.

    """
    ndim = jnp.ndim(inverse_mass_matrix)
    shape = jnp.shape(inverse_mass_matrix)[:1]

    if ndim > 2:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(inverse_mass_matrix)}."
        )

    elif ndim == 1:  # diagonal mass matrix
        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))
        dot, matmul = jnp.multiply, jnp.multiply

    elif ndim == 2:

        tril_inv = jscipy.linalg.cholesky(inverse_mass_matrix)
        identity = jnp.identity(shape[0])
        mass_matrix_sqrt = jscipy.linalg.solve_triangular(
            tril_inv, identity, lower=True
        )
        dot, matmul = jnp.dot, jnp.matmul

    def momentum_generator(rng_key: jax.random.PRNGKey, position: PyTree) -> PyTree:
        _, unravel_fn = ravel_pytree(position)
        std = jax.random.normal(rng_key, shape)
        momentum = dot(std, mass_matrix_sqrt)
        return unravel_fn(momentum)

    def kinetic_energy(momentum: PyTree, *_) -> float:
        momentum, _ = ravel_pytree(momentum)
        momentum = jnp.array(momentum)
        velocity = matmul(inverse_mass_matrix, momentum)
        return 0.5 * jnp.dot(velocity, momentum)

    return momentum_generator, kinetic_energy
