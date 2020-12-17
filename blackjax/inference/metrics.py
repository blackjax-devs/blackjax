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
import jax.scipy as scipy
from jax.tree_util import tree_flatten

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
        One of two-dimensional array corresponding respectively to a diagonal
        or dense mass matrix.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.

    """
    ndim = jnp.ndim(inverse_mass_matrix)
    shape = jnp.shape(inverse_mass_matrix)[:1]

    if ndim == 1:  # diagonal mass matrix

        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))

        def momentum_generator(rng_key: jax.random.PRNGKey, position: PyTree) -> PyTree:
            _, treedef = tree_flatten(position)
            std = jax.random.normal(rng_key, shape)
            momentum = jnp.multiply(std, mass_matrix_sqrt)
            return treedef.unflatten(momentum)

        def kinetic_energy(momentum: PyTree, *_) -> float:
            momentum, _ = tree_flatten(momentum)
            velocity = jnp.multiply(inverse_mass_matrix, momentum)
            return 0.5 * jnp.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    elif ndim == 2:

        mass_matrix_sqrt = cholesky_of_inverse(inverse_mass_matrix)

        def momentum_generator(rng_key: jax.random.PRNGKey, position: PyTree) -> PyTree:
            _, treedef = tree_flatten(position)
            std = jax.random.normal(rng_key, shape)
            momentum = jnp.dot(std, mass_matrix_sqrt)
            return treedef.unflatten(momentum)

        def kinetic_energy(momentum: PyTree, *_) -> float:
            momentum, _ = tree_flatten(momentum)
            velocity = jnp.matmul(inverse_mass_matrix, momentum)
            return 0.5 * jnp.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.dim(inverse_mass_matrix)}."
        )


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def cholesky_of_inverse(matrix):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = jnp.swapaxes(
        jnp.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = jnp.broadcast_to(jnp.identity(matrix.shape[-1]), tril_inv.shape)
    return scipy.linalg.solve_triangular(tril_inv, identity, lower=True)
