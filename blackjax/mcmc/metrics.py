# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Metric space in which the Hamiltonian dynamic is embedded.

An important particular case (and the most used in practice) of metric for the
position space in the Euclidean metric. It is defined by a definite positive
matrix :math:`M` with fixed value so that the kinetic energy of the hamiltonian
dynamic is independent of the position and only depends on the momentum
:math:`p` :cite:p:`betancourt2017geometric`.

For a Newtonian hamiltonian dynamic the kinetic energy is given by:

.. math::

    K(p) = \frac{1}{2} p^T M^{-1} p

We can also generate a relativistic dynamic :cite:p:`lu2017relativistic`.

"""
from typing import Callable, NamedTuple, Optional, Protocol, Union

import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree
from jax.scipy import stats as sp_stats

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["default_metric", "gaussian_euclidean", "gaussian_riemannian"]


class KineticEnergy(Protocol):
    def __call__(
        self, momentum: ArrayLikeTree, position: Optional[ArrayLikeTree] = None
    ) -> float:
        ...


class CheckTurning(Protocol):
    def __call__(
        self,
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: Optional[ArrayLikeTree] = None,
        position_right: Optional[ArrayLikeTree] = None,
    ) -> bool:
        ...


class Metric(NamedTuple):
    sample_momentum: Callable[[PRNGKey, ArrayLikeTree], ArrayLikeTree]
    kinetic_energy: KineticEnergy
    check_turning: CheckTurning


MetricTypes = Union[Metric, Array, Callable[[ArrayLikeTree], Array]]


def default_metric(metric: MetricTypes) -> Metric:
    """Convert an input metric into a ``Metric`` object following sensible default rules

    The metric can be specified in three different ways:

    - A ``Metric`` object that implements the full interface
    - An ``Array`` which is assumed to specify the inverse mass matrix of a static
      metric
    - A function that takes a coordinate position and returns the mass matrix at that
      location
    """
    if isinstance(metric, Metric):
        return metric

    # If the argument is a callable, we assume that it returns the mass matrix
    # at the given position and return the corresponding Riemannian metric.
    if callable(metric):
        return gaussian_riemannian(metric)

    # If we make it here then the argument should be an array, and we'll assume
    # that it specifies a static inverse mass matrix.
    return gaussian_euclidean(metric)


def gaussian_euclidean(
    inverse_mass_matrix: Array,
) -> Metric:
    r"""Hamiltonian dynamic on euclidean manifold with normally-distributed momentum
    :cite:p:`betancourt2013general`.

    The gaussian euclidean metric is a euclidean metric further characterized
    by setting the conditional probability density :math:`\pi(momentum|position)`
    to follow a standard gaussian distribution. A Newtonian hamiltonian
    dynamics is assumed.

    Parameters
    ----------
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

    Returns
    -------
    momentum_generator
        A function that generates a value for the momentum at random.
    kinetic_energy
        A function that returns the kinetic energy given the momentum.
    is_turning
        A function that determines whether a trajectory is turning back on
        itself given the values of the momentum along the trajectory.

    """
    ndim = jnp.ndim(inverse_mass_matrix)  # type: ignore[arg-type]
    shape = jnp.shape(inverse_mass_matrix)[:1]  # type: ignore[arg-type]

    if ndim == 1:  # diagonal mass matrix
        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))
        matmul = jnp.multiply

    elif ndim == 2:
        # inverse mass matrix can be factored into L*L.T. We want the cholesky
        # factor (inverse of L.T) of the mass matrix.
        L = jscipy.linalg.cholesky(inverse_mass_matrix, lower=True)
        identity = jnp.identity(shape[0])
        mass_matrix_sqrt = jscipy.linalg.solve_triangular(
            L, identity, lower=True, trans=True
        )
        # Note that mass_matrix_sqrt is a upper triangular matrix here, with
        #   jscipy.linalg.inv(mass_matrix_sqrt @ mass_matrix_sqrt.T)
        #   == inverse_mass_matrix
        # An alternative is to compute directly the cholesky factor of the inverse mass
        # matrix
        #   mass_matrix_sqrt = jscipy.linalg.cholesky(
        #       jscipy.linalg.inv(inverse_mass_matrix), lower=True)
        # which the result would instead be a lower triangular matrix.
        matmul = jnp.matmul

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {ndim}."
        )

    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(
        momentum: ArrayLikeTree, position: Optional[ArrayLikeTree] = None
    ) -> float:
        del position
        momentum, _ = ravel_pytree(momentum)
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy_val = 0.5 * jnp.dot(velocity, momentum)
        return kinetic_energy_val

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: Optional[ArrayLikeTree] = None,
        position_right: Optional[ArrayLikeTree] = None,
    ) -> bool:
        """Generalized U-turn criterion :cite:p:`betancourt2013generalizing,nuts_uturn`.

        Parameters
        ----------
        momentum_left
            Momentum of the leftmost point of the trajectory.
        momentum_right
            Momentum of the rightmost point of the trajectory.
        momentum_sum
            Sum of the momenta along the trajectory.

        """
        del position_left, position_right

        m_left, _ = ravel_pytree(momentum_left)
        m_right, _ = ravel_pytree(momentum_right)
        m_sum, _ = ravel_pytree(momentum_sum)

        velocity_left = matmul(inverse_mass_matrix, m_left)
        velocity_right = matmul(inverse_mass_matrix, m_right)

        # rho = m_sum
        rho = m_sum - (m_right + m_left) / 2
        turning_at_left = jnp.dot(velocity_left, rho) <= 0
        turning_at_right = jnp.dot(velocity_right, rho) <= 0
        return turning_at_left | turning_at_right

    return Metric(momentum_generator, kinetic_energy, is_turning)


def gaussian_riemannian(
    mass_matrix_fn: Callable,
) -> Metric:
    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayLikeTree:
        mass_matrix = mass_matrix_fn(position)
        ndim = jnp.ndim(mass_matrix)
        if ndim == 1:
            mass_matrix_sqrt = jnp.sqrt(mass_matrix)
        elif ndim == 2:
            mass_matrix_sqrt = jscipy.linalg.cholesky(mass_matrix, lower=True)
        else:
            raise ValueError(
                "The mass matrix has the wrong number of dimensions:"
                f" expected 1 or 2, got {jnp.ndim(mass_matrix)}."
            )

        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(
        momentum: ArrayLikeTree, position: Optional[ArrayLikeTree] = None
    ) -> float:
        if position is None:
            raise ValueError(
                "A Reinmannian kinetic energy function must be called with the "
                "position specified; make sure to use a Reinmannian-capable "
                "integrator like `implicit_midpoint`."
            )

        momentum, _ = ravel_pytree(momentum)
        mass_matrix = mass_matrix_fn(position)
        ndim = jnp.ndim(mass_matrix)
        if ndim == 1:
            return -jnp.sum(sp_stats.norm.logpdf(momentum, 0.0, jnp.sqrt(mass_matrix)))
        elif ndim == 2:
            return -sp_stats.multivariate_normal.logpdf(
                momentum, jnp.zeros_like(momentum), mass_matrix
            )
        else:
            raise ValueError(
                "The mass matrix has the wrong number of dimensions:"
                f" expected 1 or 2, got {jnp.ndim(mass_matrix)}."
            )

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: Optional[ArrayLikeTree] = None,
        position_right: Optional[ArrayLikeTree] = None,
    ) -> bool:
        del momentum_left, momentum_right, momentum_sum, position_left, position_right
        raise NotImplementedError(
            "NUTS sampling is not yet implemented for Riemannian manifolds"
        )

        # Here's a possible implementation of this function, but the NUTS
        # proposal will require some refactoring to work properly, since we need
        # to be able to access the coordinates at the left and right endpoints
        # to compute the mass matrix at those points.

        # m_left, _ = ravel_pytree(momentum_left)
        # m_right, _ = ravel_pytree(momentum_right)
        # m_sum, _ = ravel_pytree(momentum_sum)

        # mass_matrix_left = mass_matrix_fn(position_left)
        # mass_matrix_right = mass_matrix_fn(position_right)
        # velocity_left = jnp.linalg.solve(mass_matrix_left, m_left)
        # velocity_right = jnp.linalg.solve(mass_matrix_right, m_right)

        # # rho = m_sum
        # rho = m_sum - (m_right + m_left) / 2
        # turning_at_left = jnp.dot(velocity_left, rho) <= 0
        # turning_at_right = jnp.dot(velocity_right, rho) <= 0
        # return turning_at_left | turning_at_right

    return Metric(momentum_generator, kinetic_energy, is_turning)
