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
:math:`p` [1]_.

For a Newtonian hamiltonian dynamic the kinetic energy is given by:

.. math::
    K(p) = \frac{1}{2} p^T M^{-1} p

We can also generate a relativistic dynamic [2]_.

References
----------
.. [1]: Betancourt, Michael, et al. "The geometric foundations of hamiltonian
        monte carlo." Bernoulli 23.4A (2017): 2257-2298.
.. [2]: Lu, Xiaoyu, et al. "Relativistic monte carlo." Artificial Intelligence and Statistics. PMLR, 2017.

"""
from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree

from blackjax.types import Array, PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["gaussian_euclidean"]

EuclideanKineticEnergy = Callable[[PyTree], float]


def gaussian_euclidean(
    inverse_mass_matrix: Array,
) -> Tuple[Callable, EuclideanKineticEnergy, Callable]:
    r"""Hamiltonian dynamic on euclidean manifold with normally-distributed momentum.

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

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.

    """
    ndim = jnp.ndim(inverse_mass_matrix)  # type: ignore[arg-type]
    shape = jnp.shape(inverse_mass_matrix)[:1]  # type: ignore[arg-type]

    if ndim == 1:  # diagonal mass matrix
        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))
        matmul = jnp.multiply

    elif ndim == 2:
        tril_inv = jscipy.linalg.cholesky(inverse_mass_matrix)
        identity = jnp.identity(shape[0])
        mass_matrix_sqrt = jscipy.linalg.solve_triangular(
            tril_inv, identity, lower=True
        )
        matmul = jnp.matmul

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(inverse_mass_matrix)}."  # type: ignore[arg-type]
        )

    def momentum_generator(rng_key: PRNGKey, position: PyTree) -> PyTree:
        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(momentum: PyTree) -> float:
        momentum, _ = ravel_pytree(momentum)
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy_val = 0.5 * jnp.dot(velocity, momentum)
        return kinetic_energy_val

    def is_turning(
        momentum_left: PyTree, momentum_right: PyTree, momentum_sum: PyTree
    ) -> bool:
        """Generalized U-turn criterion.

        Parameters
        ----------
        momentum_left
            Momentum of the leftmost point of the trajectory.
        momentum_right
            Momentum of the rightmost point of the trajectory.
        momentum_sum
            Sum of the momenta along the trajectory.

        .. [1]: Betancourt, Michael J. "Generalizing the no-U-turn sampler to Riemannian manifolds." arXiv preprint arXiv:1304.1920 (2013).
        .. [2]: "NUTS misses U-turn, runs in cicles until max depth", Stan Discourse Forum
                https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/46

        """
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

    return momentum_generator, kinetic_energy, is_turning
