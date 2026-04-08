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
from typing import Callable, NamedTuple, Protocol, TypeAlias

import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree

from blackjax.types import Array, ArrayLikeTree, ArrayTree, Numeric, PRNGKey
from blackjax.util import generate_gaussian_noise, linear_map

__all__ = [
    "default_metric",
    "gaussian_euclidean",
    "gaussian_euclidean_low_rank",
    "gaussian_riemannian",
]


class KineticEnergy(Protocol):
    def __call__(
        self, momentum: ArrayLikeTree, position: ArrayLikeTree | None = None
    ) -> Numeric:
        ...


class CheckTurning(Protocol):
    def __call__(
        self,
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: ArrayLikeTree | None = None,
        position_right: ArrayLikeTree | None = None,
    ) -> bool:
        ...


class Scale(Protocol):
    def __call__(
        self,
        position: ArrayLikeTree,
        element: ArrayLikeTree,
        *,
        inv: bool,
        trans: bool,
    ) -> ArrayLikeTree:
        ...


class Metric(NamedTuple):
    sample_momentum: Callable[[PRNGKey, ArrayLikeTree], ArrayLikeTree]
    kinetic_energy: KineticEnergy
    check_turning: CheckTurning
    scale: Scale


MetricTypes: TypeAlias = Metric | Array | Callable[[ArrayLikeTree], Array]


def default_metric(metric: MetricTypes) -> Metric:
    """Convert an input metric into a ``Metric`` object following sensible default rules.

    The metric can be specified in three different ways:

    - A ``Metric`` object that implements the full interface
    - An ``Array`` which is assumed to specify the inverse mass matrix of a static
      metric
    - A function that takes a coordinate position and returns the mass matrix at that
      location

    Returns
    -------
    A ``Metric`` object with ``sample_momentum``, ``kinetic_energy``,
    ``check_turning``, and ``scale`` fields.
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
    mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = _format_covariance(
        inverse_mass_matrix, is_inv=True
    )

    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(
        momentum: ArrayLikeTree, position: ArrayLikeTree | None = None
    ) -> Numeric:
        del position
        momentum, _ = ravel_pytree(momentum)
        velocity = linear_map(inverse_mass_matrix, momentum)
        kinetic_energy_val = 0.5 * jnp.dot(velocity, momentum)
        return kinetic_energy_val

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: ArrayLikeTree | None = None,
        position_right: ArrayLikeTree | None = None,
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

        velocity_left = linear_map(inverse_mass_matrix, m_left)
        velocity_right = linear_map(inverse_mass_matrix, m_right)

        # rho = m_sum
        rho = m_sum - (m_right + m_left) / 2
        turning_at_left = jnp.dot(velocity_left, rho) <= 0
        turning_at_right = jnp.dot(velocity_right, rho) <= 0
        return turning_at_left | turning_at_right

    def scale(
        position: ArrayLikeTree,
        element: ArrayLikeTree,
        *,
        inv: bool,
        trans: bool,
    ) -> ArrayLikeTree:
        """Scale elements by the mass matrix.

        Parameters
        ----------
        position
            The current position. Not used in this metric.
        elements
            Elements to scale
        inv
            Whether to scale the elements by the inverse mass matrix or the mass matrix.
            If True, the element is scaled by the inverse square root mass matrix, i.e., elem <- (M^{1/2})^{-1} elem.
        trans
            whether to transpose mass matrix when scaling

        Returns
        -------
        scaled_elements
            The scaled elements.
        """

        ravelled_element, unravel_fn = ravel_pytree(element)

        if inv:
            left_hand_side_matrix = inv_mass_matrix_sqrt
        else:
            left_hand_side_matrix = mass_matrix_sqrt
        if trans:
            left_hand_side_matrix = left_hand_side_matrix.T

        scaled = linear_map(left_hand_side_matrix, ravelled_element)

        return unravel_fn(scaled)

    return Metric(momentum_generator, kinetic_energy, is_turning, scale)


def gaussian_euclidean_low_rank(
    sigma: Array,
    U: Array,
    lam: Array,
) -> Metric:
    r"""Euclidean metric with low-rank-modified mass matrix :cite:p:`sountsov2025preconditioning`.

    The inverse mass matrix has the form

    .. math::

        M^{-1} = \operatorname{diag}(\sigma)
                 \bigl(I + U(\Lambda - I)U^\top\bigr)
                 \operatorname{diag}(\sigma)

    where :math:`\sigma \in \mathbb{R}^d_{>0}` is a diagonal scaling,
    :math:`U \in \mathbb{R}^{d \times k}` has orthonormal columns, and
    :math:`\Lambda = \operatorname{diag}(\lambda)` with :math:`\lambda > 0`.
    When :math:`\lambda = \mathbf{1}` the metric reduces to a diagonal metric
    with scale :math:`\sigma`.  All HMC operations are :math:`O(dk)`, making
    this efficient when :math:`k \ll d`.

    Parameters
    ----------
    sigma
        Shape ``(d,)``.  Positive diagonal scaling; plays the role of marginal
        standard deviations.
    U
        Shape ``(d, k)``.  Matrix with orthonormal columns spanning the
        low-rank correction subspace.
    lam
        Shape ``(k,)``.  Positive eigenvalues for the low-rank correction.

    Returns
    -------
    A ``Metric`` object whose operations all run in :math:`O(dk)`.
    """
    inv_sigma = 1.0 / sigma  # (d,)
    sqrt_lam = jnp.sqrt(lam)  # (k,)
    inv_sqrt_lam = 1.0 / sqrt_lam  # (k,)

    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        # Sample eps ~ N(0, I) with the same pytree shape as position.
        noise = generate_gaussian_noise(rng_key, position)
        eps, unravel_fn = ravel_pytree(noise)
        # p = M^{1/2} eps  where  M^{1/2} = D^{-1} B,
        # B = I + U (Λ^{-1/2} - I) U^T,  D^{-1} = diag(1/σ).
        # D^{-1} is applied AFTER B so that E[pp^T] = D^{-1} B B D^{-1} = M.
        v = eps + U @ ((inv_sqrt_lam - 1.0) * (U.T @ eps))  # B eps
        p = inv_sigma * v  # D^{-1} B eps
        return unravel_fn(p)

    def kinetic_energy(
        momentum: ArrayLikeTree, position: ArrayLikeTree | None = None
    ) -> Numeric:
        del position
        p, _ = ravel_pytree(momentum)
        # K(p) = ½ p^T M^{-1} p = ½ q^T (I + U(Λ-I)U^T) q,  q = σ⊙p
        q = sigma * p  # (d,)
        alpha = U.T @ q  # (k,)
        return 0.5 * (jnp.dot(q, q) + jnp.dot(alpha, (lam - 1.0) * alpha))

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: ArrayLikeTree | None = None,
        position_right: ArrayLikeTree | None = None,
    ) -> bool:
        del position_left, position_right
        m_left, _ = ravel_pytree(momentum_left)
        m_right, _ = ravel_pytree(momentum_right)
        m_sum, _ = ravel_pytree(momentum_sum)

        def _inv_mass_times(p):
            # M^{-1} p = D(I + U(Λ-I)U^T)D p
            q = sigma * p
            return sigma * (q + U @ ((lam - 1.0) * (U.T @ q)))

        vel_left = _inv_mass_times(m_left)
        vel_right = _inv_mass_times(m_right)
        rho = m_sum - (m_right + m_left) / 2
        return (jnp.dot(vel_left, rho) <= 0) | (jnp.dot(vel_right, rho) <= 0)

    def scale(
        position: ArrayLikeTree,
        element: ArrayLikeTree,
        *,
        inv: bool,
        trans: bool,
    ) -> ArrayLikeTree:
        """Scale an element by the (inverse) (transposed) square-root mass matrix.

        M = D^{-1} C D^{-1} where C = I+U(Λ^{-1}-I)U^T and D = diag(σ).
        The (non-symmetric) left square root is M^{1/2} = D^{-1} B where
        B = I+U(Λ^{-1/2}-I)U^T, so M = M^{1/2} (M^{1/2})^T.

        * ``inv=False, trans=False``:  M^{1/2}    = D^{-1} B
        * ``inv=False, trans=True`` :  (M^{1/2})^T = B D^{-1}
        * ``inv=True,  trans=False``:  M^{-1/2}   = D A^{*}  where A^{*} = I+U(√Λ-I)U^T
        * ``inv=True,  trans=True`` :  (M^{-1/2})^T = A^{*} D
        """
        del position
        e, unravel_fn = ravel_pytree(element)

        if not inv and not trans:
            # M^{1/2} e = D^{-1} (B e) = (1/σ) * (B e)
            v = e + U @ ((inv_sqrt_lam - 1.0) * (U.T @ e))
            scaled = inv_sigma * v
        elif not inv and trans:
            # (M^{1/2})^T e = B (D^{-1} e) = B (e/σ)
            q = inv_sigma * e
            scaled = q + U @ ((inv_sqrt_lam - 1.0) * (U.T @ q))
        elif inv and not trans:
            # M^{-1/2} e = D (A^{*} e) = σ * (A^{*} e)  where A^{*} = I+U(√Λ-I)U^T
            v = e + U @ ((sqrt_lam - 1.0) * (U.T @ e))
            scaled = sigma * v
        else:
            # (M^{-1/2})^T e = A^{*} (D e) = A^{*} (σ⊙e)
            q = sigma * e
            scaled = q + U @ ((sqrt_lam - 1.0) * (U.T @ q))

        return unravel_fn(scaled)

    return Metric(momentum_generator, kinetic_energy, is_turning, scale)


def gaussian_riemannian(
    mass_matrix_fn: Callable,
) -> Metric:
    """Hamiltonian dynamic on Riemannian manifold with normally-distributed momentum.

    Parameters
    ----------
    mass_matrix_fn
        A callable that takes a position and returns the mass matrix at that
        location (positive definite, one or two-dimensional array).

    Returns
    -------
    A ``Metric`` object with ``sample_momentum``, ``kinetic_energy``,
    ``check_turning``, and ``scale`` fields.
    """

    def momentum_generator(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayLikeTree:
        mass_matrix = mass_matrix_fn(position)
        mass_matrix_sqrt, *_ = _format_covariance(mass_matrix, is_inv=False)

        return generate_gaussian_noise(rng_key, position, sigma=mass_matrix_sqrt)

    def kinetic_energy(
        momentum: ArrayLikeTree, position: ArrayLikeTree | None = None
    ) -> Numeric:
        if position is None:
            raise ValueError(
                "A Riemannian kinetic energy function must be called with the "
                "position specified; make sure to use a Riemannian-capable "
                "integrator like `implicit_midpoint`."
            )

        momentum, _ = ravel_pytree(momentum)
        mass_matrix = mass_matrix_fn(position)
        sqrt_mass_matrix, inv_sqrt_mass_matrix, diag = _format_covariance(
            mass_matrix, is_inv=False
        )

        return _energy(momentum, 0, sqrt_mass_matrix, inv_sqrt_mass_matrix.T, diag)

    def is_turning(
        momentum_left: ArrayLikeTree,
        momentum_right: ArrayLikeTree,
        momentum_sum: ArrayLikeTree,
        position_left: ArrayLikeTree | None = None,
        position_right: ArrayLikeTree | None = None,
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

    def scale(
        position: ArrayLikeTree,
        element: ArrayLikeTree,
        *,
        inv: bool,
        trans: bool,
    ) -> ArrayLikeTree:
        """Scale elements by the mass matrix.

        Parameters
        ----------
        position
            The current position.

        Returns
        -------
        scaled_elements
            The scaled elements.
        """
        mass_matrix = mass_matrix_fn(position)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = _format_covariance(
            mass_matrix, is_inv=False
        )
        ravelled_element, unravel_fn = ravel_pytree(element)

        if inv:
            left_hand_side_matrix = inv_mass_matrix_sqrt
        else:
            left_hand_side_matrix = mass_matrix_sqrt
        if trans:
            left_hand_side_matrix = left_hand_side_matrix.T

        scaled = linear_map(left_hand_side_matrix, ravelled_element)

        return unravel_fn(scaled)

    return Metric(momentum_generator, kinetic_energy, is_turning, scale)


def _format_covariance(cov: Array, is_inv):
    ndim = jnp.ndim(cov)
    if ndim == 1:
        cov_sqrt = jnp.sqrt(cov)
        inv_cov_sqrt = 1 / cov_sqrt
        diag = lambda x: x
        if is_inv:
            inv_cov_sqrt, cov_sqrt = cov_sqrt, inv_cov_sqrt
    elif ndim == 2:
        identity = jnp.identity(cov.shape[0])
        if is_inv:
            inv_cov_sqrt = jscipy.linalg.cholesky(cov, lower=True)
            cov_sqrt = jscipy.linalg.solve_triangular(
                inv_cov_sqrt, identity, lower=True, trans=True
            )
        else:
            cov_sqrt = jscipy.linalg.cholesky(cov, lower=False).T
            inv_cov_sqrt = jscipy.linalg.solve_triangular(
                cov_sqrt, identity, lower=True, trans=True
            )

        diag = lambda x: jnp.diag(x)

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(cov)}."
        )
    return cov_sqrt, inv_cov_sqrt, diag


def _energy(x, mean, cov_sqrt, inv_cov_sqrt, diag):
    d = x.shape[0]
    z = linear_map(inv_cov_sqrt, x - mean)
    const = jnp.sum(jnp.log(diag(cov_sqrt))) + d / 2 * jnp.log(2 * jnp.pi)
    return 0.5 * jnp.sum(z**2) + const
