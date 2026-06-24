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
"""Algorithms to adapt the mass matrix used by algorithms in the Hamiltonian
Monte Carlo family to the current geometry.

The Stan Manual :cite:p:`stan_hmc_param` is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayLike

__all__ = [
    "WelfordAlgorithmState",
    "MassMatrixAdaptationState",
    "mass_matrix_adaptation",
    "welford_algorithm",
]


class WelfordAlgorithmState(NamedTuple):
    """State carried through the Welford algorithm.

    mean
        The running sample mean.
    m2
        The running value of the sum of difference of squares. See documentation
        of the `welford_algorithm` function for an explanation.
    sample_size
        The number of successive states the previous values have been computed on;
        also the current number of iterations of the algorithm.

    """

    mean: Array
    m2: Array
    sample_size: int


class MassMatrixAdaptationState(NamedTuple):
    """State carried through the mass matrix adaptation.

    inverse_mass_matrix
        The curent value of the inverse mass matrix.
    wc_state
        The current state of the Welford Algorithm.

    """

    inverse_mass_matrix: Array
    wc_state: WelfordAlgorithmState


def mass_matrix_adaptation(
    is_diagonal_matrix: bool = True,
    imm_shrinkage_to_previous: float = 0.0,
) -> tuple[Callable, Callable, Callable]:
    """Adapts the values in the mass matrix by computing the covariance
    between parameters.

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adaps and returns a dense mass matrix.
    imm_shrinkage_to_previous
        Bayesian pseudo-count controlling shrinkage of the per-window adapted
        IMM toward the previous window's IMM. Interpretable as "the number of
        imaginary additional samples in the current window's accumulator that
        have already settled to ``IMM_prev``'s value". Combined with the
        existing Stan-pseudo-count 5 (which targets ``1e-3·I``) and the
        actual ``count`` samples in the window, the final IMM is the
        precision-weighted average:

        .. math::

            \\text{IMM}_\\text{new} =
            \\frac{\\text{count}}{\\text{denom}} \\cdot \\text{cov}_\\text{window} +
            \\frac{k_\\text{prev}}{\\text{denom}} \\cdot \\text{IMM}_\\text{prev} +
            \\frac{5}{\\text{denom}} \\cdot 10^{-3} \\cdot I

        where :math:`\\text{denom} = \\text{count} + 5 + k_\\text{prev}` and
        :math:`k_\\text{prev}` is this argument.

        - ``0.0`` (default): Stan-vanilla behavior, no shrinkage to previous.
        - ``5``: matches Stan's existing identity-shrinkage scale; mild,
          barely-perceptible persistence across windows.
        - ``≈ window_size / 4``: ~20% weight on the previous IMM; moderate
          persistence.
        - ``≈ window_size``: ~50% weight; previous IMM treated as equally
          informative as the new window's data.
        - ``>> window_size``: weight saturates near 100%; Welford effectively
          disabled (anti-pattern unless the prior IMM is *much* better than
          the chain can produce).

        Stan-default window sizes range 25 → 500 across Phase II, so the
        practical "moderate persistence" band is roughly
        ``5 ≤ k_prev ≤ 50``. Use larger values only when the prior IMM
        comes from a high-confidence source (e.g., a converged pre-warmup
        Pathfinder/multipathfinder fit on the right model). No upper bound
        is enforced — only ``k_prev >= 0.0`` is validated (raises
        ``ValueError`` on negative).

    Returns
    -------
    init
        A function that initializes the step of the mass matrix adaptation.
    update
        A function that updates the state of the mass matrix.
    final
        A function that computes the inverse mass matrix based on the current
        state.

    """
    if imm_shrinkage_to_previous < 0.0:
        raise ValueError(
            f"imm_shrinkage_to_previous must be >= 0.0, "
            f"got {imm_shrinkage_to_previous}"
        )

    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix)

    def init(
        n_dims: int,
        initial_inverse_mass_matrix: Array | None = None,
    ) -> MassMatrixAdaptationState:
        """Initialize the matrix adaptation.

        Parameters
        ----------
        n_dims
            The number of dimensions of the mass matrix, which corresponds to
            the number of dimensions of the chain position.
        initial_inverse_mass_matrix
            Optional seed value for the inverse mass matrix.  When ``None``
            (default) the standard identity initialisation is used: ``ones(d)``
            for diagonal matrices and ``identity(d)`` for dense matrices.
            When provided the array is used directly as the initial IMM; the
            Welford state is still started fresh so the seed is gradually
            overwritten by the empirical covariance as warmup windows proceed.
            Shape must match ``is_diagonal_matrix``: 1-D ``(d,)`` for diagonal,
            2-D ``(d, d)`` for dense.  Validation is the caller's
            responsibility (``window_adaptation`` checks before the JIT path).

        """
        if initial_inverse_mass_matrix is None:
            if is_diagonal_matrix:
                inverse_mass_matrix = jnp.ones(n_dims)
            else:
                inverse_mass_matrix = jnp.identity(n_dims)
        else:
            inverse_mass_matrix = jnp.asarray(initial_inverse_mass_matrix)

        wc_state = wc_init(n_dims)

        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    def update(
        mm_state: MassMatrixAdaptationState, position: ArrayLike
    ) -> MassMatrixAdaptationState:
        """Update the algorithm's state.

        Parameters
        ----------
        state:
            The current state of the mass matrix adapation.
        position:
            The current position of the chain.

        """
        inverse_mass_matrix, wc_state = mm_state
        position, _ = jax.flatten_util.ravel_pytree(position)
        wc_state = wc_update(wc_state, position)
        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    def final(mm_state: MassMatrixAdaptationState) -> MassMatrixAdaptationState:
        """Final iteration of the mass matrix adaptation.

        In this step we compute the mass matrix from the covariance matrix computed
        by the Welford algorithm, and re-initialize the later.

        The IMM is regularized as a convex combination of three terms:
        1. This window's empirical covariance (weight: count / denom)
        2. The previous window's IMM (weight: imm_shrinkage_to_previous / denom)
        3. A small identity matrix 1e-3·I (weight: 5 / denom)

        where denom = count + 5 + imm_shrinkage_to_previous.

        When imm_shrinkage_to_previous = 0.0 (default), this reduces to the
        standard Stan formula with only the first and third terms.

        """
        previous_imm, wc_state = mm_state
        covariance, count, mean = wc_final(wc_state)

        # Unified regularization formula with three shrinkage targets
        denom = count + 5 + imm_shrinkage_to_previous
        beta_data = count / denom
        beta_prev = imm_shrinkage_to_previous / denom
        beta_ident = 5 / denom

        if is_diagonal_matrix:
            inverse_mass_matrix = (
                beta_data * covariance + beta_prev * previous_imm + beta_ident * 1e-3
            )
        else:
            d = mean.shape[0]
            inverse_mass_matrix = (
                beta_data * covariance
                + beta_prev * previous_imm
                + beta_ident * 1e-3 * jnp.identity(d)
            )

        ndims = jnp.shape(inverse_mass_matrix)[-1]
        new_mm_state = MassMatrixAdaptationState(inverse_mass_matrix, wc_init(ndims))

        return new_mm_state

    return init, update, final


def welford_algorithm(is_diagonal_matrix: bool) -> tuple[Callable, Callable, Callable]:
    r"""Welford's online estimator of covariance.

    It is possible to compute the variance of a population of values in an
    on-line fashion to avoid storing intermediate results. The naive recurrence
    relations between the sample mean and variance at a step and the next are
    however not numerically stable.

    Welford's algorithm uses the sum of square of differences
    :math:`M_{2,n} = \sum_{i=1}^n \left(x_i-\overline{x_n}\right)^2`
    for updating where :math:`x_n` is the current mean and the following
    recurrence relationships

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adaps and returns a dense mass matrix.

    Note
    ----
    It might seem pedantic to separate the Welford algorithm from mass adaptation,
    but this covariance estimator is used in other parts of the library.

    """

    def init(n_dims: int) -> WelfordAlgorithmState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Parameters
        ----------
        n_dims: int
            The number of dimensions of the problem, which corresponds to the size
            of the corresponding square mass matrix.

        """
        sample_size = 0
        mean = jnp.zeros((n_dims,))
        if is_diagonal_matrix:
            m2 = jnp.zeros((n_dims,))
        else:
            m2 = jnp.zeros((n_dims, n_dims))
        return WelfordAlgorithmState(mean, m2, sample_size)

    def update(
        wa_state: WelfordAlgorithmState, value: ArrayLike
    ) -> WelfordAlgorithmState:
        """Update the M2 matrix using the new value.

        Parameters
        ----------
        wa_state:
            The current state of the Welford Algorithm
        value: Array, shape (1,)
            The new sample (typically position of the chain) used to update m2

        """
        mean, m2, sample_size = wa_state
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size
        updated_delta = value - mean
        if is_diagonal_matrix:
            new_m2 = m2 + delta * updated_delta
        else:
            new_m2 = m2 + jnp.outer(updated_delta, delta)

        return WelfordAlgorithmState(mean, new_m2, sample_size)

    def final(
        wa_state: WelfordAlgorithmState,
    ) -> tuple[Array, int, Array]:
        mean, m2, sample_size = wa_state
        covariance = m2 / (sample_size - 1)
        return covariance, sample_size, mean

    return init, update, final
