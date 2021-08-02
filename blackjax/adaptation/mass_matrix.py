"""Algorithms to adapt the mass matrix used by algorithms in the Hamiltonian
Monte Carlo family to the current geometry.

The Stan Manual [1]_ is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

.. [1]: "HMC Algorithm Parameters", Stan Manual
        https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = ["mass_matrix_adaptation", "welford_algorithm"]


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
) -> Tuple[Callable, Callable, Callable]:
    """Adapts the values in the mass matrix by computing the covariance
    between parameters.

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adaps and returns a dense mass matrix.

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
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix)

    def init(n_dims: int) -> MassMatrixAdaptationState:
        """Initialize the matrix adaptation.

        Parameters
        ----------
        ndims
            The number of dimensions of the mass matrix, which corresponds to
            the number of dimensions of the chain position.
        """
        if is_diagonal_matrix:
            inverse_mass_matrix = jnp.ones(n_dims)
        else:
            inverse_mass_matrix = jnp.identity(n_dims)

        wc_state = wc_init(n_dims)

        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    def update(
        mm_state: MassMatrixAdaptationState, position: Array
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
        """
        _, wc_state = mm_state
        covariance, count, mean = wc_final(wc_state)

        # Regularize the covariance matrix, see Stan
        scaled_covariance = (count / (count + 5)) * covariance
        shrinkage = 1e-3 * (5 / (count + 5))
        if is_diagonal_matrix:
            inverse_mass_matrix = scaled_covariance + shrinkage
        else:
            inverse_mass_matrix = scaled_covariance + shrinkage * jnp.identity(
                mean.shape[0]
            )

        ndims = jnp.shape(inverse_mass_matrix)[-1]
        new_mm_state = MassMatrixAdaptationState(inverse_mass_matrix, wc_init(ndims))

        return new_mm_state

    return init, update, final


def welford_algorithm(is_diagonal_matrix: bool) -> Tuple[Callable, Callable, Callable]:
    """Welford's online estimator of covariance.

    It is possible to compute the variance of a population of values in an
    on-line fashion to avoid storing intermediate results. The naive recurrence
    relations between the sample mean and variance at a step and the next are
    however not numerically stable.

    Welford's algorithm uses the sum of square of differences
    :math:`M_{2,n} = \\sum_{i=1}^n \\left(x_i-\\overline{x_n}\right)^2`
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

    .. math:
        M_{2,n} = M_{2, n-1} + (x_n-\\overline{x}_{n-1})(x_n-\\overline{x}_n)
        \\sigma_n^2 = \\frac{M_{2,n}}{n}
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

    def update(wa_state: WelfordAlgorithmState, value: Array) -> WelfordAlgorithmState:
        """Update the M2 matrix using the new value.

        Parameters
        ----------
        state:
            The current state of the Welford Algorithm
        position: Array, shape (1,)
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
    ) -> Tuple[Array, int, Array]:
        mean, m2, sample_size = wa_state
        covariance = m2 / (sample_size - 1)
        return covariance, sample_size, mean

    return init, update, final
