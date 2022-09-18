from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


class MEADSAdaptationState(NamedTuple):
    current_iteration: int
    step_sizes: PyTree
    alpha: float
    delta: float


def base(
    logprob_grad_fn: Callable,
    batch_fn: Callable = jax.vmap,
):
    """Maximum-Eigenvalue Adaptation of damping and dtep size for the generalized
    Hamiltonian Monte Carlo kernel [1]_.


    This algorithm performs a cross-chain adaptation scheme for the generalized
    HMC algorithm that automatically selects values for the generalized HMC's
    tunable parameters based on statistics collected from a population of many
    chains. It uses heuristics determined by the maximum eigenvalue of the
    covariance and gradient matrices given by the grouped samples of all chains
    with shape.

    This is an implementation of Algorithm 3 of [1]_ using cross-chain
    adaptation instead of parallel ensample chain adaptation.

    Parameters
    ----------
    logprob_grad_fn
        The gradient of logprob_fn, outputs the gradient PyTree for sample.
    batch_fn
        Either jax.vmap or jax.pmap to perform parallel operations.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns step size, alpha and delta given a cross-chain
        warmup state.

    References
    ----------
    .. [1]: Hoffman, M. D., & Sountsov, P. (2022). Tuning-Free Generalized
            Hamiltonian Monte Carlo. In International Conference on Artificial
            Intelligence and Statistics (pp. 7799-7813). PML
    """

    def maximum_eigenvalue(matrix: PyTree):
        """Estimate the largest eigenvalues of a matrix.

        We calculate an unbiased estimate of the ratio between the sum of the
        squared eigenvalues and the sum of the eigenvalues from the input
        matrix. This ratio approximates the largest eigenvalue well except in
        cases when there are a large number of small eigenvalues significantly
        larger than 0 but significantly smaller than the largest eigenvalue.
        This unbiased estimate is used instead of directly computing an unbiased
        estimate of the largest eigenvalue because of the latter's large
        variance.

        """
        X = jnp.vstack(
            [
                leaf.reshape(leaf.shape[0], -1).T
                for leaf in jax.tree_util.tree_leaves(matrix)
            ]
        ).T
        n, _ = X.shape
        S = X @ X.T
        diag_S = jnp.diag(S)
        lamda = jnp.sum(diag_S) / n
        lamda_sq = (jnp.sum(S**2) - jnp.sum(diag_S**2)) / (n * (n - 1))
        return lamda_sq / lamda

    def parameter_gn(positions: PyTree, current_iter: int):
        """Update the adaptation state and parameter values.

        We find new optimal values for the parameters of the generalized HMC
        kernel using heuristics based on the maximum eigenvalue of the
        covariance and gradient matrices given by an ensemble of chains.

        Parameters
        ----------
        batch_state
            The current state of every chain.
        current_iter
            The current iteration number.

        Returns
        -------
        New values for the step size, alpha and delta parameters of the
        generalized HMC kernel.

        """
        mean_position = jax.tree_map(lambda p: p.mean(axis=0), positions)
        sd_position = jax.tree_map(lambda p: p.std(axis=0), positions)
        batch_norm = jax.tree_map(
            lambda p, mu, sd: (p - mu) / sd,
            positions,
            mean_position,
            sd_position,
        )
        batch_grad = batch_fn(logprob_grad_fn)(positions)
        batch_grad_scaled = jax.tree_map(
            lambda grad, sd: grad * sd, batch_grad, sd_position
        )
        epsilon = jnp.minimum(
            0.5 / jnp.sqrt(maximum_eigenvalue(batch_grad_scaled)), 1.0
        )
        gamma = jnp.maximum(
            1.0 / jnp.sqrt(maximum_eigenvalue(batch_norm)),
            1.0 / ((current_iter + 1) * epsilon),
        )
        alpha = 1.0 - jnp.exp(-2.0 * epsilon * gamma)
        delta = alpha / 2
        step_size = jax.tree_map(lambda sd: epsilon * sd, sd_position)
        return step_size, alpha, delta

    def init(positions: PyTree):
        parameters = parameter_gn(positions, 0)
        return MEADSAdaptationState(0, *parameters)

    def update(adaptation_state: MEADSAdaptationState, positions: PyTree):
        parameters = parameter_gn(positions, adaptation_state.current_iteration)
        return MEADSAdaptationState(adaptation_state.current_iteration + 1, *parameters)

    def final(adaptation_state: MEADSAdaptationState, positions: PyTree) -> PyTree:
        """Return the final values for the step size, alpha and delta."""
        parameters = parameter_gn(
            positions,
            adaptation_state.current_iteration,
        )
        return parameters

    return init, update, final
