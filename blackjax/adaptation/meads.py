"""Public API for Maximum-Eigenvalue Adaptation of Damping and Step-size kernel"""

from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.types import PyTree


def base(
    kernel_factory: Callable,
    logprob_grad_fn: Callable,
    batch_fn: Callable = jax.vmap,
):
    """Maximum-Eigenvalue Adaptation of Damping and Step size for the Generalized
    Hamiltonian Monte Carlo kernel [1]_.

    Performs a cross-chain adaptation scheme for the Generalized HMC algorithm that
    automatically selects values for the Generalized HMC's tunable parameters
    based on statistics collected from a population of many chains. It uses heuristics
    determined by the maximum eigenvalue of the covariance and gradient matrices given
    by the grouped samples of all chains with shape (num_chain, parameter dimension).
    Specifically, it does Algorithm 3 of [1]_ with cross-chain adaptation instead of
    parallel ensample chain adaptation.

    Parameters
    ----------
    kernel_factory
        Function that takes as input the step size, alpha and delta parameters
        and outputs a Generalized HMC kernel that generates new samples.
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
        """Estimate largest eigenvalues of a matrix.

        Calculates an unbiased estimate of the ratio between the sum of the squared
        eigenvalues and the sum of the eigenvalues from the input matrix. This ratio
        approximates the largest eigenvalue well except in cases when there are a
        large number of small eigenvalues significantly larger than 0 but significantly
        smaller than the largest eigenvalue. This unbiased estimate is used instead of
        directly computing an unbiased estimate of the largest eigenvalue because of
        the latter's large variance.
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

    def parameter_gn(batch_state, current_iter):
        """Generate updated parameteters using information of cross-chain samples.

        Using heuristics from the maximum eigenvalue of the covariance and gradient
        matrices given by the grouped samples of all chains, find new optimal
        parameters for the Generalized HMC kernel.
        """

        batch_position = batch_state.position
        mean_position = jax.tree_map(lambda p: p.mean(axis=0), batch_position)
        sd_position = jax.tree_map(lambda p: p.std(axis=0), batch_position)
        batch_norm = jax.tree_map(
            lambda p, mu, sd: (p - mu) / sd,
            batch_position,
            mean_position,
            sd_position,
        )
        batch_grad = batch_fn(logprob_grad_fn)(batch_position)
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

    init, update = chain_adaptation.cross_chain(
        lambda *parameters: batch_fn(kernel_factory(*parameters)),
        parameter_gn,
    )

    def final(last_state: chain_adaptation.ChainState) -> PyTree:
        parameters = parameter_gn(
            last_state.states,
            last_state.current_iter,
        )
        return kernel_factory(*parameters)

    return init, update, final
