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
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


class MEADSAdaptationState(NamedTuple):
    """State of the MEADS adaptation scheme.

    step_size
        Value of the step_size parameter of the generalized HMC algorithm.
    position_sigma
        PyTree containing the per dimension sample standard deviation of the
        position variable. Used to scale the momentum variable on the generalized
        HMC algorithm.
    alpha
        Value of the alpha parameter of the generalized HMC algorithm.
    delta
        Value of the alpha parameter of the generalized HMC algorithm.

    """

    current_iteration: int
    step_size: float
    position_sigma: PyTree
    alpha: float
    delta: float


def base():
    """Maximum-Eigenvalue Adaptation of damping and step size for the generalized
    Hamiltonian Monte Carlo kernel [1]_.


    This algorithm performs a cross-chain adaptation scheme for the generalized
    HMC algorithm that automatically selects values for the generalized HMC's
    tunable parameters based on statistics collected from a population of many
    chains. It uses heuristics determined by the maximum eigenvalue of the
    covariance and gradient matrices given by the grouped samples of all chains
    with shape.

    This is an implementation of Algorithm 3 of [1]_ using cross-chain
    adaptation instead of parallel ensample chain adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.

    References
    ----------
    .. [1]: Hoffman, M. D., & Sountsov, P. (2022). Tuning-Free Generalized
            Hamiltonian Monte Carlo. In International Conference on Artificial
            Intelligence and Statistics (pp. 7799-7813). PML

    """

    def compute_parameters(
        positions: PyTree, potential_energy_grad: PyTree, current_iteration: int
    ):
        """Compute values for the parameters based on statistics collected from
        multiple chains.

        Parameters
        ----------
        positions:
            A PyTree that contains the current position of every chains.
        potential_energy_grad:
            A PyTree that contains the gradients of the potential energy
            function evaluated at the current position of every chains.
        current_iteration:
            The current iteration index in the adaptation process.

        Returns
        -------
        New values of the step size, and the alpha and delta parameters
        of the generalized HMC algorithm.

        """
        mean_position = jax.tree_map(lambda p: p.mean(axis=0), positions)
        sd_position = jax.tree_map(lambda p: p.std(axis=0), positions)
        normalized_positions = jax.tree_map(
            lambda p, mu, sd: (p - mu) / sd,
            positions,
            mean_position,
            sd_position,
        )

        batch_grad = jax.tree_map(lambda x: -x, potential_energy_grad)
        batch_grad_scaled = jax.tree_map(
            lambda grad, sd: grad * sd, batch_grad, sd_position
        )

        epsilon = jnp.minimum(
            0.5 / jnp.sqrt(maximum_eigenvalue(batch_grad_scaled)), 1.0
        )
        gamma = jnp.maximum(
            1.0 / jnp.sqrt(maximum_eigenvalue(normalized_positions)),
            1.0 / ((current_iteration + 1) * epsilon),
        )
        alpha = 1.0 - jnp.exp(-2.0 * epsilon * gamma)
        delta = alpha / 2
        return epsilon, sd_position, alpha, delta

    def init(positions: PyTree, potential_energy_grad: PyTree):
        parameters = compute_parameters(positions, potential_energy_grad, 0)
        return MEADSAdaptationState(0, *parameters)

    def update(
        adaptation_state: MEADSAdaptationState,
        positions: PyTree,
        potential_energy_grad: PyTree,
    ):
        """Update the adaptation state and parameter values.

        We find new optimal values for the parameters of the generalized HMC
        kernel using heuristics based on the maximum eigenvalue of the
        covariance and gradient matrices given by an ensemble of chains.

        Parameters
        ----------
        adaptation_state
            The current state of the adaptation algorithm
        positions
            The current position of every chain.
        potential_energy_grad
            The gradients of the potential energy function
            evaluated at the current position of every chain.

        Returns
        -------
        New adaptation state that contains the step size, alpha and delta
        parameters of the generalized HMC kernel.

        """
        current_iteration = adaptation_state.current_iteration
        step_size, position_sigma, alpha, delta = compute_parameters(
            positions, potential_energy_grad, current_iteration
        )

        return MEADSAdaptationState(
            current_iteration + 1, step_size, position_sigma, alpha, delta
        )

    return init, update


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

    Parameters
    ----------
    matrix
        A PyTree with equal batch shape as the first dimension of every leaf.
        The PyTree for each batch is flattened into a one dimensional array and
        these arrays are stacked vertically, giving a matrix with one row
        for every batch.

    """
    X = jax.vmap(lambda m: jax.flatten_util.ravel_pytree(m)[0])(matrix)
    n, _ = X.shape
    S = X @ X.T
    diag_S = jnp.diag(S)
    lamda = jnp.sum(diag_S) / n
    lamda_sq = (jnp.sum(S**2) - jnp.sum(diag_S**2)) / (n * (n - 1))
    return lamda_sq / lamda
