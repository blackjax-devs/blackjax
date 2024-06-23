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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["MEADSAdaptationState", "base", "maximum_eigenvalue", "meads_adaptation"]


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
        Value of the delta parameter of the generalized HMC algorithm.

    """

    current_iteration: int
    step_size: float
    position_sigma: ArrayTree
    alpha: float
    delta: float


def base():
    """Maximum-Eigenvalue Adaptation of damping and step size for the generalized
    Hamiltonian Monte Carlo kernel :cite:p:`hoffman2022tuning`.


    This algorithm performs a cross-chain adaptation scheme for the generalized
    HMC algorithm that automatically selects values for the generalized HMC's
    tunable parameters based on statistics collected from a population of many
    chains. It uses heuristics determined by the maximum eigenvalue of the
    covariance and gradient matrices given by the grouped samples of all chains
    with shape.

    This is an implementation of Algorithm 3 of :cite:p:`hoffman2022tuning` using cross-chain
    adaptation instead of parallel ensemble chain adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.

    """

    def compute_parameters(
        positions: ArrayLikeTree, logdensity_grad: ArrayLikeTree, current_iteration: int
    ):
        """Compute values for the parameters based on statistics collected from
        multiple chains.

        Parameters
        ----------
        positions:
            A PyTree that contains the current position of every chains.
        logdensity_grad:
            A PyTree that contains the gradients of the logdensity
            function evaluated at the current position of every chains.
        current_iteration:
            The current iteration index in the adaptation process.

        Returns
        -------
        New values of the step size, and the alpha and delta parameters
        of the generalized HMC algorithm.

        """
        mean_position = jax.tree.map(lambda p: p.mean(axis=0), positions)
        sd_position = jax.tree.map(lambda p: p.std(axis=0), positions)
        normalized_positions = jax.tree.map(
            lambda p, mu, sd: (p - mu) / sd,
            positions,
            mean_position,
            sd_position,
        )

        batch_grad_scaled = jax.tree.map(
            lambda grad, sd: grad * sd, logdensity_grad, sd_position
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

    def init(
        positions: ArrayLikeTree, logdensity_grad: ArrayLikeTree
    ) -> MEADSAdaptationState:
        parameters = compute_parameters(positions, logdensity_grad, 0)
        return MEADSAdaptationState(0, *parameters)

    def update(
        adaptation_state: MEADSAdaptationState,
        positions: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
    ) -> MEADSAdaptationState:
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
        logdensity_grad
            The gradients of the logdensity function evaluated at the
            current position of every chain.

        Returns
        -------
        New adaptation state that contains the step size, alpha and delta
        parameters of the generalized HMC kernel.

        """
        current_iteration = adaptation_state.current_iteration
        step_size, position_sigma, alpha, delta = compute_parameters(
            positions, logdensity_grad, current_iteration
        )

        return MEADSAdaptationState(
            current_iteration + 1, step_size, position_sigma, alpha, delta
        )

    return init, update


def meads_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    adaptation_info_fn: Callable = return_all_adapt_info,
) -> AdaptationAlgorithm:
    """Adapt the parameters of the Generalized HMC algorithm.

    The Generalized HMC algorithm depends on three parameters, each controlling
    one element of its behaviour: step size controls the integrator's dynamics,
    alpha controls the persistency of the momentum variable, and delta controls
    the deterministic transformation of the slice variable used to perform the
    non-reversible Metropolis-Hastings accept/reject step.

    The step size parameter is chosen to ensure the stability of the velocity
    verlet integrator, the alpha parameter to make the influence of the current
    state on future states of the momentum variable to decay exponentially, and
    the delta parameter to maximize the acceptance of proposal but with good
    mixing properties for the slice variable. These characteristics are targeted
    by controlling heuristics based on the maximum eigenvalues of the correlation
    and gradient matrices of the cross-chain samples, under simpifyng assumptions.

    Good tuning is fundamental for the non-reversible Generalized HMC sampling
    algorithm to explore the target space efficienty and output uncorrelated, or
    as uncorrelated as possible, samples from the target space. Furthermore, the
    single integrator step of the algorithm lends itself for fast sampling
    on parallel computer architectures.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of chains used for cross-chain warm-up training.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.

    Returns
    -------
    A function that returns the last cross-chain state, a sampling kernel with the
    tuned parameter values, and all the warm-up states for diagnostics.

    """

    ghmc_kernel = mcmc.ghmc.build_kernel()

    adapt_init, adapt_update = base()

    batch_init = jax.vmap(lambda p, r: mcmc.ghmc.init(p, r, logdensity_fn))

    def one_step(carry, rng_key):
        states, adaptation_state = carry

        keys = jax.random.split(rng_key, num_chains)
        new_states, info = jax.vmap(
            ghmc_kernel, in_axes=(0, 0, None, None, None, None, None)
        )(
            keys,
            states,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.position_sigma,
            adaptation_state.alpha,
            adaptation_state.delta,
        )
        new_adaptation_state = adapt_update(
            adaptation_state, new_states.position, new_states.logdensity_grad
        )

        return (new_states, new_adaptation_state), adaptation_info_fn(
            new_states, info, new_adaptation_state
        )

    def run(rng_key: PRNGKey, positions: ArrayLikeTree, num_steps: int = 1000):
        key_init, key_adapt = jax.random.split(rng_key)

        rng_keys = jax.random.split(key_init, num_chains)
        init_states = batch_init(positions, rng_keys)
        init_adaptation_state = adapt_init(positions, init_states.logdensity_grad)

        keys = jax.random.split(key_adapt, num_steps)
        (last_states, last_adaptation_state), info = jax.lax.scan(
            one_step, (init_states, init_adaptation_state), keys
        )

        parameters = {
            "step_size": last_adaptation_state.step_size,
            "momentum_inverse_scale": last_adaptation_state.position_sigma,
            "alpha": last_adaptation_state.alpha,
            "delta": last_adaptation_state.delta,
        }

        return AdaptationResults(last_states, parameters), info

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]


def maximum_eigenvalue(matrix: ArrayLikeTree) -> Array:
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
