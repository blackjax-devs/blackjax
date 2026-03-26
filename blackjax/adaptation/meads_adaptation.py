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

    current_iteration
        Current iteration of the adaptation.
    step_size
        Step size for each fold, shape (num_folds,).
    position_sigma
        PyTree with per-fold per-dimension sample standard deviation of the
        position variable, leading axis has size num_folds.
    alpha
        Alpha parameter (momentum persistence) for each fold, shape (num_folds,).
    delta
        Delta parameter (slice translation) for each fold, shape (num_folds,).

    """

    current_iteration: int
    step_size: Array
    position_sigma: ArrayTree
    alpha: Array
    delta: Array


def base(
    num_folds: int = 4,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
):
    """Maximum-Eigenvalue Adaptation of damping and step size for the generalized
    Hamiltonian Monte Carlo kernel :cite:p:`hoffman2022tuning`.

    Full implementation of Algorithm 3 with K-fold cross-chain adaptation and
    chain shuffling. Chains are divided into ``num_folds`` folds; at each step
    statistics from fold ``t mod K`` are used to update the parameters for fold
    ``(t+1) mod K``. Every K steps all chains are reshuffled across folds.

    Parameters
    ----------
    num_folds
        Number of folds K to split chains into. Must divide num_chains evenly.
    step_size_multiplier
        Multiplicative factor applied to the raw step size heuristic (default 0.5
        as in the paper).
    damping_slowdown
        Multiplicative factor that slows the damping adaptation relative to the
        iteration count (default 1.0 as in the paper).

    Returns
    -------
    init
        Function that initializes the warmup state.
    update
        Function that moves the warmup one step forward.

    """

    def compute_parameters(
        positions: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        current_iteration: int,
    ):
        """Compute GHMC parameters from a single fold's chains.

        Parameters
        ----------
        positions
            PyTree with leading axis of size n_per_fold.
        logdensity_grad
            PyTree with leading axis of size n_per_fold.
        current_iteration
            Global adaptation iteration index.

        Returns
        -------
        step_size, position_sigma, alpha, delta
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

        epsilon = step_size_multiplier * jnp.minimum(
            0.5 / jnp.sqrt(maximum_eigenvalue(batch_grad_scaled)), 1.0
        )
        gamma = jnp.maximum(
            1.0 / jnp.sqrt(maximum_eigenvalue(normalized_positions)),
            1.0 / (damping_slowdown * (current_iteration + 1) * epsilon),
        )
        alpha = 1.0 - jnp.exp(-2.0 * epsilon * gamma)
        delta = alpha / 2
        return epsilon, sd_position, alpha, delta

    def init(
        positions: ArrayLikeTree, logdensity_grad: ArrayLikeTree
    ) -> MEADSAdaptationState:
        """Initialize with parameters computed from all chains, replicated per fold."""
        step_size, sd_position, alpha, delta = compute_parameters(
            positions, logdensity_grad, 0
        )
        # Replicate scalar params across folds
        step_sizes = jnp.full((num_folds,), step_size)
        alphas = jnp.full((num_folds,), alpha)
        deltas = jnp.full((num_folds,), delta)
        # Replicate pytree params: (num_folds, *dims) per leaf
        sigmas = jax.tree.map(
            lambda s: jnp.repeat(s[None], num_folds, axis=0), sd_position
        )
        return MEADSAdaptationState(0, step_sizes, sigmas, alphas, deltas)

    def update(
        adaptation_state: MEADSAdaptationState,
        positions: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        source_fold: int,
    ) -> MEADSAdaptationState:
        """Update the target fold's parameters using the source fold's statistics.

        Parameters
        ----------
        adaptation_state
            Current adaptation state.
        positions
            Positions of chains in the source fold only.
        logdensity_grad
            Gradients of chains in the source fold only.
        source_fold
            Index of the fold whose statistics are used. The target fold that
            receives the updated parameters is ``(source_fold + 1) % num_folds``.

        Returns
        -------
        Updated adaptation state.
        """
        target = (source_fold + 1) % num_folds
        t = adaptation_state.current_iteration

        new_step_size, new_sigma, new_alpha, new_delta = compute_parameters(
            positions, logdensity_grad, t
        )

        new_step_sizes = adaptation_state.step_size.at[target].set(new_step_size)
        new_sigmas = jax.tree.map(
            lambda s, v: s.at[target].set(v),
            adaptation_state.position_sigma,
            new_sigma,
        )
        new_alphas = adaptation_state.alpha.at[target].set(new_alpha)
        new_deltas = adaptation_state.delta.at[target].set(new_delta)

        return MEADSAdaptationState(
            t + 1, new_step_sizes, new_sigmas, new_alphas, new_deltas
        )

    return init, update


def meads_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    num_folds: int = 4,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
    adaptation_info_fn: Callable = return_all_adapt_info,
) -> AdaptationAlgorithm:
    """Adapt the parameters of the Generalized HMC algorithm.

    Full implementation of Algorithm 3 from :cite:p:`hoffman2022tuning` with
    K-fold cross-chain adaptation and periodic chain shuffling.

    Chains are divided into ``num_folds`` folds. At adaptation step ``t``,
    cross-chain statistics are computed within fold ``t mod K`` and used to
    update the GHMC parameters for fold ``(t+1) mod K``. Every K steps all
    chains are reshuffled randomly across folds to prevent fold-assignment bias.
    Each fold independently tracks its own step size, momentum scale, alpha and
    delta parameters.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Total number of chains. Must be divisible by ``num_folds``.
    num_folds
        Number of folds K to split chains into. Default is 4 as in the paper.
    step_size_multiplier
        Multiplicative factor for the step size heuristic. Default is 0.5 as in
        the paper.
    damping_slowdown
        Slows the damping decay relative to the iteration count. Default is 1.0
        as in the paper.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base. By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.

    Returns
    -------
    A function that returns the last cross-chain state, a sampling kernel with the
    tuned parameter values (averaged across folds), and all the warm-up states for
    diagnostics.

    """
    if num_chains % num_folds != 0:
        raise ValueError(
            f"num_chains ({num_chains}) must be divisible by num_folds ({num_folds})."
        )
    n_per_fold = num_chains // num_folds

    ghmc_kernel = mcmc.ghmc.build_kernel()
    adapt_init, adapt_update = base(num_folds, step_size_multiplier, damping_slowdown)
    batch_init = jax.vmap(lambda p, r: mcmc.ghmc.init(p, r, logdensity_fn))

    def one_step(carry, rng_key):
        states, adaptation_state = carry
        t = adaptation_state.current_iteration

        # Identify source fold; target fold receives updated parameters
        source = t % num_folds

        # Extract source fold chains
        source_positions = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, source * n_per_fold, n_per_fold, axis=0
            ),
            states.position,
        )
        source_grads = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, source * n_per_fold, n_per_fold, axis=0
            ),
            states.logdensity_grad,
        )

        # Update target fold parameters from source fold statistics
        new_adaptation_state = adapt_update(
            adaptation_state, source_positions, source_grads, source
        )

        # Broadcast per-fold parameters to per-chain arrays
        chain_step_sizes = jnp.repeat(new_adaptation_state.step_size, n_per_fold)
        chain_sigmas = jax.tree.map(
            lambda s: jnp.repeat(s, n_per_fold, axis=0),
            new_adaptation_state.position_sigma,
        )
        chain_alphas = jnp.repeat(new_adaptation_state.alpha, n_per_fold)
        chain_deltas = jnp.repeat(new_adaptation_state.delta, n_per_fold)

        # Step all chains with their fold's parameters
        keys = jax.random.split(rng_key, num_chains + 1)
        chain_keys, shuffle_key = keys[:num_chains], keys[num_chains]

        new_states, info = jax.vmap(ghmc_kernel, in_axes=(0, 0, None, 0, 0, 0, 0))(
            chain_keys,
            states,
            logdensity_fn,
            chain_step_sizes,
            chain_sigmas,
            chain_alphas,
            chain_deltas,
        )

        # Every num_folds steps: reshuffle chains across folds
        perm = jax.random.permutation(shuffle_key, num_chains)
        new_states = jax.lax.cond(
            (t + 1) % num_folds == 0,
            lambda s: jax.tree.map(lambda x: x[perm], s),
            lambda s: s,
            new_states,
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

        # Return mean parameters across folds for use with a single ghmc kernel
        parameters = {
            "step_size": last_adaptation_state.step_size.mean(),
            "momentum_inverse_scale": jax.tree.map(
                lambda s: s.mean(axis=0), last_adaptation_state.position_sigma
            ),
            "alpha": last_adaptation_state.alpha.mean(),
            "delta": last_adaptation_state.delta.mean(),
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
