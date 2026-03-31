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
        Controls the damping floor in early iterations. The floor on γ is
        ``damping_slowdown / (t·ε)``, so higher values force stronger damping
        (higher α) in early iterations. Default is 1.0 as in the paper.

    Returns
    -------
    init
        Function that initializes the warmup state.
    update
        Function that moves the warmup one step forward.

    """
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")

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

        Notes
        -----
        This function uses the same chains' positions for both step-size and
        damping estimation. The full MEADS algorithm (Algorithm 3) uses
        cross-fold statistics: step size comes from the source fold, while
        damping uses the target fold's own positions. ``meads_adaptation``
        implements this correctly; this lower-level helper is an approximation
        suitable for direct use of ``base()``.
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

        # Algorithm 3, line 8: ε = min(1, step_size_multiplier / sqrt(λ_max(ḡ)))
        epsilon = jnp.minimum(
            step_size_multiplier / jnp.sqrt(maximum_eigenvalue(batch_grad_scaled)),
            1.0,
        )
        # Algorithm 3, line 9 (paper parameterization):
        #   γ = max(1/sqrt(λ_max(θ̄)), damping_slowdown / (t·ε))
        # With α = 1 - exp(-2·ε·γ) the floor gives
        #   α_floor = 1 - exp(-2·damping_slowdown/t)
        # Higher damping_slowdown → higher α_floor → stronger damping in early
        # iterations.
        gamma = jnp.maximum(
            1.0 / jnp.sqrt(maximum_eigenvalue(normalized_positions)),
            damping_slowdown / ((current_iteration + 1) * epsilon),
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
    fold ``t mod K`` is frozen (its chains do not advance, Algorithm 3 line 4).
    For each active fold k, the step size is computed from fold ``(k-1) mod K``'s
    preconditioned gradients, and the damping is computed from fold k's own
    positions using that step size. Every K steps all chains are reshuffled
    randomly across folds to prevent fold-assignment bias.

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
        as in the paper. Higher values force stronger damping in early iterations.
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
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    if num_chains % num_folds != 0:
        raise ValueError(
            f"num_chains ({num_chains}) must be divisible by num_folds ({num_folds})."
        )
    n_per_fold = num_chains // num_folds

    ghmc_kernel = mcmc.ghmc.build_kernel()
    adapt_init, _ = base(num_folds, step_size_multiplier, damping_slowdown)
    batch_init = jax.vmap(lambda p, r: mcmc.ghmc.init(p, r, logdensity_fn))

    def one_step(carry, rng_key):
        states, adaptation_state = carry
        t = adaptation_state.current_iteration

        # Fold to freeze this step (Algorithm 3, line 4: "excluding k = t mod K")
        fold_to_skip = t % num_folds

        keys = jax.random.split(rng_key, num_chains + 1)
        chain_keys, shuffle_key = keys[:num_chains], keys[num_chains]

        # Reshape chain arrays to [num_folds, n_per_fold, *dims] for per-fold ops
        def to_folds(x):
            return x.reshape((num_folds, n_per_fold) + x.shape[1:])

        folded_pos = jax.tree.map(to_folds, states.position)
        folded_grads = jax.tree.map(to_folds, states.logdensity_grad)

        # Per-fold scale (std across chains within each fold)
        # Result: PyTree with leaves [num_folds, *dims]
        folded_scales = jax.tree.map(lambda p: p.std(axis=1), folded_pos)

        # Preconditioned grads: grads_k * scale_k  (Algorithm 3, line 7)
        precond_grads = jax.tree.map(
            lambda g, s: g * jnp.expand_dims(s, axis=1),
            folded_grads,
            folded_scales,
        )

        # Per-fold step size from each fold's own preconditioned grads.
        # Then roll by 1 so fold k gets the step size from fold k-1.
        # (Algorithm 3, line 8 + cross-fold roll)
        def fold_step_size(grads_k):
            return jnp.minimum(
                step_size_multiplier / jnp.sqrt(maximum_eigenvalue(grads_k)),
                1.0,
            )

        step_size_own = jax.vmap(fold_step_size)(precond_grads)  # [num_folds]
        # fold k uses step_size from fold k-1
        step_size_rolled = jnp.roll(step_size_own, 1)  # [num_folds]
        # fold k uses the momentum scale (std) from fold k-1
        scales_rolled = jax.tree.map(  # [num_folds, *dims]
            lambda s: jnp.roll(s, 1, axis=0), folded_scales
        )

        # Per-fold damping from each fold's OWN (centered, scaled) positions
        # and the rolled step size from the left-neighbor fold.
        # Algorithm 3, lines 9-10 (paper parameterization):
        #   γ_k = max(1/sqrt(λ_max(θ̄_k)), damping_slowdown/(t·ε_k))
        #   α_k = 1 - exp(-2·ε_k·γ_k)
        def fold_damping(pos_k, eps_k):
            # Center within the fold before eigenvalue estimation
            pos_k_centered = jax.tree.map(lambda p: p - p.mean(axis=0), pos_k)
            gamma = jnp.maximum(
                1.0 / jnp.sqrt(maximum_eigenvalue(pos_k_centered)),
                damping_slowdown / ((t + 1) * eps_k),
            )
            alpha = 1.0 - jnp.exp(-2.0 * eps_k * gamma)
            delta = alpha / 2
            return alpha, delta

        # Divide each fold's positions by its own scale (centering done inside)
        precond_pos = jax.tree.map(
            lambda p, s: p / jnp.expand_dims(s, axis=1),
            folded_pos,
            folded_scales,
        )
        alphas, deltas = jax.vmap(fold_damping)(precond_pos, step_size_rolled)

        # Broadcast per-fold parameters to per-chain arrays
        chain_step_sizes = jnp.repeat(step_size_rolled, n_per_fold)
        chain_scales = jax.tree.map(
            lambda s: jnp.repeat(s, n_per_fold, axis=0), scales_rolled
        )
        chain_alphas = jnp.repeat(alphas, n_per_fold)
        chain_deltas = jnp.repeat(deltas, n_per_fold)

        # Step all chains with their fold's parameters
        new_states, info = jax.vmap(ghmc_kernel, in_axes=(0, 0, None, 0, 0, 0, 0))(
            chain_keys,
            states,
            logdensity_fn,
            chain_step_sizes,
            chain_scales,
            chain_alphas,
            chain_deltas,
        )

        # Restore fold_to_skip's chains: they do not advance this step
        # (Algorithm 3, line 4: "excluding k = t mod K").
        # When num_folds==1 there is no meaningful cross-fold split, so all
        # chains advance (no fold is frozen).
        if num_folds > 1:
            fold_is_skipped = jnp.arange(num_folds) == fold_to_skip  # [num_folds]
            chain_is_skipped = jnp.repeat(fold_is_skipped, n_per_fold)  # [num_chains]

            def restore_skipped(new_val, old_val):
                mask = chain_is_skipped.reshape(
                    chain_is_skipped.shape + (1,) * (new_val.ndim - 1)
                )
                return jnp.where(mask, old_val, new_val)

            new_states = jax.tree.map(restore_skipped, new_states, states)

        new_adaptation_state = MEADSAdaptationState(
            current_iteration=t + 1,
            step_size=step_size_rolled,
            position_sigma=scales_rolled,
            alpha=alphas,
            delta=deltas,
        )

        # Every num_folds steps: reshuffle chains across folds.
        # Skipped for num_folds==1 (single fold; shuffle would be a no-op).
        if num_folds > 1:
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
