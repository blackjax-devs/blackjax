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
from jax.flatten_util import ravel_pytree

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.mclmc_lrd_adaptation import _extract_lrd_from_samples
from blackjax.base import AdaptationAlgorithm
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
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


def _low_rank_apply(element: Array, U: Array, lam_pow: Array) -> Array:
    """Batched ``element + U @ ((lam_pow - 1) * (U.T @ element))``.

    ``element`` has shape ``(n, d)`` (a batch of ``n`` flat vectors), ``U``
    has shape ``(d, k)``, ``lam_pow`` has shape ``(k,)``. Shared building
    block for the two low-rank whitening transforms below -- it mirrors the
    ``B`` / ``A*`` matrices inside
    :func:`~blackjax.mcmc.metrics.gaussian_euclidean_low_rank`'s ``scale``
    closure, specialized to a batch of vectors so it composes with MEADS's
    per-fold ``jax.vmap``.
    """
    Ue = element @ U  # (n, k)
    return element + (Ue * (lam_pow - 1.0)) @ U.T


def _low_rank_precondition_grad(
    grad: Array, sigma: Array, U: Array, lam: Array
) -> Array:
    """Low-rank generalization of the legacy ``grad * sigma`` preconditioning.

    Mirrors ``M^{-1/2} grad`` (i.e. ``metric.scale(_, grad, inv=True,
    trans=False)`` for
    :func:`~blackjax.mcmc.metrics.gaussian_euclidean_low_rank`); reduces to
    ``grad * sigma`` when ``lam == 1`` (the diagonal limit), matching the
    legacy preconditioning bit-for-bit.
    """
    return _low_rank_apply(grad, U, jnp.sqrt(lam)) * sigma


def _low_rank_precondition_pos(pos: Array, sigma: Array, U: Array, lam: Array) -> Array:
    """Low-rank generalization of the legacy ``pos / sigma`` preconditioning.

    Mirrors ``M^{1/2} pos`` (i.e. ``metric.scale(_, pos, inv=False,
    trans=False)`` for
    :func:`~blackjax.mcmc.metrics.gaussian_euclidean_low_rank`); reduces to
    ``pos / sigma`` when ``lam == 1`` (the diagonal limit), matching the
    legacy preconditioning bit-for-bit.
    """
    return _low_rank_apply(pos, U, 1.0 / jnp.sqrt(lam)) / sigma


def meads_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    num_folds: int = 4,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
    adaptation_info_fn: Callable = return_all_adapt_info,
    low_rank_rank: int | None = None,
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
    low_rank_rank
        MEADS-LRD extension (opt-in, default ``None``). ``None`` adapts a
        *diagonal* momentum metric from the fold ensemble -- exactly the
        original behavior, bit-for-bit. An ``int`` instead adapts a
        rank-``low_rank_rank`` :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`
        from the fold ensemble (requires :func:`blackjax.mcmc.ghmc`'s
        dense/low-rank momentum-metric support, blackjax#950), generalizing
        MEADS the way MCLMC-LRD generalized MCLMC. Each fold only has
        ``num_chains // num_folds`` chains, so the *internal* per-fold
        estimate that drives the step-size/damping heuristics during warmup
        is clamped to rank ``num_chains // num_folds - 1`` (a fold's ensemble
        cannot span a higher rank than that -- raises ``ValueError`` if this
        would be < 1). The metric ultimately *returned* by ``run()`` is
        instead re-estimated once from the full final population of all
        ``num_chains`` chains (clamped to ``num_chains - 1``): a richer,
        less noisy estimate than any single fold's, and one that sidesteps
        averaging per-fold eigenbases together -- unlike a diagonal scale,
        per-fold eigenvectors have no canonical cross-fold alignment, so an
        elementwise average across folds would not be meaningful.

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

    low_rank_k: int | None = None
    if low_rank_rank is not None:
        if not hasattr(mcmc.ghmc, "_metric_from_momentum_inverse_scale"):
            raise RuntimeError(
                "low_rank_rank requires blackjax.mcmc.ghmc's dense/low-rank "
                "momentum-metric support (blackjax#950); the installed ghmc "
                "module predates it."
            )
        low_rank_k = min(low_rank_rank, n_per_fold - 1)
        if low_rank_k < 1:
            raise ValueError(
                f"low_rank_rank={low_rank_rank} cannot be honored: each fold "
                f"only has n_per_fold = num_chains // num_folds = {n_per_fold} "
                "chains, and the per-fold low-rank estimate needs "
                "n_per_fold - 1 >= 1. Increase num_chains or decrease "
                "num_folds."
            )

    ghmc_kernel = mcmc.ghmc.build_kernel()
    adapt_init, _ = base(num_folds, step_size_multiplier, damping_slowdown)
    batch_init = jax.vmap(lambda p, r: mcmc.ghmc.init(p, logdensity_fn, r))

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

        # MEADS-LRD: each fold additionally estimates its own rank-`low_rank_k`
        # correlation eigenbasis (sigma, U, lam) from its own n_per_fold
        # position snapshot, via the same sample-based extractor MCLMC-LRD
        # uses (`_extract_lrd_from_samples`). This low-rank metric -- not
        # just its diagonal sigma -- then preconditions the step-size and
        # damping heuristics below (via `_low_rank_precondition_{grad,pos}`),
        # so alpha/delta/step_size stay consistent with the metric actually
        # handed to ghmc for sampling this step.
        if low_rank_rank is not None:
            flat_folded_pos = jax.vmap(jax.vmap(lambda p: ravel_pytree(p)[0]))(
                folded_pos
            )
            flat_folded_grads = jax.vmap(jax.vmap(lambda g: ravel_pytree(g)[0]))(
                folded_grads
            )
            fold_sigma, fold_U, fold_lam, _ = jax.vmap(
                lambda flat: _extract_lrd_from_samples(flat, k=low_rank_k)
            )(flat_folded_pos)
            precond_grads_for_step_size = jax.vmap(_low_rank_precondition_grad)(
                flat_folded_grads, fold_sigma, fold_U, fold_lam
            )
        else:
            precond_grads_for_step_size = precond_grads

        # Per-fold step size from each fold's own preconditioned grads.
        # Then roll by 1 so fold k gets the step size from fold k-1.
        # (Algorithm 3, line 8 + cross-fold roll)
        def fold_step_size(grads_k):
            return jnp.minimum(
                step_size_multiplier / jnp.sqrt(maximum_eigenvalue(grads_k)),
                1.0,
            )

        step_size_own = jax.vmap(fold_step_size)(
            precond_grads_for_step_size
        )  # [num_folds]
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
        if low_rank_rank is not None:
            precond_pos_for_damping = jax.vmap(_low_rank_precondition_pos)(
                flat_folded_pos, fold_sigma, fold_U, fold_lam
            )
            # fold k's momentum metric is fold (k-1)'s low-rank estimate,
            # mirroring scales_rolled above.
            low_rank_sigma_rolled = jnp.roll(fold_sigma, 1, axis=0)
            low_rank_U_rolled = jnp.roll(fold_U, 1, axis=0)
            low_rank_lam_rolled = jnp.roll(fold_lam, 1, axis=0)
        else:
            precond_pos_for_damping = precond_pos
        alphas, deltas = jax.vmap(fold_damping)(
            precond_pos_for_damping, step_size_rolled
        )

        # Broadcast per-fold parameters to per-chain arrays
        chain_step_sizes = jnp.repeat(step_size_rolled, n_per_fold)
        chain_scales = jax.tree.map(
            lambda s: jnp.repeat(s, n_per_fold, axis=0), scales_rolled
        )
        chain_alphas = jnp.repeat(alphas, n_per_fold)
        chain_deltas = jnp.repeat(deltas, n_per_fold)

        if low_rank_rank is not None:
            # Feed ghmc a per-chain LowRankInverseMassMatrix instead of the
            # diagonal chain_scales; blackjax#950 lets ghmc's
            # momentum_inverse_scale accept this directly (no elementwise
            # squaring, unlike the legacy diagonal path).
            chain_momentum_inverse_scale = LowRankInverseMassMatrix(
                sigma=jnp.repeat(low_rank_sigma_rolled, n_per_fold, axis=0),
                U=jnp.repeat(low_rank_U_rolled, n_per_fold, axis=0),
                lam=jnp.repeat(low_rank_lam_rolled, n_per_fold, axis=0),
            )
        else:
            chain_momentum_inverse_scale = chain_scales

        # Step all chains with their fold's parameters
        new_states, info = jax.vmap(ghmc_kernel, in_axes=(0, 0, None, 0, 0, 0, 0))(
            chain_keys,
            states,
            logdensity_fn,
            chain_step_sizes,
            chain_momentum_inverse_scale,
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

        if low_rank_rank is not None:
            # Re-estimate the metric returned to the caller from the full
            # final population (num_chains chains), rather than averaging the
            # per-fold eigenbases used internally during warmup: unlike a
            # diagonal sigma, per-fold (U, lam) eigenpairs have no canonical
            # cross-fold alignment, so an elementwise average across folds
            # would not be meaningful. Using the full population also gives a
            # richer, less noisy estimate (num_chains samples instead of
            # n_per_fold), hence the looser clamp (num_chains - 1) than the
            # per-fold one (n_per_fold - 1) used above during warmup.
            flat_final_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(
                last_states.position
            )
            final_k = min(low_rank_rank, num_chains - 1)
            final_sigma, final_U, final_lam, _ = _extract_lrd_from_samples(
                flat_final_pos, k=final_k
            )
            momentum_inverse_scale = LowRankInverseMassMatrix(
                sigma=final_sigma, U=final_U, lam=final_lam
            )
        else:
            momentum_inverse_scale = jax.tree.map(
                lambda s: s.mean(axis=0), last_adaptation_state.position_sigma
            )

        # Return mean parameters across folds for use with a single ghmc kernel
        parameters = {
            "step_size": last_adaptation_state.step_size.mean(),
            "momentum_inverse_scale": momentum_inverse_scale,
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
