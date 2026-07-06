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


def _low_rank_precondition_pos(pos: Array, sigma: Array, U: Array, lam: Array) -> Array:
    """Low-rank generalization of the legacy ``pos / sigma`` preconditioning.

    Mirrors ``M^{1/2} pos`` (i.e. ``metric.scale(_, pos, inv=False,
    trans=False)`` for
    :func:`~blackjax.mcmc.metrics.gaussian_euclidean_low_rank`); reduces to
    ``pos / sigma`` when ``lam == 1`` (the diagonal limit), matching the
    legacy preconditioning bit-for-bit.
    """
    return _low_rank_apply(pos, U, 1.0 / jnp.sqrt(lam)) / sigma


class _LowRankAccumulatorState(NamedTuple):
    """Running Chan/Welford covariance accumulator for MEADS-LRD's window
    accumulation (high-d fix: see ``low_rank_rank``'s docstring).

    Carries the mean, the Chan "M2" sum-of-outer-products, and the effective
    sample count pooled across *all* ``num_chains`` chains and every
    accumulated window step, so the low-rank metric can eventually be
    estimated from effective ``n = num_chains * window_steps`` rather than a
    single ``num_chains``-sized ensemble snapshot.
    """

    mean: Array
    m2: Array
    count: Array


def _lrd_accumulator_init(d: int) -> _LowRankAccumulatorState:
    return _LowRankAccumulatorState(
        mean=jnp.zeros((d,)), m2=jnp.zeros((d, d)), count=jnp.zeros(())
    )


def _lrd_accumulator_update(
    acc: _LowRankAccumulatorState, batch: Array
) -> _LowRankAccumulatorState:
    """Merge a batch of ``n_b`` samples, shape ``(n_b, d)``, into the running
    accumulator via Chan et al.'s parallel/batch generalization of Welford's
    algorithm -- the same recurrence
    :func:`~blackjax.adaptation.mass_matrix.welford_algorithm` applies one
    sample at a time, applied here to a whole ensemble at once.
    """
    mean_a, m2_a, n_a = acc
    n_b = batch.shape[0]
    mean_b = jnp.mean(batch, axis=0)
    centered_b = batch - mean_b[None, :]
    m2_b = centered_b.T @ centered_b

    delta = mean_b - mean_a
    n_ab = n_a + n_b
    mean_ab = mean_a + delta * (n_b / n_ab)
    m2_ab = m2_a + m2_b + jnp.outer(delta, delta) * (n_a * n_b / n_ab)
    return _LowRankAccumulatorState(mean=mean_ab, m2=m2_ab, count=n_ab)


def _lrd_from_accumulated_covariance(
    acc: _LowRankAccumulatorState, k: int
) -> tuple[Array, Array, Array]:
    """Extract ``(sigma, U, lam)`` from a window-accumulated covariance
    (effective ``n = num_chains * window_steps``) via ``eigh`` of the
    accumulated correlation matrix, selecting the top-``k`` directions by
    ``|lam - 1|`` (the directions that deviate most from isotropic). This is
    what makes the low-rank metric estimable at high dimension (``d >>
    num_chains``): a single ``num_chains``-sized ensemble snapshot is
    ``p >> n`` noise-dominated once ``d`` exceeds ``num_chains``, but the
    window-accumulated covariance's effective ``n`` can comfortably exceed
    ``d`` given enough window steps.
    """
    mean, m2, count = acc
    del mean  # only the second moment is needed below
    covariance = m2 / jnp.maximum(count - 1.0, 1.0)
    variance = jnp.diag(covariance)
    sigma = jnp.sqrt(jnp.maximum(variance, 0.0))
    sigma = jnp.where(sigma <= 0.0, 1.0, sigma)  # avoid div-by-zero

    inv_sigma = 1.0 / sigma
    correlation = covariance * inv_sigma[:, None] * inv_sigma[None, :]

    # eigh gives ascending eigenvalues of the (symmetric) correlation matrix.
    lam_all, V = jnp.linalg.eigh(correlation)
    sort_idx = jnp.argsort(jnp.abs(lam_all - 1.0))[::-1]
    top_idx = sort_idx[:k]
    lam = lam_all[top_idx]
    U = V[:, top_idx]
    return sigma, U, lam


def _lrd_diagonal_fallback(flat_positions: Array, k: int) -> tuple[Array, Array, Array]:
    """Diagonal-only fallback ``(sigma, U, lam)`` for MEADS-LRD's momentum
    metric, used before the accumulation window holds enough pooled samples
    to support a rank-``k`` estimate (FIX 1).

    Estimating an eigenbasis from a single ``num_chains``-sized ensemble
    snapshot is exactly the noise-dominated (``p >> n``) estimator that
    causes the high-d step-size/momentum instability these fixes address --
    measured directly: routing through it even as a *pre-window* fallback
    (rather than attempting a low-rank correction from too little data) was
    enough to blow up the ensemble and collapse epsilon at ``d=40,
    num_chains=32``. So instead this returns ``lam = 1`` (no correction --
    ``_low_rank_apply``'s ``(lam_pow - 1)`` term vanishes identically),
    degenerating the momentum metric to exactly the diagonal-only
    preconditioning ``low_rank_rank=None`` uses. ``U``'s columns are then
    irrelevant (they multiply a zero coefficient) -- any orthonormal set
    works; the leading standard basis vectors are the cheapest choice. Only
    ``sigma`` (a per-dimension population statistic, well-estimated from as
    few as 2 samples per dimension, unlike a joint eigenbasis) carries real
    information here.
    """
    sigma = jnp.std(flat_positions, axis=0)
    sigma = jnp.where(sigma <= 0.0, 1.0, sigma)
    d = flat_positions.shape[-1]
    U = jnp.eye(d, k)
    lam = jnp.ones((k,))
    return sigma, U, lam


_LRD_EIGENVALUE_FLOOR = 1e-6


def _floor_lrd_eigenvalues(lam: Array) -> Array:
    """Clamp low-rank eigenvalues away from 0.

    A collinear or otherwise rank-deficient ensemble (e.g. a rank-1
    initial ensemble) can make the sample/accumulated correlation matrix
    singular along one or more of the selected top-k directions, giving
    ``lam ~ 0`` — and ``float32`` ``eigh`` can even return slightly
    *negative* eigenvalues, whose ``sqrt`` is NaN. The whitening transform
    (``_low_rank_precondition_pos``) and the momentum metric both scale by
    ``sqrt(lam)``, so flooring keeps those factors finite. This guard is
    intentionally redundant with the step-size decoupling (which keeps
    ``sqrt(lam)`` out of the step-size heuristic entirely): the degenerate
    collapse (``rhat = inf``, NaN step size) only occurs if *both* guards
    are defeated.
    """
    return jnp.maximum(lam, _LRD_EIGENVALUE_FLOOR)


def meads_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    num_folds: int = 4,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
    adaptation_info_fn: Callable = return_all_adapt_info,
    low_rank_rank: int | None = None,
    low_rank_window_fraction: float = 0.5,
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
        from the **full population of all ``num_chains`` chains** (requires
        :func:`blackjax.mcmc.ghmc`'s dense/low-rank momentum-metric support,
        blackjax#950), generalizing MEADS the way MCLMC-LRD generalized
        MCLMC. Unlike the diagonal scale (estimated per-fold, from each
        fold's own ``num_chains // num_folds`` chains), the low-rank
        eigenbasis is estimated *once per step* from the pooled global
        population and then shared across all folds: a single fold's
        ensemble (paper default ``num_folds=4`` gives only 16 chains/fold)
        is too small for its top-k eigenvectors to be stable step-to-step,
        and the resulting jitter destabilizes ghmc's persistent momentum
        (measured regression: low-rank underperformed diagonal at
        ``num_folds=4`` despite beating it at ``num_folds=1``, where the
        per-fold estimate happens to already be the global one). The metric
        is a shared symmetric preconditioner, not a per-fold statistic like
        step size or damping, so pooling all chains to estimate it needs no
        special justification -- it is the same practice window adaptation
        uses for its diagonal/dense metric. The per-fold step-size and
        damping heuristics (Algorithm 3) are otherwise unchanged, except they
        now whiten by this shared global metric rather than a per-fold one,
        so they stay consistent with the metric ghmc actually samples with.
        The rank is clamped to ``min(low_rank_rank, num_chains - 1, d)`` (raises
        ``ValueError`` if ``num_chains - 1 < 1``). A rank-``d`` metric equals the
        full dense metric, so clamping by ``d`` is lossless and prevents shape
        disagreements in the jax.lax.cond branches. The metric *returned* by
        ``run()`` is the final state of the same window-accumulated estimator
        described under ``low_rank_window_fraction`` below.

        Two further fixes address a validated high-dimension (``d >>
        num_chains``) failure mode where a single-snapshot low-rank metric
        made MEADS-LRD *worse* than the diagonal baseline (a p >> n noise-
        dominated eigenbasis fed into ghmc's step-size heuristic collapsed
        ``epsilon`` to ~1e-3 and froze the chains):

        - The step-size heuristic (Algorithm 3, line 8) always whitens its
          gradients by the plain per-fold diagonal scale (``grad * sigma``),
          never by the low-rank metric, even when ``low_rank_rank`` is set.
          Whitening ``epsilon`` by a noisy low-rank eigenbasis couples the
          step size to whichever direction the estimate currently
          over-weights, which is what caused the collapse above; the
          low-rank metric still preconditions the *momentum* (where it
          helps), just not the step-size proxy.
        - Selected eigenvalues are floored away from 0 (see
          ``_floor_lrd_eigenvalues``) so a collinear/rank-deficient initial
          ensemble can't seed a degenerate metric that self-reinforces into
          ``rhat = inf``. Collinear / near-collinear initial ensembles
          (e.g. all chains on a 1-D offset line) do not crash — two redundant
          guards (the step-size decoupling and the eigenvalue floor) prevent
          the NaN collapse — but expect severe under-mixing (measured
          rhat≈5 on a rank-1 init); use a dispersed, full-rank initialization.
    low_rank_window_fraction
        Only used when ``low_rank_rank`` is not ``None``. Fraction of
        warmup steps, counted from the end, over which the low-rank metric's
        covariance is accumulated (default ``0.5``: the last half of
        warmup). A single ``num_chains``-sized ensemble snapshot is
        ``p >> n`` noise-dominated once the dimension ``d`` exceeds
        ``num_chains`` -- exactly the regime a single fold's estimate was
        already too noisy for (see ``low_rank_rank`` above), just worse,
        since now the *whole* population's snapshot is undersized too, and
        (measured directly) even routing through it as a one-off fallback
        is enough to destabilize the ensemble. Instead, a running
        Chan/Welford covariance accumulator (mirroring the pattern
        :func:`~blackjax.adaptation.mass_matrix.welford_algorithm` uses for
        the mass matrix, generalized to a whole ensemble per step) is
        updated with every chain's position at every step *inside* the
        window, giving an effective sample size of ``num_chains *
        window_steps``. Once that effective size exceeds ``2 * d`` (a bare
        minimum for the estimate to not be noise-dominated), the low-rank
        momentum metric switches on and keeps improving every further
        window step; before that point -- either because the step is
        before the window (the initial, still-transient fraction of
        warmup, mirroring why Stan's window adaptation excludes its own
        initial/final fast windows from mass-matrix estimation), or because
        the window hasn't yet pooled ``2 * d`` samples -- the momentum
        metric falls back to a purely diagonal one
        (:func:`_lrd_diagonal_fallback`, i.e. no low-rank correction at
        all, matching ``low_rank_rank=None``'s momentum exactly), never a
        low-rank estimate from too little data.
        Must be in ``[0.0, 1.0]``; ``0.0`` accumulates from step 0, ``1.0``
        disables accumulation entirely (falls back to the purely diagonal
        momentum metric throughout the run).

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
        low_rank_k = min(low_rank_rank, num_chains - 1)
        if low_rank_k < 1:
            raise ValueError(
                f"low_rank_rank={low_rank_rank} cannot be honored: the "
                f"low-rank metric is estimated from the full population of "
                f"num_chains={num_chains} chains, and that estimate needs "
                "num_chains - 1 >= 1. Increase num_chains."
            )
        if not 0.0 <= low_rank_window_fraction <= 1.0:
            raise ValueError(
                "low_rank_window_fraction must be in [0.0, 1.0], got "
                f"{low_rank_window_fraction}."
            )

    ghmc_kernel = mcmc.ghmc.build_kernel()
    adapt_init, _ = base(num_folds, step_size_multiplier, damping_slowdown)
    batch_init = jax.vmap(lambda p, r: mcmc.ghmc.init(p, logdensity_fn, r))

    def one_step(carry, xs):
        rng_key, in_window = xs
        states, adaptation_state, lrd_accum = carry
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

        # MEADS-LRD: estimate ONE rank-`low_rank_k` correlation eigenbasis
        # (sigma, U, lam) per step from the FULL population of all
        # num_chains chains -- NOT from each fold's own noisy n_per_fold
        # snapshot. A fold's ensemble is too small (paper default
        # n_per_fold=16) for its top-k eigenvectors to be stable
        # step-to-step; the resulting eigenvector jitter destabilizes ghmc's
        # persistent momentum. The metric is a shared symmetric
        # preconditioner (unlike step-size/damping, which stay genuinely
        # per-fold below), so pooling all chains to estimate it needs no
        # per-fold isolation -- the same practice window adaptation uses for
        # its diagonal/dense metric. This low-rank metric then preconditions
        # the damping heuristic below (via `_low_rank_precondition_pos`), so
        # alpha/delta stay consistent with the metric actually handed to
        # ghmc for sampling this step. It does NOT precondition the
        # step-size heuristic (see the ε-decouple fix just below).
        #
        # FIX (high-d, p >> n): a single num_chains-sized snapshot is noise-
        # dominated once d exceeds num_chains, so `lrd_accum` -- a running
        # Chan/Welford covariance pooled over every chain at every step
        # *inside* the accumulation window (see `low_rank_window_fraction`'s
        # docstring) -- is used instead once its effective sample count
        # exceeds 2*d (a bare minimum for a covariance estimate to not be
        # noise-dominated). Before that (outside the window, or too early
        # inside it), fall back to a purely diagonal metric
        # (`_lrd_diagonal_fallback`) rather than ever attempting a low-rank
        # estimate from too little data -- see that function's docstring for
        # why a single-snapshot estimate is unsafe even as a fallback.
        if low_rank_rank is not None:
            flat_all_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(states.position)
            d = flat_all_pos.shape[-1]
            flat_folded_pos = flat_all_pos.reshape((num_folds, n_per_fold, d))

            updated_lrd_accum = jax.lax.cond(
                in_window,
                lambda a: _lrd_accumulator_update(a, flat_all_pos),
                lambda a: a,
                lrd_accum,
            )
            enough_accumulated = updated_lrd_accum.count >= 2 * d
            use_accumulated = jnp.logical_and(in_window, enough_accumulated)
            global_sigma, global_U, global_lam = jax.lax.cond(
                use_accumulated,
                lambda a: _lrd_from_accumulated_covariance(a, low_rank_k),
                lambda a: _lrd_diagonal_fallback(flat_all_pos, low_rank_k),
                updated_lrd_accum,
            )
            global_lam = _floor_lrd_eigenvalues(global_lam)
        else:
            updated_lrd_accum = lrd_accum

        # ε-decouple: the step-size heuristic (Algorithm 3, line 8) always
        # whitens by the plain per-fold diagonal scale, never by the
        # low-rank metric -- whitening it by a noisy low-rank eigenbasis
        # couples epsilon to whichever direction the estimate currently
        # over-weights (measured: collapsed epsilon ~20x at d=390), and the
        # low-rank metric's benefit is in the momentum, not the step size.
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
            # Same shared global (sigma, U, lam) as the step-size block above
            # -- broadcast across folds (in_axes=None), not rolled: unlike
            # the per-fold diagonal scale, there is only one metric this
            # step, so every fold whitens by (and every chain samples with)
            # the same eigenbasis.
            precond_pos_for_damping = jax.vmap(
                _low_rank_precondition_pos, in_axes=(0, None, None, None)
            )(flat_folded_pos, global_sigma, global_U, global_lam)
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
            # Feed ghmc the SAME global LowRankInverseMassMatrix for every
            # chain (no per-fold rolling -- there is only one metric this
            # step); blackjax#950 lets ghmc's momentum_inverse_scale accept
            # this directly (no elementwise squaring, unlike the legacy
            # diagonal path).
            chain_momentum_inverse_scale = LowRankInverseMassMatrix(
                sigma=jnp.repeat(global_sigma[None], num_chains, axis=0),
                U=jnp.repeat(global_U[None], num_chains, axis=0),
                lam=jnp.repeat(global_lam[None], num_chains, axis=0),
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

        return (
            new_states,
            new_adaptation_state,
            updated_lrd_accum,
        ), adaptation_info_fn(new_states, info, new_adaptation_state)

    def run(rng_key: PRNGKey, positions: ArrayLikeTree, num_steps: int = 1000):
        key_init, key_adapt = jax.random.split(rng_key)

        rng_keys = jax.random.split(key_init, num_chains)
        init_states = batch_init(positions, rng_keys)
        init_adaptation_state = adapt_init(positions, init_states.logdensity_grad)

        if low_rank_rank is not None:
            # Accumulate the covariance over the LAST `low_rank_window_fraction`
            # of warmup steps only (skip the still-transient early fraction),
            # mirroring how Stan's window adaptation delimits its own windows.
            # `window_start`/`in_window_flags` are plain Python/concrete-shape
            # values -- num_steps is always a concrete int here, never traced.
            window_start = int(low_rank_window_fraction * num_steps)
            flat_init_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(init_states.position)
            d = flat_init_pos.shape[-1]
            # Clamp the rank to the flattened dimension as well: rank > d makes the
            # two jax.lax.cond branches disagree on output shapes. A rank-d metric
            # equals the full dense metric, so this clamp is lossless.
            nonlocal low_rank_k
            low_rank_k = min(low_rank_k, d)
            init_lrd_accum = _lrd_accumulator_init(d)
        else:
            window_start = num_steps
            init_lrd_accum = None
        in_window_flags = jnp.arange(num_steps) >= window_start

        keys = jax.random.split(key_adapt, num_steps)
        (last_states, last_adaptation_state, last_lrd_accum), info = jax.lax.scan(
            one_step,
            (init_states, init_adaptation_state, init_lrd_accum),
            (keys, in_window_flags),
        )

        if low_rank_rank is not None:
            # The metric returned to the caller is the FINAL state of the
            # same window-accumulated estimator `one_step` uses per-step
            # (effective n = num_chains * window_steps), falling back to the
            # diagonal-only estimate only if the window never accumulated
            # enough samples (e.g. num_steps too small). See
            # `low_rank_window_fraction`'s docstring. `low_rank_k` (==
            # min(low_rank_rank, num_chains - 1)) is reused unchanged.
            flat_final_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(
                last_states.position
            )
            # low_rank_k is set to a non-None int whenever low_rank_rank is
            # not None (validated above); assert narrows it for mypy.
            assert low_rank_k is not None
            d_final = flat_final_pos.shape[-1]
            use_accumulated_final = last_lrd_accum.count >= 2 * d_final
            final_sigma, final_U, final_lam = jax.lax.cond(
                use_accumulated_final,
                lambda a: _lrd_from_accumulated_covariance(a, low_rank_k),
                lambda a: _lrd_diagonal_fallback(flat_final_pos, low_rank_k),
                last_lrd_accum,
            )
            final_lam = _floor_lrd_eigenvalues(final_lam)
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
