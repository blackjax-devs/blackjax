"""Public API for ChEES-HMC"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import blackjax.optimizers.dual_averaging as dual_averaging
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.mass_matrix import welford_algorithm
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

# optimal tuning for HMC, see https://arxiv.org/abs/1001.4460
OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651
# Clip the final log-space update like the original implementation in TFP (~log(2)/2 ≈ 0.35).
LOG_UPDATE_CLIP = 0.35
# Small constant to avoid division by zero or log of zero
EPS_FLOAT = 1e-20


class ChEESAdaptationState(NamedTuple):
    """State of the ChEES-HMC adaptation scheme.

    step_size
        Value of the step_size parameter of the HMC algorithm.
    log_step_size_moving_average
        Running moving average of the log step_size parameter.
    trajectory_length
        Value of the num_integration_steps * step_size parameter of
        the HMC algorithm.
    log_trajectory_length_moving_average
        Running moving average of the log num_integration_steps / step_size
        parameter.
    optim_state
        Optax optimizing state for used to maximize the ChEES criterion.
    random_generator_arg
        Utility array for generating a pseudo or quasi-random sequence of
        numbers.
    step
        Current iteration number.

    """

    step_size: float
    log_step_size_moving_average: float
    trajectory_length: float
    log_trajectory_length_moving_average: float
    da_state: dual_averaging.DualAveragingState
    optim_state: optax.OptState
    random_generator_arg: Array
    step: int


def _mass_matrix_engagement_threshold(num_dim: int) -> int:
    """Minimum pooled Welford sample count before ``mass_matrix_estimation``'s
    diagonal estimate engages (see ``chees_adaptation``'s docstring).

    A per-dimension variance estimate, unlike a joint eigenbasis (the
    ``meads_adaptation`` low-rank case, gated at ``2 * d``), does not need
    ``O(d)`` samples to escape noise domination -- each dimension is
    estimated independently. A small constant floor (64) is enough on its
    own for typical dimensions; the mild ``2 * sqrt(d)`` term only matters
    for very high-dimensional targets, where it grows the floor slightly.
    """
    return max(64, int(2 * np.sqrt(num_dim)))


def _diagonal_mass_matrix_or_fallback(mm_accum, threshold: int, num_dim: int) -> Array:
    """Return the Welford-estimated diagonal ``inverse_mass_matrix`` once
    ``mm_accum.sample_size`` reaches ``threshold``, otherwise fall back to
    ``jnp.ones(num_dim)`` -- the pre-estimation / ``mass_matrix_estimation=
    None`` behavior. Mirrors ``meads_adaptation``'s
    ``_lrd_diagonal_fallback``/engagement-gate pattern, specialized to the
    single diagonal (no low-rank) case.
    """
    _, _, wc_final = welford_algorithm(is_diagonal_matrix=True)
    enough = mm_accum.sample_size >= threshold
    return jax.lax.cond(
        enough,
        lambda acc: jnp.maximum(wc_final(acc)[0], EPS_FLOAT),
        lambda acc: jnp.ones(num_dim),
        mm_accum,
    )


# --- Slow-direction trajectory-length floor -----------------------------
# See `chees_adaptation`'s `mass_matrix_estimation` docstring / the module's
# commit message for the full derivation. Summary: composing a diagonal
# metric with ChEES's own trajectory-length criterion converges a near
# metric-invariant physical length (dominated by the well-conditioned bulk
# dimensions), which under-serves any residual (off-diagonal) slow
# correlation direction the diagonal metric can't remove. A whitened
# direction with eigenvalue lambda undergoes simple-harmonic HMC motion with
# oscillation period 2*pi*sqrt(lambda); a quarter turn is (pi/2)*sqrt(lambda).
# ChEES's own criterion already converges close to this quarter-turn rule
# for the bulk (lambda ~= 1 -> ChEES's own adapted length ~= pi/2, confirmed
# empirically), so flooring the consumed trajectory length at
# CHEES_LENGTH_FLOOR_FACTOR * sqrt(lambda_max) -- lambda_max the top
# eigenvalue of the WHITENED ensemble covariance -- extends the same rule to
# the direction the bulk-dominated criterion under-serves. Applied at
# CONSUMPTION only (see `_apply_length_floor`): never fed back into the
# ChEES optimizer's own state, so the floor is cleanly ablatable via the
# private `_length_floor` seam and has no effect on `base`/`compute_
# parameters` (unmodified by this feature).
CHEES_LENGTH_FLOOR_FACTOR: float = np.pi / 2
# Cadence (in adaptation steps) at which lambda_max is refreshed via power
# iteration. A full eigh every step would cost O(num_dim^3); power iteration
# is O(num_dim^2) per iteration, and warm-starting the eigenvector from the
# previous recompute means only a handful of iterations are needed to track
# a slowly-drifting estimate, so refreshing every
# `_LENGTH_FLOOR_RECOMPUTE_INTERVAL` steps (rather than every step) keeps
# this cheap even at large num_dim (e.g. 390).
_LENGTH_FLOOR_RECOMPUTE_INTERVAL = 32
_LENGTH_FLOOR_POWER_ITERATIONS = 5
_LENGTH_FLOOR_FINAL_POWER_ITERATIONS = 20
# Floors lambda_max away from 0 -- mirrors meads_adaptation's
# _LRD_EIGENVALUE_FLOOR (1e-6): a collinear/rank-deficient accumulated
# covariance can give a near-zero (or, from floating-point cancellation,
# slightly negative) top eigenvalue, whose sqrt would be NaN.
_LENGTH_FLOOR_LAMBDA_EPS = 1e-6


class _ChEESCovAccumulatorState(NamedTuple):
    """Running Chan/Welford full covariance accumulator for the slow-
    direction trajectory-length floor's ``lambda_max`` estimate.

    Mirrors :class:`~blackjax.adaptation.metric_buffers.MomentBlock` (identical
    mean/M2/count recurrence) -- mirrored here rather than imported, to keep
    this feature's diff self-contained to ``chees_adaptation.py``
    (``meads_adaptation.py`` is out of scope for this change). Accumulated
    over the SAME late-warmup window, from the SAME per-step ensemble batch,
    as the diagonal mass-matrix Welford accumulator (``mm_accum``) -- but
    tracks the full ``num_dim x num_dim`` second moment (not just the
    diagonal), since the floor needs the ensemble's cross-dimension
    structure, not just its per-dimension scale.
    """

    mean: Array
    m2: Array
    count: Array


def _cov_accumulator_init(num_dim: int) -> _ChEESCovAccumulatorState:
    return _ChEESCovAccumulatorState(
        mean=jnp.zeros((num_dim,)),
        m2=jnp.zeros((num_dim, num_dim)),
        count=jnp.zeros(()),
    )


def _cov_accumulator_update(
    acc: _ChEESCovAccumulatorState, batch: Array
) -> _ChEESCovAccumulatorState:
    """Merge a batch of samples, shape ``(n_b, num_dim)`` (one step's
    ensemble of ``num_chains`` positions), into the running accumulator via
    Chan et al.'s parallel/batch generalization of Welford's algorithm --
    the same recurrence :func:`~blackjax.adaptation.metric_buffers.cgl_update_batch` uses.
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
    return _ChEESCovAccumulatorState(mean=mean_ab, m2=m2_ab, count=n_ab)


class _ChEESEigState(NamedTuple):
    """Warm-startable top-eigenvector/eigenvalue estimate of the WHITENED
    ensemble covariance (``D^{-1/2} C_hat D^{-1/2}``, ``D`` = the currently
    engaged diagonal ``inverse_mass_matrix``), refreshed every
    ``_LENGTH_FLOOR_RECOMPUTE_INTERVAL`` steps by a few power iterations
    (see ``_power_iteration_lambda_max``) warm-started on ``eigenvector``.
    """

    eigenvector: Array
    lambda_max: Array


def _eig_state_init(num_dim: int) -> _ChEESEigState:
    v0 = jnp.ones((num_dim,)) / jnp.sqrt(num_dim)
    return _ChEESEigState(eigenvector=v0, lambda_max=jnp.ones(()))


def _power_iteration_lambda_max(
    matrix: Array, v0: Array, num_iterations: int
) -> tuple[Array, Array]:
    """A handful of power iterations of the symmetric PSD ``matrix``, warm-
    started from ``v0``. Returns the top-eigenvalue estimate (the Rayleigh
    quotient of the converged direction) and the normalized eigenvector --
    the latter to warm-start the NEXT recompute. ``O(num_dim**2)`` per
    iteration, versus a full ``eigh``'s ``O(num_dim**3)`` -- see the module
    comment above ``CHEES_LENGTH_FLOOR_FACTOR`` for why a full eigh every
    step is avoided.
    """

    def body(_, v):
        v_next = matrix @ v
        norm = jnp.linalg.norm(v_next)
        return v_next / jnp.where(norm > 0.0, norm, 1.0)

    v = jax.lax.fori_loop(0, num_iterations, body, v0)
    lambda_max = jnp.dot(v, matrix @ v)
    return lambda_max, v


def _recompute_eig_state(
    cov_accum: _ChEESCovAccumulatorState,
    inverse_mass_matrix: Array,
    eig_state: _ChEESEigState,
    num_iterations: int = _LENGTH_FLOOR_POWER_ITERATIONS,
) -> _ChEESEigState:
    """Whiten the accumulated covariance by the currently engaged diagonal
    ``inverse_mass_matrix`` (giving the correlation-like matrix
    ``D^{-1/2} C_hat D^{-1/2}``) and refresh the top-eigenvalue estimate via
    power iteration warm-started on ``eig_state.eigenvector``.
    """
    covariance = cov_accum.m2 / jnp.maximum(cov_accum.count - 1.0, 1.0)
    inv_sqrt_d = 1.0 / jnp.sqrt(inverse_mass_matrix)
    whitened_covariance = covariance * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
    lambda_max, eigenvector = _power_iteration_lambda_max(
        whitened_covariance, eig_state.eigenvector, num_iterations
    )
    return _ChEESEigState(
        eigenvector=eigenvector,
        lambda_max=jnp.maximum(lambda_max, _LENGTH_FLOOR_LAMBDA_EPS),
    )


def _apply_length_floor(
    trajectory_length: Array,
    lambda_max: Array,
    engaged: Array,
    enable: bool,
    max_leapfrog_steps: int = 1000,
    step_size: float = 0.1,
) -> tuple[Array, Array]:
    """Floor ``trajectory_length`` at
    ``CHEES_LENGTH_FLOOR_FACTOR * sqrt(lambda_max)`` -- the slow-direction
    trajectory-length floor. Applied at CONSUMPTION only: a pure function of
    the already-adapted length and the current ``lambda_max`` estimate,
    never fed back into the ChEES optimizer's own state, so the floor is
    cleanly ablatable. The consumed length respects the user's budget:
    ``min(max(adapted, floor), max_leapfrog_steps · step_size)``.

    Returns the consumed trajectory_length and a boolean diagnostic flag
    ``floor_clipped_by_cap`` (True when the cap binds below the slow-direction
    floor): if the cap constrains the floor below its floor value, raise
    ``max_leapfrog_steps`` to restore the guarantee.

    ``enable`` (the private ``_length_floor`` seam) is a static Python bool:
    when False, ``trajectory_length`` is returned completely unchanged --
    no ``lambda_max``/``engaged`` computation is even required to have run.
    ``engaged`` (a traced boolean, the SAME support gate as the diagonal
    mass-matrix estimate) makes the floor a no-op before that gate -- pre-
    gate, ``lambda_max`` is not yet a meaningful estimate.

    At the default ``max_leapfrog_steps=1000``, the cap is 1000·step_size, and
    ``λ_max ≤ d`` for a correlation matrix means the floor can never bind
    (belt-and-suspenders insurance; ``floor_clipped_by_cap`` will be False).
    """
    if not enable:
        return trajectory_length, jnp.asarray(False)

    floor_value = jnp.where(
        engaged, CHEES_LENGTH_FLOOR_FACTOR * jnp.sqrt(lambda_max), 0.0
    )
    cap = max_leapfrog_steps * step_size
    floored_length = jnp.maximum(trajectory_length, floor_value)
    consumed_length = jnp.minimum(floored_length, cap)
    floor_clipped_by_cap = jnp.logical_and(
        jnp.logical_and(engaged, enable), floor_value > cap
    )
    return consumed_length, floor_clipped_by_cap


def weighted_empirical_mean(x, w):
    # x: (num_chains, dim), w: (num_chains,)
    x_safe = jnp.where(jnp.isfinite(x), x, 0.0)
    w = jnp.where(jnp.isfinite(x).all(axis=-1), w, 0.0)

    w_exp = w.reshape((w.shape[0],) + (1,) * (x.ndim - 1))
    num = jnp.sum(w_exp * x_safe, axis=0)
    den = jnp.sum(w_exp, axis=0) + EPS_FLOAT
    return jax.lax.stop_gradient(num / den)


def base(
    jitter_generator: Callable,
    next_random_arg_fn: Callable,
    optim: optax.GradientTransformation,
    target_acceptance_rate: float,
    decay_rate: float,
    max_leapfrog_steps: int,
    _whiten_criterion: bool = True,
) -> tuple[Callable, Callable]:
    """Maximizing the Change in the Estimator of the Expected Square criterion
    (trajectory length) and dual averaging procedure (step size) for the jittered
    Hamiltonian Monte Carlo kernel :cite:p:`hoffman2021adaptive`.

    This adaptation algorithm tunes the step size and trajectory length, i.e.
    number of integration steps / step size, of the jittered HMC algorithm based
    on statistics collected from a population of many chains. It maximizes the Change
    in the Estimator of the Expected Square (ChEES) criterion to tune the trajectory
    length and uses dual averaging targeting an acceptance rate of 0.651 of the harmonic
    mean of the chain's acceptance probabilities to tune the step size.

    Parameters
    ----------
    jitter_generator
        Optional function that generates a value in [0, 1] used to jitter the trajectory
        lengths given a PRNGKey, used to propose the number of integration steps. If None,
        then a quasi-random Halton is used to jitter the trajectory length.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    optim
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol.
    target_acceptance_rate
        Average acceptance rate to target with dual averaging.
    decay_rate
        Float representing how much to favor recent iterations over earlier ones in the optimization
        of step size and trajectory length.
    _whiten_criterion
        Private, undocumented ablation seam (not part of the public API). When
        True (default) the ChEES criterion accounts for a non-identity
        ``inverse_mass_matrix`` passed to ``update`` (see the derivation in
        ``compute_parameters``). When False, the criterion is computed exactly
        as the original identity-mass-matrix expression regardless of the
        ``inverse_mass_matrix`` passed in -- used in validation studies to
        isolate the whitening correction's contribution relative to the mass
        matrix alone.


    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.

    """

    da_init, da_update, _ = dual_averaging.dual_averaging()

    def compute_parameters(
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
        initial_adaptation_state: ChEESAdaptationState,
        inverse_mass_matrix: Array,
    ) -> ChEESAdaptationState:
        """Compute values for the parameters based on statistics collected from
        multiple chains.

        Parameters
        ----------
        proposed_positions:
            A PyTree that contains the position proposed by the HMC algorithm of
            every chain (proposal that is accepted or rejected using MH).
        proposed_momentums:
            A PyTree that contains the momentum variable proposed by the HMC algorithm
            of every chain (proposal that is accepted or rejected using MH).
        initial_positions:
            A PyTree that contains the initial position at the start of the HMC
            algorithm of every chain.
        acceptance_probabilities:
            Metropolis-Hastings acceptance probabilty of proposals of every chain.
        initial_adaptation_state:
            ChEES adaptation step used to generate proposals and acceptance probabilities.
        inverse_mass_matrix:
            The diagonal ``inverse_mass_matrix`` (Σ) the kernel used to generate this
            step's proposals -- ``jnp.ones(num_dim)`` unless
            ``chees_adaptation``'s ``mass_matrix_estimation="diagonal"`` has engaged.
            Used to whiten the ChEES criterion so it stays metric-consistent with
            the kernel (see the derivation below). Has no effect when equal to
            ``jnp.ones(num_dim)``, since the whitening reduces to a no-op there.

        Returns
        -------
        New values of the step size and trajectory length of the jittered HMC algorithm.

        """
        (
            step_size,
            log_step_size_ma,
            trajectory_length,
            log_trajectory_length_ma,
            da_state,
            optim_state,
            random_generator_arg,
            step,
        ) = initial_adaptation_state

        harmonic_mean = 1.0 / jnp.mean(
            1.0 / acceptance_probabilities, where=~is_divergent
        )
        # Replace inf/nan harmonic mean as zero to avoid issues in dual averaging
        harmonic_mean = jnp.where(jnp.isfinite(harmonic_mean), harmonic_mean, 0.0)
        da_state_ = da_update(da_state, target_acceptance_rate - harmonic_mean)
        step_size_ = jnp.exp(da_state_.log_x)
        new_step_size, new_da_state, new_log_step_size = jax.lax.cond(
            jnp.isfinite(step_size_),
            lambda _: (step_size_, da_state_, da_state_.log_x),
            lambda _: (step_size, da_state, da_state.log_x),
            None,
        )
        update_weight = step ** (-decay_rate)
        new_log_step_size_ma = (
            1.0 - update_weight
        ) * log_step_size_ma + update_weight * new_log_step_size

        w = jnp.where(~is_divergent, acceptance_probabilities, 0.0)
        proposals_mean = jax.tree.map(
            lambda p: weighted_empirical_mean(p, w), proposed_positions
        )
        # The above weighted mean is presumably better than the simple mean:
        # proposals_mean = jax.tree.map(
        #     lambda p: jnp.nanmean(p, axis=0), proposed_positions
        # )
        initials_mean = jax.tree.map(
            lambda p: jnp.nanmean(p, axis=0), initial_positions
        )
        proposals_centered = jax.tree.map(
            lambda p, pm: p - pm, proposed_positions, proposals_mean
        )
        initials_centered = jax.tree.map(
            lambda p, pm: p - pm, initial_positions, initials_mean
        )

        vmap_flatten_op = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])
        proposals_matrix = vmap_flatten_op(proposals_centered)
        initials_matrix = vmap_flatten_op(initials_centered)
        momentums_matrix = vmap_flatten_op(proposed_momentums)

        # --- Metric-aware (whitened) ChEES criterion ------------------------
        # ChEES's trajectory-length gradient is a pathwise-derivative estimator
        # of d/dL E[||x' - E[x']||^2], built from the product
        #     (||Δx'||^2 - ||Δx||^2) * <Δx', dx'/dL>
        # where dx'/dL is proportional to the trajectory's *velocity* at the
        # endpoint. Under Hamiltonian dynamics with mass matrix M = Σ^-1,
        # velocity v = M^-1 p = Σp -- this is blackjax's own convention, e.g.
        # metrics.py's `linear_map(inverse_mass_matrix, momentum)` kinetic-
        # energy term and the NUTS U-turn check. The pre-existing code below
        # implicitly assumes M = I (so v == p): that's exactly what TFP's own
        # implementation comment flags at this spot ("implicitly assumes an
        # identity mass matrix") -- there is no reference implementation for
        # the general-M case to copy, hence the derivation here.
        #
        # For a general diagonal M, the *norm* term must be measured in the
        # geometry M induces -- the whitened space x̃ = Σ^{-1/2}x in which the
        # preconditioned sampler behaves isotropically -- otherwise a handful
        # of large-variance dimensions dominate ||Δx||^2 and swamp the
        # trajectory-length signal coming from the well-scaled ones. That is
        # edit 1: proposals/initials get whitened by Σ^{-1/2} before the dot
        # products that build the norm difference.
        #
        # The <Δx', v'> term needs the SAME whitening transform applied to
        # velocity: velocity is a tangent vector living in position-space (not
        # momentum's dual/cotangent space), so under x̃ = Σ^{-1/2}x it
        # transforms the same way position differences do, i.e.
        # ṽ = Σ^{-1/2}v -- NOT the contragredient transform momentum itself
        # would get. That is edit 2, and skipping it is the "silent-bug trap":
        # reusing the edit-1-whitened Δx' in `jnp.dot(pm, mm)` without also
        # whitening the velocity leaves a spurious Σ^{-1/2} factor on the
        # momentum term. Composing both edits explicitly:
        #     v'  = Σ  p'                (velocity, metrics.py convention)
        #     ṽ' = Σ^{-1/2} v' = Σ^{1/2} p'   (velocity whitened like position)
        #     <x̃', ṽ'> = Σ_i (Δx'_i / √Σ_i) · (√Σ_i · p'_i) = Σ_i Δx'_i p'_i
        #              = <Δx', p'>   (the RAW expression, for *any* diagonal Σ)
        # This exact cancellation is not a coincidence: Δx' and p' are
        # canonically conjugate (position, momentum), so their pairing is
        # metric-invariant -- unlike ||Δx||^2 alone. The code below still
        # computes both quantities via the explicit whitening transforms
        # (rather than hand-simplifying to the raw dot product) so that (a)
        # this derivation is directly checkable against the code and (b) with
        # Σ = ones every multiply/divide is an IEEE-754 no-op, so the whitened
        # path is bit-for-bit identical to the original expression -- see
        # ``test_whitened_criterion_reduces_to_raw_when_identity``.
        #
        # `_whiten_criterion=False` (private ablation seam, see `base`'s
        # docstring) skips both edits and always uses the raw, pre-existing
        # expression regardless of `inverse_mass_matrix` -- this is the
        # "naive" arm (kernel uses the estimated diagonal metric, but the
        # criterion does not) the validation study compares against.
        if _whiten_criterion:
            inv_sqrt_imm = 1.0 / jnp.sqrt(inverse_mass_matrix)
            proposals_matrix_w = proposals_matrix * inv_sqrt_imm
            initials_matrix_w = initials_matrix * inv_sqrt_imm
            velocity_matrix = momentums_matrix * inverse_mass_matrix  # v' = Σp'
            velocity_matrix_w = velocity_matrix * inv_sqrt_imm  # ṽ' = Σ^{-1/2}v'
        else:
            proposals_matrix_w = proposals_matrix
            initials_matrix_w = initials_matrix
            velocity_matrix_w = momentums_matrix

        trajectory_gradients = (
            jitter_generator(random_generator_arg)
            * trajectory_length  # this effectively make this gradient w.r.t. log_trajectory_length
            * jax.vmap(
                lambda pm, im, mm: (jnp.dot(pm, pm) - jnp.dot(im, im)) * jnp.dot(pm, mm)
            )(proposals_matrix_w, initials_matrix_w, velocity_matrix_w)
        )

        trajectory_gradient = jnp.sum(
            acceptance_probabilities * trajectory_gradients,
            where=~is_divergent,
        ) / jnp.sum(acceptance_probabilities + EPS_FLOAT, where=~is_divergent)

        log_trajectory_length = jnp.log(trajectory_length)
        updates, optim_state_ = optim.update(
            trajectory_gradient, optim_state, log_trajectory_length
        )

        updates = jax.tree.map(
            lambda u: jnp.clip(u, -LOG_UPDATE_CLIP, LOG_UPDATE_CLIP), updates
        )
        log_trajectory_length_ = optax.apply_updates(log_trajectory_length, updates)
        new_log_trajectory_length, new_optim_state = jax.lax.cond(
            jnp.isfinite(
                jax.flatten_util.ravel_pytree(log_trajectory_length_)[0]
            ).all(),
            lambda _: (log_trajectory_length_, optim_state_),
            lambda _: (log_trajectory_length, optim_state),
            None,
        )
        new_log_trajectory_length_ma = (
            1.0 - update_weight
        ) * log_trajectory_length_ma + update_weight * new_log_trajectory_length
        new_trajectory_length = jnp.exp(new_log_trajectory_length_ma)

        # clip new trajectory length to avoid too large trajectories, also the
        # minimum trajectory length is one integrator step
        new_trajectory_length = jnp.clip(
            new_trajectory_length,
            max=max_leapfrog_steps * new_step_size,
            min=new_step_size,
        )
        return ChEESAdaptationState(
            new_step_size,
            new_log_step_size_ma,
            new_trajectory_length,
            new_log_trajectory_length_ma,
            new_da_state,
            new_optim_state,
            next_random_arg_fn(random_generator_arg),
            step + 1,
        )

    def init(random_generator_arg: Array, step_size: float):
        return ChEESAdaptationState(
            step_size=step_size,
            log_step_size_moving_average=0.0,
            trajectory_length=step_size,
            log_trajectory_length_moving_average=0.0,
            da_state=da_init(step_size),
            optim_state=optim.init(step_size),
            random_generator_arg=random_generator_arg,
            step=1,
        )

    def update(
        adaptation_state: ChEESAdaptationState,
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
        inverse_mass_matrix: Array,
    ):
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            The current state of the adaptation algorithm
        proposed_positions:
            The position proposed by the HMC algorithm of every chain.
        proposed_momentums:
            The momentum variable proposed by the HMC algorithm of every chain.
        initial_positions:
            The initial position at the start of the HMC algorithm of every chain.
        acceptance_probabilities:
            Metropolis-Hastings acceptance probabilty of proposals of every chain.
        inverse_mass_matrix:
            The diagonal ``inverse_mass_matrix`` the kernel used to generate this
            step's proposals; see ``compute_parameters`` for how it whitens the
            ChEES criterion.

        Returns
        -------
        New adaptation state that contains the step size and trajectory length of the
        jittered HMC algorithm.

        """
        new_state = compute_parameters(
            proposed_positions,
            proposed_momentums,
            initial_positions,
            acceptance_probabilities,
            is_divergent,
            adaptation_state,
            inverse_mass_matrix,
        )

        return new_state

    return init, update


def chees_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    *,
    jitter_generator: Callable | None = None,
    jitter_amount: float = 1.0,
    target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE,
    decay_rate: float = 0.5,
    max_leapfrog_steps: int = 1000,
    adaptation_info_fn: Callable = return_all_adapt_info,
    mass_matrix_estimation: str | None = None,
    mass_matrix_window_fraction: float = 0.5,
    _whiten_criterion: bool = True,
    _length_floor: bool = True,
) -> AdaptationAlgorithm:
    """Adapt the step size and trajectory length (number of integration steps / step size)
    parameters of the jittered HMC algorthm.

    The jittered HMC algorithm depends on the value of a step size, controlling
    the discretization step of the integrator, and a trajectory length, given by the
    number of integration steps / step size, jittered by using only a random percentage
    of this trajectory length.

    This adaptation algorithm tunes the trajectory length by heuristically maximizing
    the Change in the Estimator of the Expected Square (ChEES) criterion over
    an ensamble of parallel chains. At equilibrium, the algorithm aims at eliminating
    correlations between target dimensions, making the HMC algorithm efficient.

    Jittering requires generating a random sequence of uniform variables in [0, 1].
    However, this adds another source of variance to the sampling procedure,
    which may slow adaptation or lead to suboptimal mixing. To alleviate this,
    rather than use uniform random noise to jitter the trajectory lengths, we use a
    quasi-random Halton sequence, which ensures a more even distribution of trajectory
    lengths.

    Examples
    --------

    An HMC adapted kernel can be learned and used with the following code:

    .. code::

        warmup = blackjax.chees_adaptation(logdensity_fn, num_chains)
        key_warmup, key_sample = jax.random.split(rng_key)
        optim = optax.adam(learning_rate)
        (last_states, parameters), _ = warmup.run(
            key_warmup,
            positions, #PyTree where each leaf has shape (num_chains, ...)
            initial_step_size,
            optim,
            num_warmup_steps,
        )
        kernel = blackjax.dhmc(logdensity_fn, **parameters).step
        new_states, info = jax.vmap(kernel)(key_sample, last_states)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of chains used for cross-chain warm-up training.
    jitter_generator
        Optional function that generates a value in [0, 1] used to jitter the trajectory
        lengths given a PRNGKey, used to propose the number of integration steps. If None,
        then a quasi-random Halton is used to jitter the trajectory length.
    jitter_value
        A percentage in [0, 1] representing how much of the calculated trajectory should be jitted.
    target_acceptance_rate
        Average acceptance rate to target with dual averaging. Defaults to optimal tuning for HMC.
    decay_rate
        Float representing how much to favor recent iterations over earlier ones in the optimization
        of step size and trajectory length. A value of 1 gives equal weight to all history. A value
        of 0 gives weight only to the most recent iteration.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.
    mass_matrix_estimation
        Opt-in ensemble-estimated *diagonal* ``inverse_mass_matrix`` (opt-in,
        default ``None``). ``None`` keeps the original behavior, bit-for-bit:
        the kernel always uses ``inverse_mass_matrix=jnp.ones(num_dim)``.
        ``"diagonal"`` instead estimates a per-dimension variance from the
        ensemble of all ``num_chains`` chains via a running Welford
        accumulator (:func:`~blackjax.adaptation.mass_matrix.welford_algorithm`),
        accumulated over the *last* ``mass_matrix_window_fraction`` of warmup
        (see that parameter), and uses it as the kernel's diagonal
        ``inverse_mass_matrix`` once enough samples have accumulated (see
        "Engagement gate" below). ChEES's own trajectory-length criterion is
        also whitened by this estimate so it stays metric-consistent with the
        kernel it is tuning (see the derivation in
        :func:`base`'s ``compute_parameters``) -- this whitening keeps the
        criterion metric-consistent and responsive to the preconditioned geometry;
        measured effect is small (≲12%) on NCP-standardized targets where most
        dimensions are near unit scale. The feature's main value is delivered by
        the slow-direction *length floor* (when enabled), which recovers large
        ESS-per-gradient wins on targets with a residual slow correlation
        direction the diagonal metric cannot remove.

        **Scope: validated under located (typical-set) initializations.** On
        funnel-like targets the diagonal metric can *reduce* per-draw ESS
        relative to the identity metric's fine-step regime (an efficiency caveat,
        not a bias—posterior means are unaffected); the length floor recovers most
        of it. Cold or dispersed initialization on hard geometry is a separate
        warmup-robustness limitation, out of scope for this feature.

        GPU-scale validation (2× independent runs, ensembles up to 5000 chains): the
        ``"diagonal"`` metric with the trajectory-length floor additionally makes the
        adaptation robust to *dispersed* initializations — e.g. per-chain uniform box
        inits on the unconstrained space, which catastrophically break identity-metric
        ChEES on scale-separated targets (dispersion inflates the cross-chain jump-distance
        criterion, driving the adapted trajectory length down; the floor's √λ_max term
        grows with ensemble dispersion and self-corrects). A dispersed initialization is
        also what gives R̂ its diagnostic power: initializing all chains at a single point
        (e.g. the origin) can produce clean R̂ that is structurally blind to same-basin
        non-equilibrium — prefer dispersed inits and treat single-point-init R̂ with
        caution.

        Engagement gate: before the pooled accumulator (effective sample
        size ``num_chains * window_steps``) exceeds ``max(64, 2*sqrt(num_dim))``
        -- a modest floor chosen because a per-dimension variance estimate,
        unlike a joint eigenbasis, does not need ``O(d)`` samples to escape
        noise domination, so a small constant plus a mild ``d``-scaling for
        very high-dimensional targets is enough -- the kernel and criterion
        both use ``jnp.ones(num_dim)`` (identical to ``None``). The final
        ``inverse_mass_matrix`` returned by ``run()`` is the engaged-or-
        fallback estimate at the end of the accumulation window.
    mass_matrix_window_fraction
        Only used when ``mass_matrix_estimation="diagonal"``. Fraction of
        warmup steps, counted from the end, over which the diagonal
        ``inverse_mass_matrix``'s Welford accumulator collects samples
        (default ``0.5``: the last half of warmup, mirroring
        ``meads_adaptation``'s ``low_rank_window_fraction`` and Stan's
        practice of excluding window adaptation's own early/transient
        fraction). Must be in ``[0.0, 1.0]``; ``0.0`` accumulates from step
        0, ``1.0`` disables accumulation entirely (falls back to
        ``jnp.ones(num_dim)`` throughout the run).

    Returns
    -------
    A function that returns the last cross-chain state, a sampling kernel with the
    tuned parameter values, and all the warm-up states for diagnostics.

    """
    if mass_matrix_estimation not in (None, "diagonal"):
        raise ValueError(
            "mass_matrix_estimation must be None or 'diagonal', got "
            f"{mass_matrix_estimation!r}."
        )
    if not 0.0 <= mass_matrix_window_fraction <= 1.0:
        raise ValueError(
            "mass_matrix_window_fraction must be in [0.0, 1.0], got "
            f"{mass_matrix_window_fraction}."
        )
    estimate_mass_matrix = mass_matrix_estimation == "diagonal"
    # The slow-direction length floor (see the module comment above
    # CHEES_LENGTH_FLOOR_FACTOR) only ever engages when the diagonal mass
    # matrix is also engaged -- private `_length_floor=False` (the ablation
    # seam) skips building the extra d x d covariance accumulator entirely,
    # so the "metric-without-floor" ablation arm has zero overhead beyond
    # the pre-existing mass_matrix_estimation="diagonal" feature.
    enable_length_floor = estimate_mass_matrix and _length_floor

    def run(
        rng_key: PRNGKey,
        positions: ArrayLikeTree,
        step_size: float,
        optim: optax.GradientTransformation,
        num_steps: int = 1000,
        *,
        max_sampling_steps: int = 1000,
    ):
        assert all(
            jax.tree.flatten(
                jax.tree.map(lambda p: p.shape[0] == num_chains, positions)
            )[0]
        ), "initial `positions` leading dimension must be equal to the `num_chains`"
        num_dim = pytree_size(positions) // num_chains

        next_random_arg_fn = lambda i: i + 1
        init_random_arg = 0

        if jitter_generator is not None:
            rng_key, carry_key = jax.random.split(rng_key)
            jitter_gn = lambda i: jitter_generator(
                jax.random.fold_in(carry_key, i)
            ) * jitter_amount + (1.0 - jitter_amount)
        else:
            max_bits = np.ceil(np.log2(num_steps + max_sampling_steps))
            jitter_gn = lambda i: dynamic_hmc.halton_sequence(
                i, max_bits
            ) * jitter_amount + (1.0 - jitter_amount)

        def integration_steps_fn(random_generator_arg, num_leapfrog_steps):
            return jnp.asarray(
                jnp.ceil(jitter_gn(random_generator_arg) * num_leapfrog_steps),
                dtype=int,
            )

        step_fn = dynamic_hmc.build_kernel(
            next_random_arg_fn=next_random_arg_fn,
            integration_steps_fn=integration_steps_fn,
        )

        init, update = base(
            jitter_gn,
            next_random_arg_fn,
            optim,
            target_acceptance_rate,
            decay_rate,
            max_leapfrog_steps,
            _whiten_criterion,
        )

        # Opt-in ensemble-estimated diagonal inverse_mass_matrix (Σ). Mirrors
        # meads_adaptation's low_rank_rank / low_rank_window_fraction pattern:
        # a Welford accumulator (reused verbatim from mass_matrix.py, not
        # reinvented) pools every chain's position at every step inside the
        # last `mass_matrix_window_fraction` of warmup, and the estimate only
        # engages once enough samples have accumulated (see the docstring
        # above for the gate + rationale). Before engagement, and always when
        # mass_matrix_estimation is None, the kernel and criterion both use
        # jnp.ones(num_dim) -- exactly the pre-existing behavior.
        if estimate_mass_matrix:
            wc_init, wc_update, _ = welford_algorithm(is_diagonal_matrix=True)
            window_start = int(mass_matrix_window_fraction * num_steps)
            mm_engagement_threshold = _mass_matrix_engagement_threshold(num_dim)
        else:
            window_start = num_steps
        in_window_flags = jnp.arange(num_steps) >= window_start

        # Cheap regardless of enable_length_floor (no num_dim-sized arrays
        # here); when the floor is disabled this is built but never read by
        # one_step (mirrors in_window_flags's unconditional construction
        # above).
        if enable_length_floor:
            recompute_flags = (
                jnp.arange(num_steps) % _LENGTH_FLOOR_RECOMPUTE_INTERVAL
            ) == 0
        else:
            recompute_flags = jnp.zeros(num_steps, dtype=bool)

        def _fold_welford_batch(wc_state, batch):
            """Fold a batch of ``num_chains`` samples into ``wc_state`` one
            row at a time -- ``welford_algorithm``'s ``update`` is single-
            sample, so the whole ensemble is pooled via a sequential scan
            rather than reinventing a batched Chan-style merge."""
            new_state, _ = jax.lax.scan(
                lambda carry, x: (wc_update(carry, x), None), wc_state, batch
            )
            return new_state

        def one_step(carry, xs):
            rng_key, in_window, do_recompute = xs
            states, adaptation_state, mm_accum, cov_accum, eig_state = carry

            if estimate_mass_matrix:
                current_imm = _diagonal_mass_matrix_or_fallback(
                    mm_accum, mm_engagement_threshold, num_dim
                )
            else:
                current_imm = jnp.ones(num_dim)

            # Floor applied at CONSUMPTION only: `adaptation_state.
            # trajectory_length` (the value `update` below reads to compute
            # its OWN gradient step) is untouched -- only the length handed
            # to `_step_fn` is floored, so the floor never feeds back into
            # the ChEES optimizer's state.
            if enable_length_floor:
                engaged = mm_accum.sample_size >= mm_engagement_threshold
                consumed_trajectory_length, _ = _apply_length_floor(
                    adaptation_state.trajectory_length,
                    eig_state.lambda_max,
                    engaged,
                    _length_floor,
                    max_leapfrog_steps,
                    adaptation_state.step_size,
                )
            else:
                consumed_trajectory_length = adaptation_state.trajectory_length

            keys = jax.random.split(rng_key, num_chains)
            _step_fn = partial(
                step_fn,
                logdensity_fn=logdensity_fn,
                step_size=adaptation_state.step_size,
                inverse_mass_matrix=current_imm,
                integration_steps_params=(
                    consumed_trajectory_length / adaptation_state.step_size,
                ),
            )
            new_states, info = jax.vmap(_step_fn)(keys, states)
            new_adaptation_state = update(
                adaptation_state,
                info.proposal.position,
                info.proposal.momentum,
                states.position,
                info.acceptance_rate,
                info.is_divergent,
                current_imm,
            )

            if estimate_mass_matrix:
                flat_positions = jax.vmap(
                    lambda p: jax.flatten_util.ravel_pytree(p)[0]
                )(new_states.position)
                new_mm_accum = jax.lax.cond(
                    in_window,
                    lambda acc: _fold_welford_batch(acc, flat_positions),
                    lambda acc: acc,
                    mm_accum,
                )
            else:
                new_mm_accum = mm_accum

            if enable_length_floor:
                # `flat_positions` is always defined here: enable_length_floor
                # implies estimate_mass_matrix (see its derivation above).
                new_cov_accum = jax.lax.cond(
                    in_window,
                    lambda acc: _cov_accumulator_update(acc, flat_positions),
                    lambda acc: acc,
                    cov_accum,
                )
                new_eig_state = jax.lax.cond(
                    jnp.logical_and(in_window, do_recompute),
                    lambda es: _recompute_eig_state(new_cov_accum, current_imm, es),
                    lambda es: es,
                    eig_state,
                )
            else:
                new_cov_accum = cov_accum
                new_eig_state = eig_state

            return (
                new_states,
                new_adaptation_state,
                new_mm_accum,
                new_cov_accum,
                new_eig_state,
            ), adaptation_info_fn(new_states, info, new_adaptation_state)

        batch_init = jax.vmap(
            lambda p: dynamic_hmc.init(p, logdensity_fn, init_random_arg)
        )
        init_states = batch_init(positions)
        init_adaptation_state = init(init_random_arg, step_size)
        init_mm_accum = wc_init(num_dim) if estimate_mass_matrix else None
        init_cov_accum = _cov_accumulator_init(num_dim) if enable_length_floor else None
        init_eig_state = _eig_state_init(num_dim) if enable_length_floor else None

        keys_step = jax.random.split(rng_key, num_steps)
        (
            last_states,
            last_adaptation_state,
            last_mm_accum,
            last_cov_accum,
            last_eig_state,
        ), info = jax.lax.scan(
            one_step,
            (
                init_states,
                init_adaptation_state,
                init_mm_accum,
                init_cov_accum,
                init_eig_state,
            ),
            (keys_step, in_window_flags, recompute_flags),
        )

        final_inverse_mass_matrix = (
            _diagonal_mass_matrix_or_fallback(
                last_mm_accum, mm_engagement_threshold, num_dim
            )
            if estimate_mass_matrix
            else jnp.ones(num_dim)
        )

        # Floor applied at the SECOND consumption point: the final
        # integration_steps_params handed back to the caller. Bit-for-bit
        # identical to the pre-floor computation whenever the floor is
        # disabled (mass_matrix_estimation=None OR _length_floor=False) --
        # see `enable_length_floor`'s derivation above.
        if enable_length_floor:
            final_engaged = last_mm_accum.sample_size >= mm_engagement_threshold
            final_eig_state = _recompute_eig_state(
                last_cov_accum,
                final_inverse_mass_matrix,
                last_eig_state,
                num_iterations=_LENGTH_FLOOR_FINAL_POWER_ITERATIONS,
            )
            trajectory_length_ma = jnp.exp(
                last_adaptation_state.log_trajectory_length_moving_average
            )
            step_size_ma = jnp.exp(last_adaptation_state.log_step_size_moving_average)
            consumed_trajectory_length_ma, floor_clipped_by_cap = _apply_length_floor(
                trajectory_length_ma,
                final_eig_state.lambda_max,
                final_engaged,
                _length_floor,
                max_leapfrog_steps,
                float(step_size_ma),
            )
            num_leapfrog_steps = consumed_trajectory_length_ma / step_size_ma
        else:
            num_leapfrog_steps = jnp.exp(
                last_adaptation_state.log_trajectory_length_moving_average
                - last_adaptation_state.log_step_size_moving_average
            )
            floor_clipped_by_cap = jnp.asarray(False)

        parameters = {
            "step_size": jnp.exp(last_adaptation_state.log_step_size_moving_average),
            "inverse_mass_matrix": final_inverse_mass_matrix,
            "next_random_arg_fn": next_random_arg_fn,
            "integration_steps_fn": integration_steps_fn,
            "integration_steps_params": (num_leapfrog_steps,),
        }

        # Attach floor_clipped_by_cap diagnostic to final info via a transparent
        # wrapper that delegates to the original info but adds the diagnostic flag
        # as an additional attribute. This ensures the flag is accessible without
        # breaking the existing API.
        class _AdaptationInfoWrapper:
            """Transparent wrapper for adaptation info with floor diagnostic flag."""

            def __init__(self, info_obj, floor_flag):
                self._wrapped_info = info_obj
                self.floor_clipped_by_cap = floor_flag

            def __getattr__(self, name):
                return getattr(self._wrapped_info, name)

        enriched_info = _AdaptationInfoWrapper(info, floor_clipped_by_cap)

        return AdaptationResults(last_states, parameters), enriched_info

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]
