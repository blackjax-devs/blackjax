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
"""Implementation of the Pathfinder warmup for the HMC family of sampling algorithms."""
import warnings
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.vi as vi
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.vi.multipathfinder import (
    MultipathfinderState,
    multi_approximate,
    psis_weights,
)

__all__ = ["PathfinderAdaptationState", "base", "pathfinder_adaptation"]


class PathfinderAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState
    step_size: float
    inverse_mass_matrix: Array


def _psis_weighted_mixture_covariance(
    mpf_state: MultipathfinderState,
    log_weights: Array,
) -> Array:
    """Compute the PSIS-weighted mixture covariance of the multipathfinder approximation.

    Implements the law-of-total-variance for a mixture of Laplace approximations:
    ``p(theta) approx sum_i w_i * N(mu_i, Sigma_i)`` where ``w_i`` are aggregate
    PSIS weights.

    The mixture covariance is:

    .. math::

        \\Sigma_{\\text{mix}} = \\underbrace{\\sum_i w_i \\Sigma_i}_{\\text{within}}
            + \\underbrace{\\sum_i w_i (\\mu_i - \\bar{\\mu})(\\mu_i - \\bar{\\mu})^T}_{\\text{between}}

        \\bar{\\mu} = \\sum_i w_i \\mu_i

    Parameters
    ----------
    mpf_state
        Output of :func:`blackjax.vi.multipathfinder.multi_approximate`.
        ``mpf_state.path_states`` are ELBO-argmax-selected states with shape
        ``(n_paths, ...)``.
    log_weights
        Normalised log PSIS weights, shape ``(n_paths * num_samples_per_path,)``,
        as returned by :func:`blackjax.vi.multipathfinder.psis_weights`.

    Returns
    -------
    Array
        Dense ``(d, d)`` positive-definite mixture covariance.

    Notes
    -----
    When ``n_paths=1`` the between-component term is zero and the result
    equals the single path's L-BFGS inverse Hessian exactly.
    """
    path_states = mpf_state.path_states
    # path_states.position: (n_paths, d)
    # path_states.alpha:    (n_paths, d)
    # path_states.beta:     (n_paths, d, 2*maxcor)
    # path_states.gamma:    (n_paths, 2*maxcor, 2*maxcor)
    # logp has shape (n_paths, num_samples_per_path) and is typed as Array.
    num_samples_per_path = mpf_state.logp.shape[1]
    n_paths = log_weights.shape[0] // num_samples_per_path

    # Reshape log_weights from (n_paths * num_samples,) to (n_paths, num_samples)
    log_weights_per_path = log_weights.reshape(n_paths, num_samples_per_path)

    # Aggregate to per-path weights via logsumexp then softmax-normalise.
    # log_path_weights[i] = log sum_j exp(log_weights_per_path[i, j])
    log_path_weights = jax.scipy.special.logsumexp(log_weights_per_path, axis=1)
    # Normalise so sum(w_i) = 1
    log_path_weights_norm = log_path_weights - jax.scipy.special.logsumexp(
        log_path_weights
    )
    w = jnp.exp(log_path_weights_norm)  # (n_paths,)

    # Per-path means: mu_i = path_states.position[i], shape (n_paths, d).
    # PathfinderState.position is stored in the user's *pytree* shape (e.g.
    # ``{'x': (n_paths, d_x), 'y': (n_paths, d_y)}`` for a dict-position
    # model).  Ravel each path's pytree into a flat ``(d,)`` array via
    # ``jax.vmap`` over the leading ``n_paths`` axis.  For models whose
    # position is already a flat ``Array``, this is a no-op (a single leaf
    # raveled to itself); for dict / tuple / nested pytrees it concatenates
    # the flattened leaves in canonical leaf order.
    def _ravel_one_path(pos_pytree):
        flat, _ = ravel_pytree(pos_pytree)
        return flat

    mu_per_path = jax.vmap(_ravel_one_path)(path_states.position)  # (n_paths, d)

    # Per-path covariance Sigma_i = lbfgs_inverse_hessian_formula_1(alpha_i, beta_i, gamma_i)
    # Returns (d, d) per path; vmap over paths gives (n_paths, d, d)
    sigmas = jax.vmap(lbfgs_inverse_hessian_formula_1)(
        path_states.alpha, path_states.beta, path_states.gamma
    )  # (n_paths, d, d)

    # Mixture mean: mu_mix = sum_i w_i mu_i
    mu_mix = jnp.einsum("i,id->d", w, mu_per_path)  # (d,)

    # Within-component term: sum_i w_i Sigma_i
    sigma_within = jnp.einsum("i,ijk->jk", w, sigmas)  # (d, d)

    # Between-component term: sum_i w_i (mu_i - mu_mix)(mu_i - mu_mix)^T
    delta = mu_per_path - mu_mix[None, :]  # (n_paths, d)
    sigma_between = jnp.einsum("i,ij,ik->jk", w, delta, delta)  # (d, d)

    return sigma_within + sigma_between  # (d, d)


def base(
    target_acceptance_rate: float = 0.80,
):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.

    This adaptation runs in two steps:

    1. The Pathfinder algorithm is ran and we subsequently compute an estimate
    for the value of the inverse mass matrix, as well as a new initialization
    point for the markov chain that is supposedly closer to the typical set.
    2. We then start sampling with the MCMC algorithm and use the samples to
    adapt the value of the step size using an optimization algorithm so that
    the mcmc algorithm reaches a given target acceptance rate.

    Parameters
    ----------
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup state.

    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        alpha,
        beta,
        gamma,
        initial_step_size: float,
    ) -> PathfinderAdaptationState:
        """Initialze the adaptation state and parameter values.

        We use the Pathfinder algorithm to compute an estimate of the inverse
        mass matrix that will stay constant throughout the rest of the
        adaptation.

        Parameters
        ----------
        alpha, beta, gamma
            Factored representation of the inverse Hessian computed by the
            Pathfinder algorithm.
        initial_step_size
            The initial value for the step size.

        """
        inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        da_state = da_init(initial_step_size)
        warmup_state = PathfinderAdaptationState(
            da_state, initial_step_size, inverse_mass_matrix
        )

        return warmup_state

    def init_from_imm(
        inverse_mass_matrix: Array,
        initial_step_size: float,
    ) -> PathfinderAdaptationState:
        """Initialize adaptation from a pre-computed inverse mass matrix.

        Used in the multi-chain / multi-path dispatch where the IMM is derived
        analytically or empirically from the multipathfinder output.

        Parameters
        ----------
        inverse_mass_matrix
            Pre-computed dense IMM array, shape ``(d, d)``.
        initial_step_size
            The initial value for the step size.

        """
        da_state = da_init(initial_step_size)
        return PathfinderAdaptationState(
            da_state, initial_step_size, inverse_mass_matrix
        )

    def update(
        adaptation_state: PathfinderAdaptationState,
        position: ArrayLikeTree,
        acceptance_rate: float,
    ) -> PathfinderAdaptationState:
        """Update the adaptation state and parameter values.

        Since the value of the inverse mass matrix is already known we only
        update the state of the step size adaptation algorithm.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last MCMC step.

        Returns
        -------
        The updated states of the chain and the warmup.

        """
        new_ss_state = da_update(adaptation_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return PathfinderAdaptationState(
            new_ss_state, new_step_size, adaptation_state.inverse_mass_matrix
        )

    def final(warmup_state: PathfinderAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and inverse mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, init_from_imm, update, final


def pathfinder_adaptation(
    algorithm,
    logdensity_fn: Callable,
    *,
    num_chains: int = 1,
    n_paths: int | None = None,
    num_samples_per_path: int = 200,
    psis_imm_n_samples: int = 2000,
    imm_estimator: Literal[
        "lbfgs_psis_mixture", "psis_empirical"
    ] = "lbfgs_psis_mixture",
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    adaptation_info_fn: Callable = return_all_adapt_info,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC family.

    Supports single-chain (original behaviour) and multi-chain dispatch via
    `blackjax.vi.multipathfinder`.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of independent chains to initialise.  When 1 (default) the
        behaviour is identical to the original single-chain pathfinder
        adaptation.  When > 1, ``num_chains`` chains are run in parallel with
        per-chain init positions drawn from the PSIS mixture and a shared
        dense ``(d, d)`` IMM.
    n_paths
        Number of independent L-BFGS paths for the multi-path Pathfinder run.
        Defaults to ``num_chains`` (one path per chain).  Ignored when both
        ``num_chains == 1`` and ``n_paths`` is ``None`` (or explicitly 1) ---
        the original single-path code path is used in that case.
    num_samples_per_path
        Number of samples drawn per path to estimate ELBO and PSIS weights.
        Only used when ``effective_n_paths >= 2``.  Default 200.
    psis_imm_n_samples
        Number of PSIS-resampled draws used to estimate the post-PSIS
        empirical covariance for the IMM.  Only used when
        ``imm_estimator="psis_empirical"`` and ``effective_n_paths >= 2``.
        Default 2000.
    imm_estimator
        Method for deriving the inverse mass matrix when ``effective_n_paths >= 2``.

        ``"lbfgs_psis_mixture"`` *(default)*
            Analytic PSIS-weighted mixture covariance via the law of total
            variance (within-component + between-component terms).  Returns a
            dense ``(d, d)`` matrix.  When ``n_paths=1`` this equals the
            single-path L-BFGS inverse Hessian exactly.

        ``"psis_empirical"``
            Empirical covariance estimated from ``psis_imm_n_samples`` draws
            importance-resampled according to PSIS weights.  Returns a dense
            ``(d, d)`` matrix.

        Ignored for the single-chain single-path dispatch (``num_chains=1``,
        ``effective_n_paths=1``).  Passing a non-default value in that case
        raises a ``UserWarning``.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the
    tuned parameter values from an initial state.

    Notes
    -----
    **Dispatch table** --- the ``(num_chains, effective_n_paths)`` combination
    selects the internal code path:

    +-----------+--------------------+--------------------+---------------------------------------------+
    | num_chains| effective_n_paths  | imm_estimator      | Behaviour                                   |
    +===========+====================+====================+=============================================+
    | 1         | 1                  | (ignored)          | **Original** single-path Pathfinder + DA.   |
    |           |                    |                    | scalar step_size, ``(d, d)`` IMM.           |
    +-----------+--------------------+--------------------+---------------------------------------------+
    | 1         | >= 2               | lbfgs_psis_mixture | Multipathfinder; analytic PSIS-weighted     |
    |           |                    |                    | mixture covariance; ``(d, d)`` dense IMM.   |
    +-----------+--------------------+--------------------+---------------------------------------------+
    | 1         | >= 2               | psis_empirical     | Multipathfinder; empirical covariance from  |
    |           |                    |                    | psis_imm_n_samples resampled draws;         |
    |           |                    |                    | ``(d, d)`` dense IMM.                       |
    +-----------+--------------------+--------------------+---------------------------------------------+
    | >= 2      | 1                  | (ignored)          | Single-path Pathfinder; broadcast init x    |
    |           |                    |                    | chains; vmap DA. step_size (num_chains,).   |
    |           |                    |                    | ``(d, d)`` IMM from L-BFGS inverse Hessian. |
    +-----------+--------------------+--------------------+---------------------------------------------+
    | >= 2      | >= 2               | lbfgs_psis_mixture | **Paper-canonical** (Zhang et al. 2022).    |
    |           |                    |                    | Multipathfinder -> PSIS init -> analytic    |
    |           |                    |                    | mixture covariance -> vmap DA.              |
    +-----------+--------------------+--------------------+---------------------------------------------+
    | >= 2      | >= 2               | psis_empirical     | Multipathfinder -> PSIS init -> empirical   |
    |           |                    |                    | covariance -> vmap DA.                      |
    +-----------+--------------------+--------------------+---------------------------------------------+

    **Return contract:**

    * ``num_chains == 1``: identical to the pre-change API.
      ``parameters["step_size"]`` is a scalar;
      ``parameters["inverse_mass_matrix"]`` is uniformly ``(d, d)``.
    * ``num_chains > 1``: ``parameters["step_size"]`` is ``(num_chains,)``;
      ``parameters["inverse_mass_matrix"]`` is the single shared ``(d, d)`` IMM
      (not broadcast --- the caller broadcasts if needed).
    * When ``effective_n_paths >= 2``, ``parameters`` also includes
      ``"_pathfinder_psis_pareto_k"`` (scalar) for downstream diagnostics.

    **PSIS-weighted mixture covariance math** (``imm_estimator="lbfgs_psis_mixture"``):

    Multipathfinder produces a mixture of Laplace approximations
    ``p(theta) approx sum_i w_i * N(mu_i, Sigma_i)`` where ``mu_i`` is path ``i``'s
    L-BFGS optimum position and ``Sigma_i`` is its L-BFGS inverse-Hessian estimate.
    Weights ``w_i`` are aggregate (per-path sum) PSIS weights.

    By the law of total variance:

    .. math::

        \\Sigma_{\\text{mix}} = \\sum_i w_i \\Sigma_i
            + \\sum_i w_i (\\mu_i - \\bar\\mu)(\\mu_i - \\bar\\mu)^T

    where :math:`\\bar\\mu = \\sum_i w_i \\mu_i`.
    """
    # --- Input validation (before JIT) ---
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}")
    if n_paths is not None and n_paths < 1:
        raise ValueError(f"n_paths must be >= 1 or None, got {n_paths}")

    effective_n_paths = n_paths if n_paths is not None else num_chains

    # Warn if imm_estimator is set to non-default on single-path single-chain.
    if (
        num_chains == 1
        and effective_n_paths == 1
        and imm_estimator != "lbfgs_psis_mixture"
    ):
        warnings.warn(
            f"imm_estimator={imm_estimator!r} is ignored for the single-chain "
            f"single-path dispatch (num_chains=1, effective_n_paths=1). "
            f"The single-path L-BFGS inverse Hessian is always used.",
            UserWarning,
            stacklevel=2,
        )

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_init_from_imm, adapt_update, adapt_final = base(
        target_acceptance_rate,
    )

    def one_step(carry, rng_key):
        state, adaptation_state = carry
        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_update(
            adaptation_state, new_state.position, info.acceptance_rate
        )
        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 400):
        """Run pathfinder adaptation.

        Parameters
        ----------
        rng_key
            PRNG key.
        position
            Single-chain position pytree.  The runner replicates internally
            when ``num_chains > 1`` or ``effective_n_paths > 1``.
        num_steps
            Number of DA adaptation steps per chain.  Default 400.

        Returns
        -------
        An ``AdaptationResults`` object and the adaptation info.
        """
        if num_chains == 1 and effective_n_paths == 1:
            # ----------------------------------------------------------------
            # PATH A: original single-chain single-path behaviour
            # ----------------------------------------------------------------
            return _run_single(rng_key, position, num_steps)
        elif effective_n_paths == 1:
            # ----------------------------------------------------------------
            # PATH B: num_chains > 1 but single pathfinder fit
            # Broadcast single-path L-BFGS IMM to all chains, vmap DA.
            # ----------------------------------------------------------------
            return _run_multichain_single_path(rng_key, position, num_steps)
        else:
            # ----------------------------------------------------------------
            # PATH C: multi-path (multipathfinder) --- covers num_chains 1 or >1
            # ----------------------------------------------------------------
            return _run_multipathfinder(rng_key, position, num_steps)

    def _run_single(rng_key, position, num_steps):
        """Original single-chain single-path Pathfinder + DA."""
        init_key, sample_key, rng_key = jax.random.split(rng_key, 3)

        pathfinder_state, _ = vi.pathfinder.approximate(
            init_key, logdensity_fn, position
        )
        init_warmup_state = adapt_init(
            pathfinder_state.alpha,
            pathfinder_state.beta,
            pathfinder_state.gamma,
            initial_step_size,
        )

        init_position, _ = vi.pathfinder.sample(sample_key, pathfinder_state)
        init_state = algorithm.init(init_position, logdensity_fn)

        keys = jax.random.split(rng_key, num_steps)
        last_state, info = jax.lax.scan(
            one_step,
            (init_state, init_warmup_state),
            keys,
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return AdaptationResults(last_chain_state, parameters), info

    def _run_multichain_single_path(rng_key, position, num_steps):
        """Single Pathfinder fit; broadcast init and IMM to num_chains; vmap DA."""
        pf_key, rng_key = jax.random.split(rng_key)
        pathfinder_state, _ = vi.pathfinder.approximate(pf_key, logdensity_fn, position)
        # Derive the L-BFGS-based IMM from the single Pathfinder fit.
        # lbfgs_inverse_hessian_formula_1 returns a dense (d, d) matrix.
        shared_imm = lbfgs_inverse_hessian_formula_1(
            pathfinder_state.alpha,
            pathfinder_state.beta,
            pathfinder_state.gamma,
        )  # (d, d)

        # Sample num_chains init positions from the single Pathfinder fit.
        sample_keys = jax.random.split(rng_key, num_chains)

        @jax.vmap
        def sample_one(key):
            pos, _ = vi.pathfinder.sample(key, pathfinder_state)
            return pos

        init_positions = sample_one(sample_keys)

        # Initialize chains.
        @jax.vmap
        def init_chain(pos):
            init_state = algorithm.init(pos, logdensity_fn)
            init_adapt = adapt_init_from_imm(shared_imm, initial_step_size)
            return init_state, init_adapt

        init_states, init_adapt_states = init_chain(init_positions)

        def one_step_chain(carry, rng_key):
            state, adaptation_state = carry
            new_state, info = mcmc_kernel(
                rng_key,
                state,
                logdensity_fn,
                adaptation_state.step_size,
                adaptation_state.inverse_mass_matrix,
                **extra_parameters,
            )
            new_adaptation_state = adapt_update(
                adaptation_state, new_state.position, info.acceptance_rate
            )
            return (
                (new_state, new_adaptation_state),
                adaptation_info_fn(new_state, info, new_adaptation_state),
            )

        chain_keys = jax.random.split(rng_key, num_chains)

        @jax.vmap
        def run_one_chain(init_state_and_adapt, chain_rng_key):
            init_state, init_adapt = init_state_and_adapt
            keys_for_steps = jax.random.split(chain_rng_key, num_steps)
            last_state, info = jax.lax.scan(
                one_step_chain,
                (init_state, init_adapt),
                keys_for_steps,
            )
            last_chain_state, last_warmup_state = last_state
            step_size, _ = adapt_final(last_warmup_state)
            return last_chain_state, step_size, info

        last_chain_states, step_sizes, infos = run_one_chain(
            (init_states, init_adapt_states), chain_keys
        )

        parameters = {
            "step_size": step_sizes,  # (num_chains,)
            "inverse_mass_matrix": shared_imm,  # (d, d) shared
            **extra_parameters,
        }
        return AdaptationResults(last_chain_states, parameters), infos

    def _run_multipathfinder(rng_key, position, num_steps):
        """Multi-path Pathfinder -> PSIS -> IMM -> DA (single or multi-chain)."""
        pf_key, resample_key, imm_key, rng_key = jax.random.split(rng_key, 4)

        # Replicate single position to (n_paths, ...) for the multi-path fit.
        init_positions = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None], (effective_n_paths,) + x.shape),
            position,
        )

        # Run multi-path Pathfinder.
        mpf_state, _ = multi_approximate(
            pf_key,
            logdensity_fn,
            init_positions,
            num_samples=num_samples_per_path,
        )

        # Compute PSIS weights.
        log_weights, pareto_k = psis_weights(mpf_state)  # (n_paths*num_samples,)

        # ----- Derive shared dense (d, d) IMM -----
        if imm_estimator == "lbfgs_psis_mixture":
            # Analytic PSIS-weighted mixture covariance (law of total variance).
            shared_imm = _psis_weighted_mixture_covariance(mpf_state, log_weights)
        else:
            # psis_empirical: empirical covariance from PSIS-resampled draws.
            samples_flat = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), mpf_state.samples
            )
            total_pool = log_weights.shape[0]
            probs = jnp.exp(log_weights)
            imm_indices = jax.random.choice(
                imm_key, total_pool, shape=(psis_imm_n_samples,), replace=True, p=probs
            )
            imm_samples_pytree = jax.tree.map(lambda x: x[imm_indices], samples_flat)
            # Flatten each sample to (d,) for covariance estimation.
            flat_imm_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(
                imm_samples_pytree
            )
            # (psis_imm_n_samples, d) -> dense (d, d) covariance
            shared_imm = jnp.cov(flat_imm_samples.T)  # (d, d)

        # ----- Resample init positions via PSIS -----
        samples_flat_for_init = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), mpf_state.samples
        )
        total_pool = log_weights.shape[0]
        probs = jnp.exp(log_weights)
        n_resample = max(num_chains, 1)
        init_indices = jax.random.choice(
            resample_key, total_pool, shape=(n_resample,), replace=True, p=probs
        )
        init_from_psis = jax.tree.map(lambda x: x[init_indices], samples_flat_for_init)
        # init_from_psis: pytree with leading dim n_resample

        if num_chains == 1:
            # ----------------------------------------------------------------
            # Single-chain multi-path: pick first (only) position, run DA.
            # ----------------------------------------------------------------
            single_pos = jax.tree.map(lambda x: x[0], init_from_psis)
            init_state = algorithm.init(single_pos, logdensity_fn)
            init_adapt = adapt_init_from_imm(shared_imm, initial_step_size)

            chain_rng_key = rng_key
            keys = jax.random.split(chain_rng_key, num_steps)
            last_state, info = jax.lax.scan(
                one_step,
                (init_state, init_adapt),
                keys,
            )
            last_chain_state, last_warmup_state = last_state
            step_size, _ = adapt_final(last_warmup_state)
            parameters = {
                "step_size": step_size,
                "inverse_mass_matrix": shared_imm,
                "_pathfinder_psis_pareto_k": pareto_k,
                **extra_parameters,
            }
            return AdaptationResults(last_chain_state, parameters), info

        else:
            # ----------------------------------------------------------------
            # Multi-chain multi-path: vmap DA over num_chains chains.
            # ----------------------------------------------------------------
            @jax.vmap
            def init_chain(pos):
                init_state = algorithm.init(pos, logdensity_fn)
                init_adapt = adapt_init_from_imm(shared_imm, initial_step_size)
                return init_state, init_adapt

            init_states, init_adapt_states = init_chain(init_from_psis)

            def one_step_chain(carry, rng_key_step):
                state, adaptation_state = carry
                new_state, info = mcmc_kernel(
                    rng_key_step,
                    state,
                    logdensity_fn,
                    adaptation_state.step_size,
                    adaptation_state.inverse_mass_matrix,
                    **extra_parameters,
                )
                new_adaptation_state = adapt_update(
                    adaptation_state, new_state.position, info.acceptance_rate
                )
                return (
                    (new_state, new_adaptation_state),
                    adaptation_info_fn(new_state, info, new_adaptation_state),
                )

            chain_keys = jax.random.split(rng_key, num_chains)

            @jax.vmap
            def run_one_chain(init_state_and_adapt, chain_rng_key):
                init_state, init_adapt = init_state_and_adapt
                keys_for_steps = jax.random.split(chain_rng_key, num_steps)
                last_state, info = jax.lax.scan(
                    one_step_chain,
                    (init_state, init_adapt),
                    keys_for_steps,
                )
                last_chain_state, last_warmup_state = last_state
                step_size, _ = adapt_final(last_warmup_state)
                return last_chain_state, step_size, info

            last_chain_states, step_sizes, infos = run_one_chain(
                (init_states, init_adapt_states), chain_keys
            )
            parameters = {
                "step_size": step_sizes,  # (num_chains,)
                "inverse_mass_matrix": shared_imm,  # (d, d) shared
                "_pathfinder_psis_pareto_k": pareto_k,
                **extra_parameters,
            }
            return AdaptationResults(last_chain_states, parameters), infos

    return AdaptationAlgorithm(run)
