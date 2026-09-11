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
"""Implementation of the Stan warmup for the HMC family of sampling algorithms.

The public surface of this module is unchanged.  Internally, :func:`window_adaptation`
is now a thin compatibility shim over :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`;
:func:`build_schedule` is defined in :mod:`blackjax.adaptation.staged_adaptation` and
re-exported here for backward compatibility.

:data:`WindowAdaptationState` is an alias for
:class:`~blackjax.adaptation.staged_adaptation.StagedAdaptationState`; both names
refer to the same class object so ``isinstance`` checks using either name continue
to work without modification.

The :func:`base` function is retained at its released API for downstream code that
calls it directly.  It is not exercised by the :func:`window_adaptation` shim (which
delegates to :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`).
Fisher-diagonal adaptation is accessible via
``staged_adaptation(metric="fisher_diag")`` only.
"""
import warnings
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import return_all_adapt_info
from blackjax.adaptation.mass_matrix import mass_matrix_adaptation
from blackjax.adaptation.metric_recipes import lookup_recipe
from blackjax.adaptation.staged_adaptation import (
    build_schedule,  # canonical definition in staged_adaptation; re-exported here
)
from blackjax.adaptation.staged_adaptation import (
    StagedAdaptationState,
    staged_adaptation,
)
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree
from blackjax.util import pytree_size

__all__ = ["WindowAdaptationState", "base", "build_schedule", "window_adaptation"]

# WindowAdaptationState is the canonical name for StagedAdaptationState in this
# module.  They are the SAME class object: isinstance(x, WindowAdaptationState)
# is identical to isinstance(x, StagedAdaptationState).
WindowAdaptationState = StagedAdaptationState


def base(
    is_mass_matrix_diagonal: bool,
    target_acceptance_rate: float = 0.80,
    initial_inverse_mass_matrix: Array | None = None,
    imm_shrinkage_to_previous: float = 0.0,
) -> tuple[Callable, Callable, Callable]:
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.
    The schedule and algorithms used match Stan's :cite:p:`stan_hmc_param` as closely as possible.

    Unlike several other libraries, we separate the warmup and sampling phases
    explicitly. This ensure a better modularity; a change in the warmup does
    not affect the sampling. It also allows users to run their own warmup
    should they want to.
    We also decouple generating a new sample with the mcmc algorithm and
    updating the values of the parameters.

    Stan's warmup consists in the three following phases:

    1. A fast adaptation window where only the step size is adapted using
    Nesterov's dual averaging scheme to match a target acceptance rate.
    2. A succession of slow adapation windows (where the size of a window is
    double that of the previous window) where both the mass matrix and the step
    size are adapted. The mass matrix is recomputed at the end of each window;
    the step size is re-initialized to a "reasonable" value.
    3. A last fast adaptation window where only the step size is adapted.

    Schematically:

    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    |1        |2  |3     |3           |3                       |3     |
    +---------+---+------+------------+------------------------+------+

    Step (1) consists in find a "reasonable" first step size that is used to
    initialize the dual averaging scheme. In (2) we initialize the mass matrix
    to the matrix. In (3) we compute the mass matrix to use in the kernel and
    re-initialize the mass matrix adaptation. The step size is still adapated
    in slow adaptation windows, and is not re-initialized between windows.

    Parameters
    ----------
    is_mass_matrix_diagonal
        Create and adapt a diagonal mass matrix if True, a dense matrix
        otherwise.
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.
    initial_inverse_mass_matrix
        Optional seed value for the inverse mass matrix passed through to
        ``mass_matrix_adaptation``.  ``None`` (default) uses the standard
        identity initialisation.
    imm_shrinkage_to_previous
        Pseudo-count controlling shrinkage of the IMM toward the previous
        window's IMM. Default 0.0 gives the current Stan behavior. Passed
        through to ``mass_matrix_adaptation``.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup
        state.

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use :func:`blackjax.window_adaptation` for the standard warmup, or
        :func:`blackjax.staged_adaptation` for custom metric recipes.

    """
    warnings.warn(
        "window_adaptation.base() is deprecated and will be removed in a future "
        "release. Use blackjax.window_adaptation for the standard warmup, or "
        "blackjax.staged_adaptation for custom metric recipes.",
        DeprecationWarning,
        stacklevel=2,
    )

    mm_init, mm_update, mm_final = mass_matrix_adaptation(
        is_mass_matrix_diagonal, imm_shrinkage_to_previous
    )
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        position: ArrayLikeTree, initial_step_size: float
    ) -> WindowAdaptationState:
        """Initialze the adaptation state and parameter values.

        Unlike the original Stan window adaptation we do not use the
        `find_reasonable_step_size` algorithm which we found to be unnecessary.
        We may reconsider this choice in the future.

        """
        num_dimensions = pytree_size(position)
        imm_state = mm_init(num_dimensions, initial_inverse_mass_matrix)

        ss_state = da_init(initial_step_size)

        return WindowAdaptationState(
            ss_state,
            imm_state,
            initial_step_size,
            imm_state.inverse_mass_matrix,
        )

    def fast_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
    ) -> WindowAdaptationState:
        """Update the adaptation state when in a "fast" window.

        Only the step size is adapted in fast windows. "Fast" refers to the fact
        that the optimization algorithms are relatively fast to converge
        compared to the covariance estimation with Welford's algorithm

        """
        del position

        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state,
            warmup_state.imm_state,
            new_step_size,
            warmup_state.inverse_mass_matrix,
        )

    def slow_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
    ) -> WindowAdaptationState:
        """Update the adaptation state when in a "slow" window.

        Both the mass matrix adaptation *state* and the step size state are
        adapted in slow windows. The value of the step size is updated as well,
        but the new value of the inverse mass matrix is only computed at the end
        of the slow window. "Slow" refers to the fact that we need many samples
        to get a reliable estimation of the covariance matrix used to update the
        value of the mass matrix.

        """
        new_imm_state = mm_update(warmup_state.imm_state, position)
        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state, new_imm_state, new_step_size, warmup_state.inverse_mass_matrix
        )

    def slow_final(warmup_state: WindowAdaptationState) -> WindowAdaptationState:
        """Update the parameters at the end of a slow adaptation window.

        We compute the value of the mass matrix and reset the mass matrix
        adapation's internal state since middle windows are "memoryless".

        """
        new_imm_state = mm_final(warmup_state.imm_state)
        new_ss_state = da_init(da_final(warmup_state.ss_state))
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state,
            new_imm_state,
            new_step_size,
            new_imm_state.inverse_mass_matrix,
        )

    def update(
        adaptation_state: WindowAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        acceptance_rate: float,
    ) -> WindowAdaptationState:
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last mcmc step.

        Returns
        -------
        The updated adaptation state.

        """
        stage, is_middle_window_end = adaptation_stage

        warmup_state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            acceptance_rate,
            adaptation_state,
        )

        warmup_state = jax.lax.cond(
            is_middle_window_end,
            slow_final,
            lambda x: x,
            warmup_state,
        )

        return warmup_state

    def final(warmup_state: WindowAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.imm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


def _pick_recipe_name(*, is_mass_matrix_diagonal: bool) -> str:
    """Map the is_mass_matrix_diagonal flag to a metric recipe registry name.
    Used by the :func:`window_adaptation` shim."""
    if is_mass_matrix_diagonal:
        return "welford_diag"
    else:
        return "welford_dense"


def window_adaptation(
    algorithm,
    logdensity_fn: Callable,
    is_mass_matrix_diagonal: bool = True,
    initial_inverse_mass_matrix: Array | None = None,
    imm_shrinkage_to_previous: float = 0.0,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily. See Blackjax.hmc_family

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_inverse_mass_matrix
        Optional seed value for the inverse mass matrix used at the start of
        warmup.  When ``None`` (default) the standard identity initialisation
        is used (``ones(d)`` for diagonal, ``identity(d)`` for dense).  When
        provided the array seeds the first window's step-size adaptation with a
        better geometric hint; the Welford algorithm still starts from scratch
        so the seed is gradually overwritten by the empirical covariance.

        Shape must be consistent with ``is_mass_matrix_diagonal``:

        * diagonal (``is_mass_matrix_diagonal=True``): 1-D array of shape
          ``(d,)`` where ``d`` is the number of model parameters.
        * dense (``is_mass_matrix_diagonal=False``): 2-D square array of shape
          ``(d, d)``.

        A ``ValueError`` is raised at construction time (before any JIT
        tracing) if the shape is inconsistent.
    imm_shrinkage_to_previous
        Bayesian pseudo-count controlling shrinkage of the per-window
        adapted inverse mass matrix toward the *previous* window's IMM, in
        addition to the existing Stan-style shrinkage toward
        ``1e-3 · I`` (pseudo-count 5). Default ``0.0`` reproduces Stan's
        behavior exactly: each window's Welford estimate replaces the
        previous IMM (no persistence). A positive value blends a fraction
        ``k_prev / (count + 5 + k_prev)`` of the previous IMM into the
        new one, where ``count`` is the number of samples in the window
        and ``k_prev`` is this argument.

        Useful when ``initial_inverse_mass_matrix`` carries high-confidence
        information (e.g., from a converged pre-warmup Pathfinder fit) that
        should persist beyond window 1's reset. Practical band for typical
        Stan window sizes (25–500): ``5 ≤ k_prev ≤ 50`` gives mild-to-
        moderate persistence; ``k_prev ≈ window_size`` gives balanced 50/50
        weight between the previous IMM and the new window's data;
        ``k_prev >> window_size`` effectively freezes the IMM at
        ``initial_inverse_mass_matrix`` (anti-pattern unless the seed is
        truly known-correct). See ``mass_matrix_adaptation`` for the full
        precision-weighted-average formula.

        Validated at construction time — negative values raise
        ``ValueError`` before any JIT tracing.
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
    A function that runs the adaptation and returns an `AdaptationResult` object.

    Notes
    -----
    This function is a thin compatibility shim over
    :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`.  The
    public interface and return type are frozen; no breaking changes will be
    made in this module.

    Wrap ``warmup.run(...)`` in :func:`blackjax.progress_bar` to display a
    progress bar, e.g. ``with blackjax.progress_bar(): warmup.run(...)``.

    """
    # Validate initial_inverse_mass_matrix shape against is_mass_matrix_diagonal.
    # Do this BEFORE any JIT-traced path so the user gets a clear Python error.
    if initial_inverse_mass_matrix is not None:
        imm = jnp.asarray(initial_inverse_mass_matrix)
        if is_mass_matrix_diagonal:
            if imm.ndim != 1:
                raise ValueError(
                    f"is_mass_matrix_diagonal=True requires "
                    f"initial_inverse_mass_matrix.ndim == 1, got ndim={imm.ndim}"
                )
        else:
            if imm.ndim != 2 or imm.shape[0] != imm.shape[1]:
                raise ValueError(
                    f"is_mass_matrix_diagonal=False requires "
                    f"initial_inverse_mass_matrix to be a 2-D square array, "
                    f"got shape={imm.shape}"
                )

    # Validate imm_shrinkage_to_previous before any JIT-traced path.
    if imm_shrinkage_to_previous < 0.0:
        raise ValueError(
            f"imm_shrinkage_to_previous must be >= 0.0, "
            f"got {imm_shrinkage_to_previous}"
        )

    # Map the old parameter names to a registered MetricRecipe and build
    # a MetricCore (pre-builds the core so staged_adaptation sees a MetricCore
    # directly and skips the lookup step).
    recipe_name = _pick_recipe_name(
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
    )
    metric_core = lookup_recipe(recipe_name).build_core(
        imm_shrinkage_to_previous=imm_shrinkage_to_previous,
        initial_inverse_mass_matrix=initial_inverse_mass_matrix,
    )

    return staged_adaptation(
        algorithm,
        logdensity_fn,
        metric=metric_core,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        adaptation_info_fn=adaptation_info_fn,
        integrator=integrator,
        **extra_parameters,
    )
