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
"""Staged warmup adaptation engine for HMC-family algorithms.

This module provides the :func:`staged_adaptation` engine and the
:func:`build_schedule` function (previously in ``window_adaptation.py``;
re-exported from there for backward compatibility).

:func:`staged_adaptation` adapts step size and inverse mass matrix via the
Stan warmup schedule for any algorithm whose kernel has signature::

    kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, **extra)

Supported: :data:`blackjax.nuts`, :data:`blackjax.hmc`, :data:`blackjax.mhmc`,
:data:`blackjax.barker`, and others accepting the above contract.
Excluded: RMHMC (kernel takes ``mass_matrix: Metric``, not ``inverse_mass_matrix``);
GHMC/MEADS (kernel lacks ``inverse_mass_matrix``); MCLMC (has own warmup);
dynamic_hmc (init requires ``random_generator_arg``).  ``WindowAdaptationState`` in
:mod:`~blackjax.adaptation.window_adaptation` is an alias for
:class:`StagedAdaptationState` (same class object).

Notes
-----
``build_schedule`` is defined here (canonical location) and re-exported from
``window_adaptation`` for backward compatibility.  Import from either module.
"""
import inspect
import warnings
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.metric_recipes import MetricCore, MetricRecipe, lookup_recipe
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = [
    "StagedAdaptationState",
    "build_schedule",
    "staged_adaptation",
]


# ---------------------------------------------------------------------------
# State type (canonical definition; aliased as WindowAdaptationState in
# window_adaptation.py)
# ---------------------------------------------------------------------------


class StagedAdaptationState(NamedTuple):
    """Scan-carry state for the staged adaptation engine.

    Field names intentionally mirror the previous
    ``WindowAdaptationState`` fields so that any downstream code accessing
    adaptation info by field name (``.ss_state``, ``.imm_state``, …)
    continues to work without modification.

    ``WindowAdaptationState`` in :mod:`blackjax.adaptation.window_adaptation`
    is an alias of this type (``WindowAdaptationState = StagedAdaptationState``);
    both names refer to the same NamedTuple class object, so ``isinstance``
    checks using either name are equivalent.

    Parameters
    ----------
    ss_state
        Current state of the dual-averaging step-size adaptation.
    imm_state
        Current mass-matrix adaptation core state.  One of
        :class:`~blackjax.adaptation.mass_matrix.MassMatrixAdaptationState`
        or :class:`~blackjax.adaptation.mass_matrix.FisherMassMatrixAdaptationState`.
        Typed ``Any`` here to avoid a hard dependency on the concrete types;
        the MetricCore protocol guarantees the right type at construction time.
    step_size
        Current (exponential-space) step-size estimate; read by the MCMC kernel
        at every scan step.
    inverse_mass_matrix
        Current inverse mass matrix; updated at each slow-window boundary and
        read by the MCMC kernel at every scan step.
    """

    ss_state: DualAveragingAdaptationState
    imm_state: Any  # MassMatrixAdaptationState | FisherMassMatrixAdaptationState
    step_size: float
    inverse_mass_matrix: Array


# ---------------------------------------------------------------------------
# Engine internals
# ---------------------------------------------------------------------------


def _make_engine(
    metric_core: MetricCore,
    *,
    target_acceptance_rate: float,
    n_da_updates: int = 1,
) -> tuple[Callable, Callable, Callable]:
    """Build the (init, update, final) triple for the staged adaptation HOST.

    This is the low-level engine constructor.  Step-size dual averaging and
    the stage-dispatch logic live here; mass-matrix adaptation is fully
    delegated to ``metric_core``.

    Parameters
    ----------
    metric_core
        Pre-built :class:`~blackjax.adaptation.metric_recipes.MetricCore`
        (init/update/final bundle for the inverse mass matrix).
    target_acceptance_rate
        Target acceptance rate for dual-averaging step-size adaptation.
    n_da_updates
        When ``1`` (single-chain), calls ``da_update`` once with the scalar
        acceptance rate.  When ``> 1`` (multi-chain shared-ε), calls
        ``da_update`` once on ``jnp.mean(acceptance_rates)``: M chains stepping
        at ONE shared ε produce M measurements of the SAME acceptance quantity,
        so the correct observation model is one mean observation (not M
        sequential ones).  Sequencing M updates inflates the DA primal gain
        ~√M and advances the Polyak schedule M× too fast, causing a warmup
        limit cycle.  Set to ``n_chains`` for the multi-chain path (the caller
        in :func:`staged_adaptation` does this).  Defaults to ``1``.

    Returns
    -------
    init
        ``(position, initial_step_size) -> StagedAdaptationState``
    update
        ``(adaptation_state, adaptation_stage, position, grad, acceptance_rate)
        -> StagedAdaptationState``
    final
        ``(warmup_state) -> (step_size, inverse_mass_matrix)``
    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def _maybe_multi_da_update(
        ss_state: DualAveragingAdaptationState,
        acceptance_rates: Array,
    ) -> DualAveragingAdaptationState:
        """Apply one DA update using the mean acceptance rate.

        When ``n_da_updates == 1`` (single-chain), calls ``da_update`` with
        the scalar ``acceptance_rates`` unchanged.  When ``n_da_updates > 1``
        (multi-chain shared-ε), calls ``da_update`` once on
        ``jnp.mean(acceptance_rates)`` — the correct observation model for M
        chains stepping at one shared ε.  The previous sequential scan
        (M updates, one per chain) inflated the DA gain ~√M and caused
        self-sustained limit cycles; the mean-pool form treats the M
        measurements as one mean observation with M-fold reduced variance.
        """
        if n_da_updates == 1:
            return da_update(ss_state, acceptance_rates)

        return da_update(ss_state, jnp.mean(acceptance_rates))

    def init(
        position: ArrayLikeTree, initial_step_size: float
    ) -> StagedAdaptationState:
        n_dims = pytree_size(position)
        imm_state = metric_core.init(n_dims)
        ss_state = da_init(initial_step_size)
        return StagedAdaptationState(
            ss_state,
            imm_state,
            initial_step_size,
            imm_state.inverse_mass_matrix,
        )

    def fast_update(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: Array,
        ws: StagedAdaptationState,
    ) -> StagedAdaptationState:
        """Update adaptation state during a fast (step-size-only) window."""
        del position, grad
        new_ss = _maybe_multi_da_update(ws.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss.log_step_size)
        return StagedAdaptationState(
            new_ss, ws.imm_state, new_step_size, ws.inverse_mass_matrix
        )

    def slow_update(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: Array,
        ws: StagedAdaptationState,
    ) -> StagedAdaptationState:
        """Update adaptation state during a slow (step-size + mass-matrix) window.

        Delegates mass-matrix accumulation to ``metric_core.update``.  For
        welford-path cores, ``grad`` is accepted for interface uniformity but
        is not consumed by the underlying estimator (the ``isinstance`` dispatch
        inside ``mass_matrix_adaptation.update`` resolves at JAX trace time).
        For fisher-path cores, ``grad`` is forwarded to the Fisher-block
        accumulator.
        """
        new_metric_st = metric_core.update(ws.imm_state, position, grad)
        new_ss = _maybe_multi_da_update(ws.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss.log_step_size)
        # Propagate the core's current inverse_mass_matrix to the MCMC kernel
        # at every slow step, not just at window-end.  For all non-accumulating
        # cores (welford, fisher-diag, reset-low-rank, sample-cov) update() does
        # not write inverse_mass_matrix, so new_metric_st.inverse_mass_matrix ==
        # ws.inverse_mass_matrix and this is bit-identical to the previous
        # ws.inverse_mass_matrix return.  For the accumulating core, update() may
        # recompute inverse_mass_matrix mid-window (when recompute_counter %
        # recompute_every == 0), and this line surfaces that to MCMC — matching
        # the legacy base() accumulating scan loop's slow_recompute_only() which
        # also updated adaptation_state.sigma/U/lam immediately.  Consistent with
        # slow_final(), which already reads new_metric_st.inverse_mass_matrix.
        return StagedAdaptationState(
            new_ss, new_metric_st, new_step_size, new_metric_st.inverse_mass_matrix
        )

    def slow_final(ws: StagedAdaptationState) -> StagedAdaptationState:
        """Finalize a slow window: recompute IMM and re-initialise step-size DA.

        Delegates IMM computation and window-buffer reset to
        ``metric_core.final``.  The new inverse mass matrix is read from
        ``new_metric_st.inverse_mass_matrix`` and stored in the returned state
        for the MCMC kernel to use in the next window.
        """
        new_metric_st = metric_core.final(ws.imm_state)
        new_ss = da_init(da_final(ws.ss_state))
        new_step_size = jnp.exp(new_ss.log_step_size)
        return StagedAdaptationState(
            new_ss,
            new_metric_st,
            new_step_size,
            new_metric_st.inverse_mass_matrix,
        )

    def update(
        adaptation_state: StagedAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: float,
    ) -> StagedAdaptationState:
        """Dispatch one warmup step to the correct fast/slow update.

        Parameters
        ----------
        adaptation_state
            Current warmup state (the scan carry).
        adaptation_stage
            ``(stage_label, is_middle_window_end)`` pair from the schedule.
            ``stage_label`` is ``0`` (fast) or ``1`` (slow).
            ``is_middle_window_end`` is ``True`` on the last step of a slow
            window, triggering ``slow_final``.
        position
            Current MCMC position.
        grad
            Log-density gradient at ``position``.
        acceptance_rate
            Metropolis acceptance rate from the last MCMC step.

        Returns
        -------
        StagedAdaptationState
            Updated warmup state.
        """
        stage, is_middle_window_end = adaptation_stage

        ws = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            grad,
            acceptance_rate,
            adaptation_state,
        )

        ws = jax.lax.cond(
            is_middle_window_end,
            slow_final,
            lambda x: x,
            ws,
        )

        return ws

    def final(ws: StagedAdaptationState) -> tuple[float, Array]:
        """Return the final step size and inverse mass matrix after warmup."""
        step_size = jnp.exp(ws.ss_state.log_step_size_avg)
        inverse_mass_matrix = ws.imm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


# ---------------------------------------------------------------------------
# Build schedule (canonical location; re-exported from window_adaptation.py)
# ---------------------------------------------------------------------------


def build_schedule(
    num_steps: int,
    initial_buffer_size: int = 75,
    final_buffer_size: int = 50,
    first_window_size: int = 25,
) -> list[tuple[int, bool]]:
    """Return the schedule for Stan's warmup.

    The schedule below is intended to be as close as possible to Stan's :cite:p:`stan_hmc_param`.
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set. Only the step size is
    adapted in this window.
    2. "Slow" parameters that require global information (typically covariance)
    are estimated in a series of expanding intervals with no memory; the step
    size is re-initialized at the end of each window. Each window is twice the
    size of the preceding window.
    3. A final fast interval during which the step size is adapted using the
    computed mass matrix.

    Schematically:

    ```
    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    ```

    The distinction slow/fast comes from the speed at which the algorithms
    converge to a stable value; in the common case, estimation of covariance
    requires more steps than dual averaging to give an accurate value. See :cite:p:`stan_hmc_param`
    for a more detailed explanation.

    Fast intervals are given the label 0 and slow intervals the label 1.

    Parameters
    ----------
    num_steps: int
        The number of warmup steps to perform.
    initial_buffer_size: int
        The width of the initial fast adaptation interval.
    first_window_size: int
        The width of the first slow adaptation interval.
    final_buffer_size: int
        The width of the final fast adaptation interval.

    Returns
    -------
    A list of tuples (window_label, is_middle_window_end).

    """
    schedule = []

    # Give up on mass matrix adaptation when the number of warmup steps is too small.
    if num_steps < 20:
        schedule += [(0, False)] * num_steps
    else:
        # When the number of warmup steps is smaller that the sum of the provided (or default)
        # window sizes we need to resize the different windows.
        if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
            initial_buffer_size = int(0.15 * num_steps)
            final_buffer_size = int(0.1 * num_steps)
            first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # First stage: adaptation of fast parameters
        schedule += [(0, False)] * (initial_buffer_size - 1)
        schedule.append((0, False))

        # Second stage: adaptation of slow parameters in successive windows
        # doubling in size.
        final_buffer_start = num_steps - final_buffer_size

        next_window_size = first_window_size
        next_window_start = initial_buffer_size
        while next_window_start < final_buffer_start:
            current_start, current_size = next_window_start, next_window_size
            if 3 * current_size <= final_buffer_start - current_start:
                next_window_size = 2 * current_size
            else:
                current_size = final_buffer_start - current_start
            next_window_start = current_start + current_size
            schedule += [(1, False)] * (next_window_start - 1 - current_start)
            schedule.append((1, True))

        # Last stage: adaptation of fast parameters
        schedule += [(0, False)] * (num_steps - 1 - final_buffer_start)
        schedule.append((0, False))

    schedule = jnp.array(schedule)

    return schedule


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_metric_and_schedule(
    metric: str | MetricRecipe | MetricCore,
    schedule_fn: Callable | None,
    max_grad_budget: int | None,
    *,
    n_chains: int = 1,
    imm_shrinkage_to_previous: float = 0.0,
    initial_inverse_mass_matrix: Array | None = None,
) -> tuple[MetricCore, Callable]:
    """Resolve (metric, schedule_fn) to a (MetricCore, schedule_callable) pair.

    Extracted from :func:`staged_adaptation` for testability.

    The auto-override rule is:
    - ``metric="auto"`` AND ``schedule_fn is None`` (caller did not specify one)
      → schedule is set to :func:`~blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule`.
    - ``metric="auto"`` AND ``schedule_fn`` is an explicit callable
      → the explicit schedule is **honored as-is** (NOT replaced by the growing-window
      schedule even though ``metric="auto"``).
    - All other metric paths with ``schedule_fn is None`` → :func:`build_schedule` (Stan doubling).

    Parameters
    ----------
    metric
        The mass-matrix adaptation specification (same as :func:`staged_adaptation`).
    schedule_fn
        An explicit schedule callable, or ``None`` meaning "use the default for
        this metric path".
    max_grad_budget
        Required when ``metric="auto"``; ignored otherwise.
    imm_shrinkage_to_previous, initial_inverse_mass_matrix
        Forwarded to ``recipe.build_core`` for recipe-based metrics; ignored when
        ``metric`` is already a :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    Returns
    -------
    tuple[MetricCore, Callable]
        ``(metric_core, resolved_schedule_fn)``
    """
    if metric == "auto":
        if max_grad_budget is None:
            raise ValueError(
                "staged_adaptation: max_grad_budget is required when metric='auto'. "
                "Pass a positive integer, e.g. "
                "staged_adaptation(nuts, logdensity_fn, metric='auto', max_grad_budget=50_000)."
            )
        if n_chains > 1:
            from blackjax.adaptation.meta import build_multi_chain_meta_core

            metric_core = build_multi_chain_meta_core(max_grad_budget, n_chains)
        else:
            from blackjax.adaptation.meta import build_meta_adaptation_core

            metric_core = build_meta_adaptation_core(max_grad_budget)
        # Override the schedule ONLY when the caller has not specified one.
        # Using None as the sentinel (not build_schedule) is load-bearing: an
        # explicit schedule_fn=build_schedule must be preserved — the old
        # `if schedule_fn is build_schedule` sentinel could not distinguish
        # between "user explicitly chose Stan" and "user passed nothing".
        if schedule_fn is None:
            from blackjax.adaptation.low_rank_adaptation import (
                build_growing_window_schedule,
            )

            resolved_schedule: Callable = build_growing_window_schedule
        else:
            resolved_schedule = (
                schedule_fn  # explicit Callable, not None-guarded ternary
            )
    elif isinstance(metric, str):
        recipe = lookup_recipe(metric)
        metric_core = recipe.build_core(
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            initial_inverse_mass_matrix=initial_inverse_mass_matrix,
        )
        if schedule_fn is not None:
            resolved_schedule = schedule_fn
        else:
            resolved_schedule = build_schedule
    elif isinstance(metric, MetricRecipe):
        metric_core = metric.build_core(
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            initial_inverse_mass_matrix=initial_inverse_mass_matrix,
        )
        if schedule_fn is not None:
            resolved_schedule = schedule_fn
        else:
            resolved_schedule = build_schedule
    elif isinstance(metric, MetricCore):
        # The core is pre-built; imm_shrinkage_to_previous and
        # initial_inverse_mass_matrix are ignored (already closed over).
        metric_core = metric
        if schedule_fn is not None:
            resolved_schedule = schedule_fn
        else:
            resolved_schedule = build_schedule
    else:
        raise TypeError(
            f"staged_adaptation: metric must be a str, MetricRecipe, or MetricCore "
            f"(got {type(metric).__name__}). "
            f"Pass a registry name (e.g. 'welford_diag') or construct a "
            f"MetricRecipe or MetricCore directly."
        )
    return metric_core, resolved_schedule


def staged_adaptation(
    algorithm,
    logdensity_fn: Callable,
    metric: str | MetricRecipe | MetricCore = "welford_diag",
    *,
    max_grad_budget: int | None = None,
    n_chains: int = 1,
    imm_shrinkage_to_previous: float = 0.0,
    initial_inverse_mass_matrix: Array | None = None,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    schedule_fn: Callable | None = None,
    initial_metric_state: Any = None,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the step size and inverse mass matrix for HMC-family algorithms.

    The :func:`staged_adaptation` engine implements the same Stan warmup
    schedule as :func:`~blackjax.adaptation.window_adaptation.window_adaptation`
    but exposes a composable :class:`~blackjax.adaptation.metric_recipes.MetricCore`
    interface for the mass-matrix adaptation component.  The step-size dual-averaging
    and the stage schedule live in the HOST (this function); the mass-matrix
    estimation is fully delegated to the ``metric`` argument.

    Parameters
    ----------
    algorithm
        A sampling algorithm whose kernel signature is ``(rng_key, state,
        logdensity_fn, step_size, inverse_mass_matrix, **extra_parameters)``,
        e.g. :data:`blackjax.nuts`, :data:`blackjax.hmc`, :data:`blackjax.mhmc`.
        The algorithm's ``build_kernel`` method is inspected to decide whether
        to pass an integrator.
    logdensity_fn
        The log density probability density function to sample.
    metric
        The mass-matrix adaptation specification.  Accepts:

        - ``"auto"`` — the meta-adaptation controller
          (:mod:`~blackjax.adaptation.meta_adaptation`). Automatically selects
          the diagonal vs low-rank path and the growing-window schedule.
          Requires ``max_grad_budget`` to be set.  The emitted metric is always
          a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` (with
          U=0, lam=1 when the controller stays diagonal — bit-equivalent to
          the diagonal metric).

          .. warning::
             ``metric="auto"`` is **experimental (v1)**.  The low-rank
             escalation is not robustly calibrated at high dimension: when
             the residual spectrum's dominant structure sits near the detection
             boundary, whether the controller escalates can depend on the
             random seed.  Use for exploration and algorithm development, not
             for production efficiency claims.  A multi-chain escalation
             trigger (planned for v2) is expected to make the decision robust.
        - **str** — a registry name (``"welford_diag"`` (default),
          ``"welford_dense"``, ``"fisher_diag"``); looked up via
          :func:`~blackjax.adaptation.metric_recipes.lookup_recipe` and built
          with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
        - :class:`~blackjax.adaptation.metric_recipes.MetricRecipe` — built
          with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
        - :class:`~blackjax.adaptation.metric_recipes.MetricCore` — used
          directly as-is; ``imm_shrinkage_to_previous`` and
          ``initial_inverse_mass_matrix`` are ignored (closed over in the core).
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations).  Required when
        ``metric="auto"``; ignored otherwise.  The meta-adaptation controller
        converts this to a warmup step count via a conservative divisor (see
        :mod:`~blackjax.adaptation.meta_adaptation`).  Passed as-is; use
        :func:`~blackjax.adaptation.meta_adaptation.extract_meta_verdict` after
        ``warmup.run()`` to get the structured routing verdict and true gradient
        counts.
    imm_shrinkage_to_previous
        Pseudo-count controlling shrinkage of the per-window IMM toward the
        previous window's IMM (Bayesian persistence).  Default ``0.0``
        reproduces Stan's per-window-reset behavior exactly.  Ignored when
        ``metric`` is a :class:`~blackjax.adaptation.metric_recipes.MetricCore`.
    initial_inverse_mass_matrix
        Optional seed array for the initial inverse mass matrix.  Ignored when
        ``metric`` is a :class:`~blackjax.adaptation.metric_recipes.MetricCore`.
    initial_step_size
        Step size used to seed the dual-averaging adaptation.
    target_acceptance_rate
        Target Metropolis acceptance rate for step-size adaptation.  Default
        ``0.80`` (Stan default).
    adaptation_info_fn
        Function to select the adaptation info returned at each step.  See
        :func:`~blackjax.adaptation.base.return_all_adapt_info` and
        :func:`~blackjax.adaptation.base.get_filter_adapt_info_fn`.  By default
        all information is saved — this can result in excessive memory usage
        if the information is unused.
    integrator
        The symplectic integrator passed to ``algorithm.build_kernel``; only
        used if ``build_kernel`` accepts arguments.  Defaults to
        :func:`~blackjax.mcmc.integrators.velocity_verlet`.
    schedule_fn
        Callable ``(num_steps: int) -> Array`` that returns a
        ``(num_steps, 2)`` array of ``(stage, is_window_end)`` pairs, or
        ``None`` (default) to use the path-appropriate default.
        When ``None`` and ``metric="auto"``, the default is
        :func:`~blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule`
        (nutpie's proportional-to-tune, 1.5×-growing-window schedule).
        When ``None`` and any other ``metric``, the default is
        :func:`build_schedule` (Stan's fixed-absolute, 2×-doubling schedule).
        An explicit callable is always honored regardless of ``metric``.
    initial_metric_state
        Optional pre-built mass-matrix adaptation core state.  When not
        ``None``, overrides the ``metric_core.init(n_dims)`` call at warmup
        start — the provided state is used as-is.  The object must be a
        valid state for the chosen ``metric`` core (its
        ``inverse_mass_matrix`` field is unpacked into
        :class:`StagedAdaptationState` immediately).  Intended for callers
        that seed the initial state from external data (e.g., gradient-based
        diagonal-scale initialisation); ``None`` (the default) reproduces
        the standard identity/zero initialisation.
    **extra_parameters
        Algorithm-specific parameters forwarded to the MCMC kernel at every step,
        e.g. ``num_integration_steps`` for HMC/MHMC (divides budget when
        ``metric='auto'``) or ``num_max_steps`` for dynamic HMC.

    Returns
    -------
    AdaptationAlgorithm
        An :class:`~blackjax.base.AdaptationAlgorithm` wrapping a ``run``
        function with signature ``(rng_key, position, num_steps=1000)`` that
        returns ``(AdaptationResults, info)``.

    Notes
    -----
    Wrap ``warmup.run(...)`` in :func:`blackjax.progress_bar` to display a
    progress bar, e.g. ``with blackjax.progress_bar(): warmup.run(...)``.

    See Also
    --------
    blackjax.adaptation.window_adaptation.window_adaptation :
        Thin compatibility shim over this engine; preserves the old parameter
        interface exactly.
    blackjax.adaptation.metric_recipes.REGISTRY :
        Registry of named :class:`~blackjax.adaptation.metric_recipes.MetricRecipe`
        objects for the ``metric`` string argument.
    """
    # Resolve metric → MetricCore and schedule_fn → schedule callable.
    # The helper encapsulates the auto-override rule (growing-window only when
    # schedule_fn is None, never when the caller supplied one explicitly).
    # Use a distinct name so mypy knows _resolved_schedule_fn is Callable (not None).
    # Validate n_chains before resolution (fail early with a clear message).
    if n_chains < 1:
        raise ValueError(f"staged_adaptation: n_chains must be >= 1, got {n_chains}.")
    if n_chains > 1 and metric != "auto":
        raise ValueError(
            "staged_adaptation: n_chains > 1 is only supported with metric='auto' "
            "(the multi-chain pooled gate is implemented in the meta-adaptation "
            "controller). For other metric strings pass n_chains=1 (default) and "
            "vmap the warmup call externally."
        )

    metric_core, _resolved_schedule_fn = _resolve_metric_and_schedule(
        metric,
        schedule_fn,
        max_grad_budget,
        n_chains=n_chains,
        imm_shrinkage_to_previous=imm_shrinkage_to_previous,
        initial_inverse_mass_matrix=initial_inverse_mass_matrix,
    )

    # Closure variables for the auto-metric path: used inside run() to derive
    # num_steps from max_grad_budget when the caller does not supply one.
    _is_auto_metric: bool = metric == "auto"
    _is_multi_chain: bool = n_chains > 1
    _n_chains: int = n_chains
    _auto_max_grad_budget: int | None = max_grad_budget if _is_auto_metric else None

    if len(inspect.signature(algorithm.build_kernel).parameters) > 0:
        mcmc_kernel = algorithm.build_kernel(integrator)
    else:
        mcmc_kernel = algorithm.build_kernel()

    # Introspect the built kernel once to know which warmup-only overrides it
    # can accept.  Kernels that declare **kwargs accept everything; kernels with
    # an explicit parameter set (e.g. HMC accepts num_integration_steps but NOT
    # max_num_doublings) must not receive unknown kwargs — they raise TypeError.
    _kernel_sig_params = inspect.signature(mcmc_kernel).parameters
    _kernel_accepts_doublings = "max_num_doublings" in _kernel_sig_params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in _kernel_sig_params.values()
    )

    adapt_init, adapt_step, adapt_final = _make_engine(
        metric_core,
        target_acceptance_rate=target_acceptance_rate,
        # Multi-chain shared-ε: pass n_da_updates=n_chains to signal the
        # multi-chain path.  _make_engine will call da_update once on the mean
        # acceptance rate rather than running M sequential updates — the correct
        # statistical model for M chains sharing one epsilon.
        # Single-chain path (n_da_updates=1) is unchanged.
        n_da_updates=_n_chains if _is_multi_chain else 1,
    )

    if initial_metric_state is not None:
        # Narrow seam: override core.init with the caller-supplied state.
        # The base adapt_init still runs (sets up ss_state, step_size), and we
        # then replace imm_state + inverse_mass_matrix in the returned carry.
        _base_adapt_init = adapt_init

        def adapt_init(  # noqa: F811 — intentional shadowing; seam is local
            position: ArrayLikeTree, initial_step_size_: float
        ) -> StagedAdaptationState:
            state = _base_adapt_init(position, initial_step_size_)
            return state._replace(
                imm_state=initial_metric_state,
                inverse_mass_matrix=initial_metric_state.inverse_mass_matrix,
            )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            new_state.logdensity_grad,
            info.acceptance_rate,
        )

        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int | None = None):
        # Resolve num_steps: when metric="auto" and the caller did not supply a
        # value, derive it from max_grad_budget so the growing-window schedule's
        # largest window has enough draws to support the rank-detection check.
        # A None sentinel distinguishes "caller did not supply" from an explicit
        # value (even if that explicit value happens to equal 1000).
        # For multi-chain (n_chains > 1) the budget is split across chains, so
        # each chain runs total // n_chains steps.
        if num_steps is None:
            if _is_auto_metric:
                from blackjax.adaptation.meta._calibration import (
                    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
                )

                # _auto_max_grad_budget is non-None when _is_auto_metric is True
                # (_resolve_metric_and_schedule already validated it); the assert
                # lets mypy narrow the Optional[int] type.
                assert _auto_max_grad_budget is not None

                # Compute grads-per-step divisor. For algorithms with fixed
                # num_integration_steps (e.g. HMC), use it directly. For NUTS
                # (no num_integration_steps in extra_parameters), use the
                # NUTS-calibrated conservative constant.
                _grads_per_step = extra_parameters.get(
                    "num_integration_steps", _ASSUMED_AVG_LEAPFROGS_PER_STEP
                )
                _denom = _grads_per_step * (_n_chains if _is_multi_chain else 1)
                num_steps = max(_auto_max_grad_budget // _denom, 1)
            else:
                num_steps = 1000

        # Default effective schedule is the resolved one; overridden for the
        # multi-chain metric="auto" path inside the block below (BLOCKER-1 fix).
        _eff_schedule_fn = _resolved_schedule_fn

        # For metric="auto": warn when the largest warmup window is below the
        # rank-detection support floor for this model's dimension.  The check
        # runs once at run() call time (Python, not JAX-traced), so model
        # dimension is available from position.
        if _is_auto_metric:
            import numpy as _np

            from blackjax.adaptation.meta._calibration import (
                _MAX_RANK_CAP,
                _MIN_TRAIN_K_RATIO,
            )

            # For multi-chain, position has shape (M, d); use one chain's size.
            _pos_for_size = jnp.asarray(position)[0] if _is_multi_chain else position
            d = pytree_size(_pos_for_size)
            _actual_rank = min(_MAX_RANK_CAP, max(d // 2, 1))
            _min_rank_support = 2 * _MIN_TRAIN_K_RATIO * (_actual_rank + 1)

            # Pooled-aware schedule override for multi-chain path (BLOCKER-1 fix).
            # The single-chain growing-window schedule produces windows with
            # n_pool = per_chain_n ≤ 80 < min_n_proj = 208 for typical budgets,
            # making escalation structurally impossible.  The MC schedule starts at
            # n1 = ceil(min_n_proj / M) so every window ≥ n1 is escalation-eligible
            # when pooled (n_pool = M * per_chain_n ≥ min_n_proj).
            if _is_multi_chain:
                from blackjax.adaptation.meta._schedule import _build_mc_window_schedule

                _eff_schedule_fn = lambda _ns: _build_mc_window_schedule(
                    _ns, _n_chains, _actual_rank
                )
            else:
                _eff_schedule_fn = _resolved_schedule_fn

            # Find the largest slow window in the effective schedule.
            _sched_np = _np.asarray(_eff_schedule_fn(num_steps))
            _max_window = 0
            _window_start = 0
            for _i, (_stage, _end) in enumerate(zip(_sched_np[:, 0], _sched_np[:, 1])):
                if _end and _stage == 1:
                    _window_size = _i - _window_start + 1
                    if _window_size > _max_window:
                        _max_window = _window_size
                    _window_start = _i + 1

            # For multi-chain, pooled count = M * per-chain window size — this is
            # the effective support; compare it against _min_rank_support.
            _eff_max_window = _max_window * (_n_chains if _is_multi_chain else 1)
            if _max_window > 0 and _eff_max_window < _min_rank_support:
                _chains_suffix = f", n_chains={_n_chains}" if _is_multi_chain else ""
                _pool_suffix = (
                    f" (pooled count = {_eff_max_window})" if _is_multi_chain else ""
                )
                warnings.warn(
                    f"metric='auto': the largest warmup window ({_max_window} steps"
                    f"{_pool_suffix}) "
                    f"is below the rank-detection support floor for this model "
                    f"(d={d}, estimated rank capacity {_actual_rank}, "
                    f"floor={_min_rank_support} steps). "
                    f"Low-rank structure in the posterior may not be detectable "
                    f"with this budget (num_steps={num_steps}{_chains_suffix}). "
                    "Increase max_grad_budget to enable low-rank escalation "
                    "at this dimension.",
                    UserWarning,
                    stacklevel=2,
                )

        if not _is_multi_chain:
            # ----------------------------------------------------------------
            # Single-chain path (n_chains=1): unchanged from v1.
            # ----------------------------------------------------------------
            init_state = algorithm.init(position, logdensity_fn)
            init_adaptation_state = adapt_init(position, initial_step_size)

            start_state = (init_state, init_adaptation_state)
            keys = jax.random.split(rng_key, num_steps)
            schedule = _eff_schedule_fn(num_steps)
            last_state, info = jax.lax.scan(
                one_step,
                start_state,
                (jnp.arange(num_steps), keys, schedule),
            )

            last_chain_state, last_warmup_state, *_ = last_state

        else:
            # ----------------------------------------------------------------
            # Multi-chain path (n_chains > 1): vmap MCMC kernel over M chains.
            # position shape: (M, d); each chain gets its own rng_key split.
            # The metric_core.update/final receive (M, d) positions/grads.
            # num_steps is already the per-chain step count (total // n_chains).
            # ----------------------------------------------------------------

            # Warmup-only treedepth cap (metric="auto" multi-chain path, NUTS only).
            # The identity-metric first window with M dispersed inits produces
            # deep trees (987 lf/step on ill_cond vs 31 equilibrated), burning
            # warmup budget before the metric is known.  Cap max_num_doublings=5
            # (31 lf max) during warmup only; sampling runs uncapped (default 10,
            # or whatever the user set).  The cap is NOT included in the returned
            # parameters dict — it is a warmup-loop-only override.
            # Guard: only inject when the kernel actually accepts max_num_doublings
            # (NUTS does; HMC and other non-NUTS kernels do not — they raise TypeError
            # on an unknown kwarg if we inject it unconditionally).
            _WARMUP_DOUBLINGS_CAP = 5
            if _is_auto_metric and _kernel_accepts_doublings:
                _user_doublings = extra_parameters.get("max_num_doublings", 10)
                _warmup_extra_params: dict = {
                    **extra_parameters,
                    "max_num_doublings": min(_user_doublings, _WARMUP_DOUBLINGS_CAP),
                }
            else:
                _warmup_extra_params = extra_parameters

            init_states = jax.vmap(lambda pos: algorithm.init(pos, logdensity_fn))(
                position
            )
            # Adapt init uses one chain's position so pytree_size → d (not M*d).
            init_adaptation_state = adapt_init(
                jnp.asarray(position)[0], initial_step_size
            )

            def one_step_mc(carry, xs):
                _, rng_key_mc, adaptation_stage = xs
                states_mc, adaptation_state = carry

                # Split one key per chain for independent proposals.
                chain_keys = jax.random.split(rng_key_mc, _n_chains)

                def _step_one(key, state):
                    return mcmc_kernel(
                        key,
                        state,
                        logdensity_fn,
                        adaptation_state.step_size,
                        adaptation_state.inverse_mass_matrix,
                        **_warmup_extra_params,
                    )

                new_states_mc, infos_mc = jax.vmap(_step_one)(chain_keys, states_mc)

                # Pass (M, d) positions and grads to adapt_step; the multi-chain
                # metric core's update() handles (M, d) arrays natively.
                positions_mc = new_states_mc.position
                grads_mc = new_states_mc.logdensity_grad
                # Shared-ε: pass the full per-chain acceptance rate vector (M,)
                # to _maybe_multi_da_update, which takes jnp.mean and runs ONE
                # DA update on the mean — the correct observation model (M chains
                # at a shared epsilon contribute one mean observation).
                per_chain_accepts = infos_mc.acceptance_rate  # shape (M,)

                new_adaptation_state = adapt_step(
                    adaptation_state,
                    adaptation_stage,
                    positions_mc,
                    grads_mc,
                    per_chain_accepts,
                )

                return (
                    (new_states_mc, new_adaptation_state),
                    adaptation_info_fn(new_states_mc, infos_mc, new_adaptation_state),
                )

            start_state = (init_states, init_adaptation_state)
            keys = jax.random.split(rng_key, num_steps)
            schedule = _eff_schedule_fn(num_steps)
            last_state, info = jax.lax.scan(
                one_step_mc,
                start_state,
                (jnp.arange(num_steps), keys, schedule),
            )

            last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)
