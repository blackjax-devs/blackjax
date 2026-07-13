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
"""Staged warmup adaptation engine for the HMC family.

This module provides the :func:`staged_adaptation` engine and the
:func:`build_schedule` function (previously in ``window_adaptation.py``;
re-exported from there for backward compatibility).

Architecture (layer doctrine)
------------------------------
- :class:`StagedAdaptationState` — the scan-carry for the warmup.
- :func:`_make_engine` — builds the HOST: stage schedule dispatching +
  step-size dual averaging + metric core hooks.  Only the MetricCore protocol
  crosses the host/core boundary.
- :func:`staged_adaptation` — public entry point; accepts a recipe name, a
  :class:`~blackjax.adaptation.metric_recipes.MetricRecipe`, or a pre-built
  :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

The metric core (:class:`~blackjax.adaptation.metric_recipes.MetricCore`) is
the separable, embeddable component — its init/update/final protocol runs on
the engine's clock (Stan window schedule for slice 1).  Step-size dual
averaging lives in the HOST layer (this module), not in the core.

``WindowAdaptationState`` in :mod:`~blackjax.adaptation.window_adaptation`
is defined as ``WindowAdaptationState = StagedAdaptationState``.  Both names
refer to the same class object; ``isinstance`` checks using either name continue
to work.

Notes
-----
``build_schedule`` is defined here (the canonical location) and re-exported
from ``window_adaptation`` for backward compatibility.  Import it from either
module; the object is identical.
"""
import inspect
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
        acceptance_rate: float,
        ws: StagedAdaptationState,
    ) -> StagedAdaptationState:
        """Update adaptation state during a fast (step-size-only) window."""
        del position, grad
        new_ss = da_update(ws.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss.log_step_size)
        return StagedAdaptationState(
            new_ss, ws.imm_state, new_step_size, ws.inverse_mass_matrix
        )

    def slow_update(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: float,
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
        new_ss = da_update(ws.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss.log_step_size)
        return StagedAdaptationState(
            new_ss, new_metric_st, new_step_size, ws.inverse_mass_matrix
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


def staged_adaptation(
    algorithm,
    logdensity_fn: Callable,
    metric: str | MetricRecipe | MetricCore = "welford_diag",
    *,
    imm_shrinkage_to_previous: float = 0.0,
    initial_inverse_mass_matrix: Array | None = None,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
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
        An algorithm from the HMC family (e.g. :data:`blackjax.nuts`,
        :data:`blackjax.hmc`).  The algorithm's ``build_kernel`` method is
        inspected to decide whether to pass an integrator.
    logdensity_fn
        The log density probability density function to sample.
    metric
        The mass-matrix adaptation specification.  Accepts:

        - **str** — a registry name (``"welford_diag"`` (default),
          ``"welford_dense"``, ``"fisher_diag"``); looked up via
          :func:`~blackjax.adaptation.metric_recipes.lookup_recipe` and built
          with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
        - :class:`~blackjax.adaptation.metric_recipes.MetricRecipe` — built
          with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
        - :class:`~blackjax.adaptation.metric_recipes.MetricCore` — used
          directly as-is; ``imm_shrinkage_to_previous`` and
          ``initial_inverse_mass_matrix`` are ignored (closed over in the core).
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
    **extra_parameters
        Additional parameters forwarded to the MCMC kernel at every step, e.g.
        ``num_integration_steps`` for HMC.

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
    # Resolve the metric argument to a MetricCore.
    if isinstance(metric, str):
        recipe = lookup_recipe(metric)
        metric_core = recipe.build_core(
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            initial_inverse_mass_matrix=initial_inverse_mass_matrix,
        )
    elif isinstance(metric, MetricRecipe):
        metric_core = metric.build_core(
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            initial_inverse_mass_matrix=initial_inverse_mass_matrix,
        )
    elif isinstance(metric, MetricCore):
        # The core is pre-built; imm_shrinkage_to_previous and
        # initial_inverse_mass_matrix are ignored (already closed over).
        metric_core = metric
    else:
        raise TypeError(
            f"staged_adaptation: metric must be a str, MetricRecipe, or MetricCore "
            f"(got {type(metric).__name__}). "
            f"Pass a registry name (e.g. 'welford_diag') or construct a "
            f"MetricRecipe or MetricCore directly."
        )

    if len(inspect.signature(algorithm.build_kernel).parameters) > 0:
        mcmc_kernel = algorithm.build_kernel(integrator)
    else:
        mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_step, adapt_final = _make_engine(
        metric_core,
        target_acceptance_rate=target_acceptance_rate,
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

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_state = algorithm.init(position, logdensity_fn)
        init_adaptation_state = adapt_init(position, initial_step_size)

        start_state = (init_state, init_adaptation_state)
        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)
        last_state, info = jax.lax.scan(
            one_step,
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
