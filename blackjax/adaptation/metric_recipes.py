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
"""Metric recipes and the embeddable MetricCore protocol for staged_adaptation.

Available recipes
-----------------
Pass any of the following string names as the ``metric=`` argument to
:func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`:

- ``"welford_diag"`` — Stan-default diagonal Welford estimator; reproduces
  :func:`~blackjax.adaptation.window_adaptation.window_adaptation` exactly.
- ``"welford_dense"`` — Dense Welford covariance, same Stan schedule.
- ``"fisher_diag"`` — Fisher-divergence-minimising diagonal estimator
  (situational; requires position *and* gradient samples; see registry
  provenance note for operational guidance).
- ``"fisher_low_rank"`` — Fisher-divergence-minimising LOW-RANK estimator;
  requires position *and* gradient samples; needs a ``buffer_size``
  argument in ``build_core()``.  Algorithm 1 of
  :cite:p:`seyboldt2026preconditioning` / nutpie's mass-matrix estimator.
- ``"sample_cov_low_rank"`` — Sample-covariance low-rank estimator (MEADS
  / Scheme-B form); draws only, no gradients, no regularisation.  Needs a
  ``buffer_size`` argument in ``build_core()``.

Usage::

    # String sugar (registry lookup):
    wu = staged_adaptation(nuts, logdensity_fn, metric="welford_diag")

    # Low-rank: pre-build the core with buffer_size, then pass to engine:
    from blackjax.adaptation.metric_recipes import REGISTRY, _build_fisher_low_rank_core
    core = _build_fisher_low_rank_core(buffer_size=256, max_rank=10, gamma=1e-5, cutoff=2.0)
    wu = staged_adaptation(nuts, logdensity_fn, metric=core, schedule_fn=my_schedule_fn)

    # Via the recipe (also needs buffer_size):
    recipe = REGISTRY["fisher_low_rank"]
    core = recipe.build_core(buffer_size=256)
    wu = staged_adaptation(nuts, logdensity_fn, metric=core)

Design
------
A :class:`MetricRecipe` declares an (estimator, buffer, representation,
support_gate) tuple with construction-time validation of the coupling
contract (``needs ⊆ provides`` and ``emits == representation``): incompatible
combos fail at Python level with a clear message, never inside traced code.

A :class:`MetricCore` is the embeddable mass-matrix adaptation component —
the separable piece that the staged_adaptation engine hosts.  Step-size
dual averaging and the stage schedule are HOST-layer concerns; this core
handles only the inverse-mass-matrix estimation.

Layer doctrine:

- The METRIC CORE (:class:`MetricCore`) handles mass-matrix tuning only.
- Step-size adaptation and the stage schedule live in the HOST
  (:mod:`~blackjax.adaptation.staged_adaptation`).
- L3 step-size proxy independence (RFC §4.3): for HMC/NUTS, the dual-averaging
  step-size proxy is the scalar Metropolis acceptance rate — NOT an eigenvalue
  quantity of the adapted metric.  This differs from MCLMC-LRD where
  step_size ∝ 1/√λ_max caused a 20.6×→1.27× effective-sample collapse (the
  "ε-collapse" RFC finding).  For HMC/NUTS the full low-rank metric feeds the
  MCMC kernel (correct coupling), and dual averaging reads only acceptance_rate
  — L3 is naturally satisfied; no diagonal-reference split is needed.  This
  analysis applies to ALL recipes in this module, including the low-rank slice.

Notes
-----
The :class:`MetricRecipe` schema (field types) is provisional:
estimator/buffer/support_gate are currently string tags.  Future slices will
replace these with proper constructor objects once the schema is stable across
all recipe families.  Import directly from ``blackjax.adaptation.metric_recipes``
— :class:`MetricRecipe` is not exported at the ``blackjax`` top level.
"""
import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.flatten_util as fu
import jax.numpy as jnp

from blackjax.adaptation.mass_matrix import (
    FisherMassMatrixAdaptationState,
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.adaptation.metric_estimators import (
    _compute_low_rank_metric,
    fisher_score_diagonal_from_moments,
    sample_covariance_eigh_low_rank,
)
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "LowRankMetricCoreState",
    "MetricCore",
    "MetricRecipe",
    "REGISTRY",
    "lookup_recipe",
    "seed_low_rank_sigma_from_grad",
]


# ---------------------------------------------------------------------------
# MetricCore — the embeddable init/update/final protocol
# ---------------------------------------------------------------------------


class MetricCore(NamedTuple):
    """Embeddable mass-matrix adaptation core: init/update/final protocol.

    A NamedTuple-of-callables (house style) bundling the three operations that
    together constitute mass-matrix adaptation.  The engine hosts this core;
    step-size adaptation and the stage schedule remain in the host layer.

    This core is hostable by warmups that declare no intrinsic metric adaptation
    scheme (i.e. the metric core can be swapped without changing the host's step-
    size or schedule logic).  It is NOT wired into MEADS, whose fold-based metric
    is co-designed with its damping and step rules and cannot be factored out.

    Parameters
    ----------
    init : Callable
        ``(n_dims: int) -> MetricCoreState``.  Creates the initial mass-matrix
        adaptation state.  Closes over ``initial_inverse_mass_matrix`` and
        ``imm_shrinkage_to_previous`` when constructed via
        :meth:`MetricRecipe.build_core`.
    update : Callable
        ``(state, position: ArrayLikeTree, grad: ArrayLikeTree | None) -> MetricCoreState``.
        Accumulates one sample.  For welford-path recipes ``grad`` is accepted
        (interface uniformity) but ignored.
    final : Callable
        ``(state) -> MetricCoreState``.  Called at each slow-window boundary:
        computes the new inverse mass matrix, writes it to
        ``state.inverse_mass_matrix``, resets the accumulator.  The host
        reads ``new_state.inverse_mass_matrix`` for the next window.

    Notes
    -----
    ``MetricCoreState`` is one of
    :class:`~blackjax.adaptation.mass_matrix.MassMatrixAdaptationState` or
    :class:`~blackjax.adaptation.mass_matrix.FisherMassMatrixAdaptationState` —
    the existing in-tree types; this core is a thin protocol wrapper, not a
    re-implementation.
    """

    init: Callable
    update: Callable
    final: Callable


# ---------------------------------------------------------------------------
# LowRankMetricCoreState — state type for the low-rank MetricCore
# ---------------------------------------------------------------------------


class LowRankMetricCoreState(NamedTuple):
    """Scan-carry state for the low-rank mass-matrix MetricCore.

    Holds the current low-rank inverse mass matrix factors, the draw/gradient
    circular buffer, and the buffer bookkeeping counters.  The engine reads
    ``inverse_mass_matrix`` at each window boundary; the core's ``final()``
    updates it.

    Parameters
    ----------
    inverse_mass_matrix
        Current low-rank IMM as a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple
        ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
        ``(max_rank,)``.  This field is read by the engine at window boundaries
        (via ``StagedAdaptationState.inverse_mass_matrix``) and by the MCMC
        kernel's ``default_metric`` dispatch.
    mu_star
        Optimal translation ``x̄ + σ² ⊙ ᾱ``, shape ``(d,)``.  Not part of the
        engine's host protocol; the shim reads this from the last adaptation
        state to re-initialize the chain after warmup.  Always zero for the
        ``"sample_cov_low_rank"`` recipe (no optimal translation in that
        estimator).
    draws_buffer
        Circular buffer of chain positions, shape ``(buffer_size, d)``.
        The first ``buffer_idx`` rows are valid; the remainder are zero-padded.
        Dropped (replaced with ``None``) by the default OOM-guard
        ``adaptation_info_fn`` in the shim to avoid O(num_steps × buffer_size × d)
        allocations inside ``jax.lax.scan``.
    grads_buffer
        Circular buffer of log-density gradients, shape ``(buffer_size, d)``.
        Same layout and OOM-guard treatment as ``draws_buffer``.  Always zeros
        for the ``"sample_cov_low_rank"`` recipe (not used by that estimator).
    buffer_idx
        Number of draws written to the buffer so far (monotonically increasing,
        NOT wrapped).  Modular indexing in ``update()`` handles wrap-around so
        the most recent ``buffer_size`` draws are always in the buffer.
        Reset to 0 by ``final()`` under the default ``"reset"`` buffer policy.
    background_split
        Number of the buffer's leading rows considered "background" (for the
        accumulating buffer policy only).  Always 0 under the default
        ``"reset"`` policy.
    recompute_counter
        Steps since the last metric recompute (for accumulating periodic
        recompute only).  Always 0 under the default ``"reset"`` policy.
    """

    inverse_mass_matrix: LowRankInverseMassMatrix
    mu_star: Array
    draws_buffer: Array
    grads_buffer: Array
    buffer_idx: int
    background_split: int
    recompute_counter: int


# ---------------------------------------------------------------------------
# Seeding helpers for gradient_based_init
# ---------------------------------------------------------------------------


def seed_low_rank_sigma_from_grad(
    state: LowRankMetricCoreState,
    grad: ArrayLikeTree,
) -> LowRankMetricCoreState:
    """Seed the diagonal scale ``sigma`` from the initial log-density gradient.

    Implements nutpie's ``gradient_based_init`` logic: instead of starting from
    the identity (``sigma=1`` for every coordinate), set
    ``sigma_i = 1/sqrt(clip(|grad_i|, 1e-20, 1e20))`` so that the initial
    diagonal inverse mass matrix equals ``M^{-1}_i = sigma_i^2 = 1/|grad_i|``,
    matching ``M = diag(|grad|)`` (a diagonal Hessian approximation at the
    starting point; cf. L-BFGS and paper §3.1).

    Coordinates where ``|grad_i| < 1e-10`` fall back to ``sigma_i = 1.0``
    (identity) rather than the astronomically large ``sigma_i = 1e10`` that the
    raw formula would give.  This defends the real edge case of initialising
    at (or very near) a stationary point of the target — e.g. ``x=0`` on any
    centered/standardised density — where the gradient is exactly zero and an
    extreme initial scale causes near-certain divergence on the very first
    trajectory (root-caused via the Fisher 2×2 calibration study).

    This function is a **named seeding entry point** so that any host (the
    window-adaptation shim, ChEES, etc.) can call the same code path and the
    seeding logic is independently testable.

    Parameters
    ----------
    state
        Initial :class:`LowRankMetricCoreState` from ``core.init(n_dims)``
        (before any gradient information).
    grad
        Log-density gradient at the initial position.  Must be the same pytree
        structure as the chain's position.

    Returns
    -------
    LowRankMetricCoreState
        State with ``sigma`` replaced by the gradient-seeded values and
        ``inverse_mass_matrix`` updated accordingly (``U``/``lam`` unchanged,
        ``mu_star`` unchanged).
    """
    grad_flat, _ = fu.ravel_pytree(grad)
    abs_grad = jnp.abs(grad_flat)
    near_zero_grad_threshold = 1e-10
    safe_sigma = jnp.power(jnp.clip(abs_grad, 1e-20, 1e20), -0.5)
    sigma = jnp.where(abs_grad < near_zero_grad_threshold, 1.0, safe_sigma)
    new_imm = LowRankInverseMassMatrix(
        sigma=sigma,
        U=state.inverse_mass_matrix.U,
        lam=state.inverse_mass_matrix.lam,
    )
    return state._replace(inverse_mass_matrix=new_imm)


# ---------------------------------------------------------------------------
# Buffer utility (for accumulating policy; currently used by reset path too)
# ---------------------------------------------------------------------------


def _shift_buffer_left(buf: Array, shift: Array) -> Array:
    """Drop the first ``shift`` rows of ``buf``, shifting the remainder forward.

    Implements nutpie's partial-forget buffer pop under JAX's static-shape
    constraint.  Canonical source: ``blackjax.adaptation.low_rank_adaptation``
    ``._shift_buffer_left`` (same logic, defined here to avoid a circular
    import since ``low_rank_adaptation`` is downstream of ``metric_recipes``
    in the import chain).

    Used by the accumulating buffer policy's ``final()`` (``slow_switch``
    logic).  Not used by the default ``"reset"`` policy.
    """
    capacity = buf.shape[0]
    shift = jnp.clip(shift, 0, capacity)
    padded = jnp.concatenate([buf, jnp.zeros_like(buf)], axis=0)
    return jax.lax.dynamic_slice_in_dim(padded, shift, capacity, axis=0)


# ---------------------------------------------------------------------------
# MetricRecipe — frozen dataclass with construction-time validation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MetricRecipe:
    """Configuration bundle for a mass-matrix adaptation recipe.

    Declares an (estimator, buffer, representation, support_gate) tuple with
    construction-time validation of the coupling contract (``needs ⊆ provides``
    and ``emits == representation``): incompatible combos fail at Python level
    with a clear message, never inside traced code.

    .. note::

        **Schema is provisional.**  The field types for ``estimator``,
        ``buffer``, and ``support_gate`` are string tags; future slices will
        replace these with proper constructor objects once the schema is stable
        across all recipe families.  This class is not exported at the
        ``blackjax`` top level — import directly from
        ``blackjax.adaptation.metric_recipes``.

    Parameters
    ----------
    representation
        The inverse-mass-matrix representation this recipe produces.
        Slice-1 values: ``"diag"`` (1D array) or ``"dense"`` (2D array).
    estimator
        String tag for the estimator function family.
        Slice-1 values: ``"welford"``, ``"fisher_diag"``.
    buffer
        String tag for the buffer/data-feeding policy.
        Slice-1 value: ``"reset_window"``.
    support_gate
        String tag for the support gate, or ``None`` (slice-1 default — no
        gating beyond the estimator's intrinsic validation).
    needs
        ``frozenset[str]`` declaring what data the estimator requires from the
        buffer.  Validated at construction: ``needs ⊆ provides``.
        Slice-1 values: ``frozenset({"positions"})`` or
        ``frozenset({"positions", "gradients"})``.
    provides
        ``frozenset[str]`` declaring what data the buffer provides.
    emits
        Representation tag this estimator emits.  Validated at construction:
        ``emits == representation``.
    provenance
        Human-readable guidance string, stamped with benchmark evidence where
        available.
    """

    representation: str
    estimator: str
    buffer: str
    support_gate: str | None
    needs: frozenset
    provides: frozenset
    emits: str
    provenance: str
    # Low-rank-specific fields (None for diag/dense recipes).
    max_rank: int | None = None
    gamma: float | None = None
    cutoff: float | None = None

    def __post_init__(self) -> None:
        """Validate coupling contract at construction time."""
        if not self.needs <= self.provides:
            missing = sorted(self.needs - self.provides)
            raise ValueError(
                f"MetricRecipe coupling violation: estimator needs {missing} "
                f"but the buffer only provides {sorted(self.provides)}. "
                f"Choose a buffer that provides the missing data, or choose a "
                f"compatible estimator that does not need {missing}."
            )
        if self.emits != self.representation:
            raise ValueError(
                f"MetricRecipe coupling violation: estimator emits {self.emits!r} "
                f"but recipe declares representation {self.representation!r}. "
                f"Choose an estimator whose emits tag matches the declared "
                f"representation, or update representation to {self.emits!r}."
            )

    def build_core(
        self,
        *,
        imm_shrinkage_to_previous: float = 0.0,
        initial_inverse_mass_matrix: Array | None = None,
        buffer_size: int | None = None,
    ) -> MetricCore:
        """Build an embeddable :class:`MetricCore` from this recipe.

        Parameters
        ----------
        imm_shrinkage_to_previous
            Pseudo-count controlling shrinkage of the per-window IMM toward the
            previous window's IMM.  Default ``0.0`` (Stan vanilla, no
            persistence).  Forwarded to
            :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`.
            Not supported for ``"fisher_diag"`` or the low-rank recipes
            (``ValueError`` there per ``mass_matrix_adaptation``'s validation).
        initial_inverse_mass_matrix
            Optional seed array for the initial inverse mass matrix.  ``None``
            (default) uses the standard identity initialisation (``ones(d)``
            for diagonal, ``identity(d)`` for dense).  Ignored for the
            low-rank estimators (``"fisher_low_rank"``, ``"sample_cov_low_rank"``).
        buffer_size
            Required for low-rank recipes (``"fisher_low_rank"`` and
            ``"sample_cov_low_rank"``).  Size of the circular draw/gradient
            buffer (number of rows).  Use the schedule-derived heuristic in
            :func:`~blackjax.adaptation.low_rank_adaptation.window_adaptation_low_rank`
            or compute it yourself:
            ``min(2 * max(num_steps // 5, 128), max(num_steps, 1))`` for the
            reset policy; :func:`~blackjax.adaptation.low_rank_adaptation
            ._accumulating_buffer_capacity` for the accumulating policy.
            Ignored for diag/dense recipes.

        Returns
        -------
        MetricCore
            Embeddable init/update/final bundle ready for the engine.

        Raises
        ------
        ValueError
            If the estimator tag is not supported, or if ``buffer_size`` is
            ``None`` for a low-rank estimator.
        """
        if self.estimator == "welford":
            is_diagonal = self.representation == "diag"
            return _build_welford_core(
                is_diagonal=is_diagonal,
                imm_shrinkage_to_previous=imm_shrinkage_to_previous,
                initial_inverse_mass_matrix=initial_inverse_mass_matrix,
            )
        elif self.estimator == "fisher_diag":
            return _build_fisher_diag_core(
                initial_inverse_mass_matrix=initial_inverse_mass_matrix,
            )
        elif self.estimator == "fisher_low_rank":
            if buffer_size is None:
                raise ValueError(
                    "MetricRecipe.build_core: buffer_size is required for "
                    "the 'fisher_low_rank' estimator.  Pass buffer_size=<int> "
                    "or build the core directly via _build_fisher_low_rank_core()."
                )
            # max_rank/gamma/cutoff are guaranteed non-None by the registry
            # constructor; the Optional typing is for the dataclass default.
            assert (
                self.max_rank is not None
            ), "fisher_low_rank recipe must have max_rank"
            assert self.gamma is not None, "fisher_low_rank recipe must have gamma"
            assert self.cutoff is not None, "fisher_low_rank recipe must have cutoff"
            return _build_fisher_low_rank_core(
                buffer_size=buffer_size,
                max_rank=self.max_rank,
                gamma=self.gamma,
                cutoff=self.cutoff,
            )
        elif self.estimator == "sample_cov_low_rank":
            if buffer_size is None:
                raise ValueError(
                    "MetricRecipe.build_core: buffer_size is required for "
                    "the 'sample_cov_low_rank' estimator.  Pass buffer_size=<int> "
                    "or build the core directly via _build_sample_cov_low_rank_core()."
                )
            assert (
                self.max_rank is not None
            ), "sample_cov_low_rank recipe must have max_rank"
            return _build_sample_cov_low_rank_core(
                buffer_size=buffer_size,
                max_rank=self.max_rank,
            )
        else:
            raise ValueError(
                f"Unknown estimator tag {self.estimator!r} in MetricRecipe.build_core. "
                f"Known estimators: 'welford', 'fisher_diag', 'fisher_low_rank', "
                f"'sample_cov_low_rank'."
            )


# ---------------------------------------------------------------------------
# Private core builders
# ---------------------------------------------------------------------------


def _build_welford_core(
    *,
    is_diagonal: bool,
    imm_shrinkage_to_previous: float,
    initial_inverse_mass_matrix: Array | None,
) -> MetricCore:
    """Build a MetricCore wrapping the Welford (Stan-default) estimator path.

    Delegates to :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`
    with ``diagonal_estimator='welford'``.  The MetricCore ``final()`` calls
    ``mm_final`` directly — same reduction order, same arithmetic, bit-exact.

    Parameters
    ----------
    is_diagonal
        ``True`` for diagonal (``welford_diag``), ``False`` for dense
        (``welford_dense``).
    imm_shrinkage_to_previous
        Forwarded to ``mass_matrix_adaptation``.
    initial_inverse_mass_matrix
        Optional seed IMM; closed over in ``init``.
    """
    mm_init, mm_update, mm_final = mass_matrix_adaptation(
        is_diagonal_matrix=is_diagonal,
        imm_shrinkage_to_previous=imm_shrinkage_to_previous,
        diagonal_estimator="welford",
    )

    def init(n_dims: int) -> MassMatrixAdaptationState:
        return mm_init(n_dims, initial_inverse_mass_matrix)

    def update(
        state: MassMatrixAdaptationState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,
    ) -> MassMatrixAdaptationState:
        # grad is accepted for interface uniformity but ignored by mm_update
        # on the welford path (mm_update only reads grad for
        # FisherMassMatrixAdaptationState — checked via isinstance at trace time).
        return mm_update(state, position, grad)

    def final(state: MassMatrixAdaptationState) -> MassMatrixAdaptationState:
        # Directly calls mm_final: same Welford reduction, same regularization
        # formula as window_adaptation.base()'s slow_final (welford path).
        return mm_final(state)

    return MetricCore(init=init, update=update, final=final)


def _build_fisher_diag_core(
    *,
    initial_inverse_mass_matrix: Array | None,
) -> MetricCore:
    """Build a MetricCore wrapping the Fisher-diagonal estimator path.

    Delegates to :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`
    with ``diagonal_estimator='fisher'`` for ``init`` and ``update``.  The
    ``final()`` below composes the IMM computation and block reset — logic
    that cannot live inside ``mass_matrix_adaptation.final()`` due to a
    circular import (``metric_estimators`` imports ``welford_algorithm``
    from ``mass_matrix``).

    Parameters
    ----------
    initial_inverse_mass_matrix
        Optional seed IMM; closed over in ``init``.
    """
    mm_init, mm_update, mm_final = mass_matrix_adaptation(
        is_diagonal_matrix=True,
        imm_shrinkage_to_previous=0.0,
        diagonal_estimator="fisher",
    )

    def init(n_dims: int) -> FisherMassMatrixAdaptationState:
        return mm_init(n_dims, initial_inverse_mass_matrix)

    def update(
        state: FisherMassMatrixAdaptationState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,
    ) -> FisherMassMatrixAdaptationState:
        return mm_update(state, position, grad)

    def final(
        state: FisherMassMatrixAdaptationState,
    ) -> FisherMassMatrixAdaptationState:
        # Fisher IMM computation + block reset: the only stitch site.
        block = state.fisher_block
        denom = jnp.maximum(block.count - 1.0, 1.0)
        var_x = block.m2_x / denom  # (d,) Bessel-corrected position variance
        var_g = block.m2_g / denom  # (d,) Bessel-corrected gradient variance
        new_inverse_mass_matrix = fisher_score_diagonal_from_moments(var_x, var_g)
        # mm_final resets the block for the next window.
        reset_state = mm_final(state)
        return FisherMassMatrixAdaptationState(
            inverse_mass_matrix=new_inverse_mass_matrix,
            fisher_block=reset_state.fisher_block,
        )

    return MetricCore(init=init, update=update, final=final)


def _build_fisher_low_rank_core(
    *,
    buffer_size: int,
    max_rank: int,
    gamma: float,
    cutoff: float,
) -> MetricCore:
    """Build a MetricCore for the Fisher-score low-rank estimator (reset policy).

    Implements the same window-end recompute as
    :func:`~blackjax.adaptation.low_rank_adaptation.base`'s ``slow_final``
    under ``buffer_policy="reset"``:

    1. ``init(n_dims)`` — creates a zero-filled draw/gradient buffer of shape
       ``(buffer_size, n_dims)`` with identity sigma, zero U, ones lam.
    2. ``update(state, position, grad)`` — writes (position, grad) at
       ``buffer_idx % buffer_size`` via dynamic update; increments ``buffer_idx``.
    3. ``final(state)`` — calls :func:`~blackjax.adaptation.metric_estimators
       ._compute_low_rank_metric` on the buffer, stores new sigma/mu_star/U/lam
       in ``inverse_mass_matrix`` and ``mu_star``, resets the buffer to zeros.

    **L3 note** (RFC §4.3): the dual-averaging step-size proxy reads only the
    scalar Metropolis acceptance rate, NOT an eigenvalue quantity — L3 is
    naturally satisfied for HMC/NUTS regardless of metric rank.  See module
    docstring for the full analysis.

    Parameters
    ----------
    buffer_size
        Number of rows in the circular draw/gradient buffer.
    max_rank
        Maximum number of eigenvectors retained in the low-rank correction.
        Default 10, matching :func:`~blackjax.adaptation.low_rank_adaptation
        .window_adaptation_low_rank`.
    gamma
        Regularisation scale (nutpie convention — projected covariance divided
        by ``gamma`` before adding identity, no ``n`` scaling).  Default
        1e-5.
    cutoff
        Eigenvalues in ``[1/cutoff, cutoff]`` are masked to 1.  Default 2.0.
    """

    def init(n_dims: int) -> LowRankMetricCoreState:
        return LowRankMetricCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, max_rank)),
                lam=jnp.ones(max_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((buffer_size, n_dims)),
            grads_buffer=jnp.zeros((buffer_size, n_dims)),
            buffer_idx=0,
            background_split=0,
            recompute_counter=0,
        )

    def update(
        state: LowRankMetricCoreState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,
    ) -> LowRankMetricCoreState:
        pos_flat, _ = fu.ravel_pytree(position)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B  # wrap to avoid out-of-bounds during trace
        new_draws = jax.lax.dynamic_update_slice(
            state.draws_buffer, pos_flat[None, :], (idx, 0)
        )
        new_grads = jax.lax.dynamic_update_slice(
            state.grads_buffer, grad_flat[None, :], (idx, 0)
        )
        return state._replace(
            draws_buffer=new_draws,
            grads_buffer=new_grads,
            buffer_idx=state.buffer_idx + 1,
        )

    def final(state: LowRankMetricCoreState) -> LowRankMetricCoreState:
        # Recompute metric from buffer, then hard-reset for the next window.
        sigma, mu_star, U, lam = _compute_low_rank_metric(
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
            max_rank,
            gamma,
            cutoff,
        )
        B, d = state.draws_buffer.shape
        return LowRankMetricCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam),
            mu_star=mu_star,
            draws_buffer=jnp.zeros((B, d)),
            grads_buffer=jnp.zeros((B, d)),
            buffer_idx=0,
            background_split=0,
            recompute_counter=0,
        )

    return MetricCore(init=init, update=update, final=final)


def _build_sample_cov_low_rank_core(
    *,
    buffer_size: int,
    max_rank: int,
) -> MetricCore:
    """Build a MetricCore for the sample-covariance low-rank estimator (MEADS form).

    Draws-only variant (no gradient data, no regularisation, raw top-k eigh
    selection) following MEADS / Scheme-B.  Delegates to
    :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`
    in ``final()``.

    1. ``init(n_dims)`` — same buffer layout as the Fisher core; ``grads_buffer``
       is allocated but stays zero (not used).
    2. ``update(state, position, grad)`` — writes only ``position`` to
       ``draws_buffer``; grad is accepted for interface uniformity but ignored.
    3. ``final(state)`` — computes the masked sample covariance matrix from the
       buffer, calls :func:`~blackjax.adaptation.metric_estimators
       .sample_covariance_eigh_low_rank`, resets buffer.  ``mu_star`` is always
       zero (this estimator does not compute an optimal translation).

    Parameters
    ----------
    buffer_size
        Number of rows in the circular draw buffer.
    max_rank
        Maximum number of eigenvectors retained.
    """

    def init(n_dims: int) -> LowRankMetricCoreState:
        return LowRankMetricCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, max_rank)),
                lam=jnp.ones(max_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((buffer_size, n_dims)),
            grads_buffer=jnp.zeros((buffer_size, n_dims)),  # unused; zeros always
            buffer_idx=0,
            background_split=0,
            recompute_counter=0,
        )

    def update(
        state: LowRankMetricCoreState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,  # accepted for interface uniformity
    ) -> LowRankMetricCoreState:
        pos_flat, _ = fu.ravel_pytree(position)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B
        new_draws = jax.lax.dynamic_update_slice(
            state.draws_buffer, pos_flat[None, :], (idx, 0)
        )
        return state._replace(
            draws_buffer=new_draws,
            buffer_idx=state.buffer_idx + 1,
        )

    def final(state: LowRankMetricCoreState) -> LowRankMetricCoreState:
        B, d = state.draws_buffer.shape
        n = state.buffer_idx
        # Compute masked mean and accumulated sum of squared deviations from
        # the buffer (same masking pattern as _compute_low_rank_metric).
        mask = (jnp.arange(B) < n).astype(state.draws_buffer.dtype)
        n_safe = jnp.maximum(n, 2).astype(state.draws_buffer.dtype)
        mean = (mask[:, None] * state.draws_buffer).sum(0) / n_safe  # (d,)
        diff = mask[:, None] * (state.draws_buffer - mean[None, :])  # (B, d)
        m2 = diff.T @ diff  # (d, d) accumulated sum of squared deviations
        new_imm = sample_covariance_eigh_low_rank(m2, n, max_rank)
        return LowRankMetricCoreState(
            inverse_mass_matrix=new_imm,
            mu_star=jnp.zeros(d),  # no optimal translation for this estimator
            draws_buffer=jnp.zeros((B, d)),
            grads_buffer=jnp.zeros((B, d)),
            buffer_idx=0,
            background_split=0,
            recompute_counter=0,
        )

    return MetricCore(init=init, update=update, final=final)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, MetricRecipe] = {
    "welford_diag": MetricRecipe(
        representation="diag",
        estimator="welford",
        buffer="reset_window",
        support_gate=None,
        needs=frozenset({"positions"}),
        provides=frozenset({"positions"}),
        emits="diag",
        provenance=(
            "Stan-default diagonal Welford covariance estimator. "
            "Reproduces window_adaptation exact behavior (bit-exact via the same "
            "mass_matrix.welford_algorithm reduction). Use as the baseline recipe."
        ),
    ),
    "welford_dense": MetricRecipe(
        representation="dense",
        estimator="welford",
        buffer="reset_window",
        support_gate=None,
        needs=frozenset({"positions"}),
        provides=frozenset({"positions"}),
        emits="dense",
        provenance=(
            "Dense Welford covariance estimator (Stan-style reset-window schedule, "
            "no low-rank compression). Appropriate when d is small enough to afford "
            "O(d^2) storage and the full correlation structure matters."
        ),
    ),
    "fisher_diag": MetricRecipe(
        representation="diag",
        estimator="fisher_diag",
        buffer="reset_window",
        support_gate=None,
        needs=frozenset({"positions", "gradients"}),
        provides=frozenset({"positions", "gradients"}),
        emits="diag",
        provenance=(
            "Situational estimator, not a default. "
            "Helps concentrated-anisotropy hierarchical geometry "
            "(~1.3–1.7x effective-samples-per-gradient); "
            "degrades well-conditioned targets with correlated coordinate blocks "
            "(~0.6–0.7x). "
            "Mechanism: for near-Gaussian posteriors this estimator equals the "
            "Welford diagonal shrunk by sqrt(1 - R_i^2) per coordinate "
            "(the marginal-to-conditional precision pull); without an off-diagonal "
            "correction the shrink under-explores correlated coordinates, raising "
            "the slowest-mode variance. "
            "Risk boundary: correlated blocks (R^2 >= ~0.5) on an already "
            "well-conditioned target with no rank-k correction. "
            "Degradation carries no R-hat/divergence signature — gate estimator "
            "choices on effective-samples-per-gradient, not health flags. "
            "Validated 2026-07-13."
        ),
    ),
    "fisher_low_rank": MetricRecipe(
        representation="low_rank",
        estimator="fisher_low_rank",
        buffer="reset_window",
        support_gate=None,
        needs=frozenset({"positions", "gradients"}),
        provides=frozenset({"positions", "gradients"}),
        emits="low_rank",
        max_rank=10,
        gamma=1e-5,
        cutoff=2.0,
        provenance=(
            "Fisher-divergence-minimising LOW-RANK estimator (Algorithm 1, "
            "seyboldt2026preconditioning; nutpie reference implementation). "
            "Defaults match window_adaptation_low_rank: max_rank=10, gamma=1e-5, "
            "cutoff=2.0, reset buffer policy.  Requires buffer_size in build_core(). "
            "Uses position AND gradient samples; requires x64 for float32 chains "
            "(see _compute_low_rank_metric dtype note).  Composes with any "
            "schedule_fn; pair with build_growing_window_schedule for the nutpie "
            "schedule.  L3 is naturally satisfied for HMC/NUTS (acceptance rate "
            "proxy is not an eigenvalue quantity — see module docstring)."
        ),
    ),
    "sample_cov_low_rank": MetricRecipe(
        representation="low_rank",
        estimator="sample_cov_low_rank",
        buffer="reset_window",
        support_gate=None,
        needs=frozenset({"positions"}),
        provides=frozenset({"positions"}),
        emits="low_rank",
        max_rank=10,
        gamma=None,
        cutoff=None,
        provenance=(
            "Sample-covariance low-rank estimator (MEADS / Scheme-B form). "
            "Draws only — no gradient data, no regularisation (no gamma), "
            "raw top-k eigh selection (no informativeness cutoff masking). "
            "Delegates to sample_covariance_eigh_low_rank.  Appropriate when "
            "gradients are unavailable or expensive and the full covariance "
            "structure matters more than the AIRM geometric-mean correction.  "
            "No optimal translation (mu_star=0).  Requires buffer_size in "
            "build_core(); max_rank=10 default."
        ),
    ),
}


def lookup_recipe(name: str) -> MetricRecipe:
    """Look up a named recipe from the :data:`REGISTRY`.

    Parameters
    ----------
    name
        Registry key.  Current names:

        - ``"welford_diag"`` (default; reproduces ``window_adaptation`` exactly)
        - ``"welford_dense"``
        - ``"fisher_diag"``
        - ``"fisher_low_rank"`` (Algorithm 1, seyboldt2026; needs ``buffer_size``)
        - ``"sample_cov_low_rank"`` (MEADS/Scheme-B; needs ``buffer_size``)

    Returns
    -------
    MetricRecipe
        The registered recipe for ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry, with a sorted list of known names.
        Pass a :class:`MetricRecipe` or :class:`MetricCore` directly for
        custom or experimental recipes that are not registry-stamped.
    """
    if name not in REGISTRY:
        known = sorted(REGISTRY.keys())
        raise ValueError(
            f"Unknown metric recipe {name!r}. "
            f"Known registry names: {known}. "
            f"Pass a MetricRecipe or MetricCore constructor directly for "
            f"custom recipes that have not been registered."
        )
    return REGISTRY[name]
