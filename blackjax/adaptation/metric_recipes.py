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
- Step-size proxy independence: for diag/dense representations (this slice),
  the dual-averaging acceptance-rate proxy is not an eigenvalue proxy of the
  adapted metric, so the step-size and metric adaptation are decoupled.  A
  ``diag_reference`` accessor will be added in the low-rank slice where this
  independence no longer holds.

Registry (this slice):

- ``"welford_diag"`` — Stan-default; reproduces :func:`window_adaptation`
  exactly.
- ``"welford_dense"`` — dense covariance, same Stan schedule.
- ``"fisher_diag"`` — Fisher-divergence-minimising diagonal estimator;
  situational; see registry provenance note for operational guidance.

Usage::

    # String sugar (registry lookup):
    wu = staged_adaptation(nuts, logdensity_fn, metric="welford_diag")

    # Recipe constructor (testability / custom configuration):
    from blackjax.adaptation.metric_recipes import REGISTRY
    core = REGISTRY["welford_diag"].build_core(imm_shrinkage_to_previous=5.0)
    wu = staged_adaptation(nuts, logdensity_fn, metric=core)

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

import jax.numpy as jnp

from blackjax.adaptation.mass_matrix import (
    FisherMassMatrixAdaptationState,
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.adaptation.metric_estimators import fisher_score_diagonal_from_moments
from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "MetricCore",
    "MetricRecipe",
    "REGISTRY",
    "lookup_recipe",
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
    ) -> MetricCore:
        """Build an embeddable :class:`MetricCore` from this recipe.

        Parameters
        ----------
        imm_shrinkage_to_previous
            Pseudo-count controlling shrinkage of the per-window IMM toward the
            previous window's IMM.  Default ``0.0`` (Stan vanilla, no
            persistence).  Forwarded to
            :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`.
            Not supported for ``"fisher_diag"`` (``ValueError`` there per
            ``mass_matrix_adaptation``'s validation).
        initial_inverse_mass_matrix
            Optional seed array for the initial inverse mass matrix.  ``None``
            (default) uses the standard identity initialisation (``ones(d)``
            for diagonal, ``identity(d)`` for dense).

        Returns
        -------
        MetricCore
            Embeddable init/update/final bundle ready for the engine.

        Raises
        ------
        ValueError
            If the estimator tag is not supported by this slice.
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
        else:
            raise ValueError(
                f"Unknown estimator tag {self.estimator!r} in MetricRecipe.build_core. "
                f"Slice-1 supports 'welford' and 'fisher_diag'."
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
        # formula — bit-exact vs the current window_adaptation.base() path.
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

    .. note::

        **Twin implementation alert (stitch #1 of 2).**  The Fisher IMM
        computation + block-reset stitch in ``final()`` is also present in
        :func:`~blackjax.adaptation.window_adaptation.base`'s ``slow_final``
        closure (fisher path).  These two copies must stay in sync until
        ``base()``'s disposition is decided (retire or rewire through the
        engine).  See the comment at the matching site in
        ``window_adaptation.base``.

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

    def final(state: FisherMassMatrixAdaptationState) -> FisherMassMatrixAdaptationState:
        # Stitch #1 of 2: Fisher IMM computation + block reset.
        # Twin copy: window_adaptation.base()'s slow_final (fisher path).
        # Consolidation deferred until base() disposition is decided.
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
            "Fisher-divergence-minimising diagonal estimator "
            "(Seyboldt 2026, arXiv:2603.18845, §3.1). "
            "Inverse mass matrix: sigma^2 = sqrt(Var[x] / Var[grad log p]) per coordinate. "
            "Situational benefit for concentrated-anisotropy hierarchical geometry "
            "(~1.3-1.7x min-ESS/grad improvement measured 2026-07-13); "
            "degrades on well-conditioned GLM and diffuse AR(1) geometry "
            "(~0.6-0.7x). Not a default. "
            "Operational warning: degradation carries no R-hat/divergence signature "
            "(healthy flags while effective-samples-per-gradient halves) — gate "
            "estimator swaps on ESS-per-gradient, not health flags."
        ),
    ),
}


def lookup_recipe(name: str) -> MetricRecipe:
    """Look up a named recipe from the :data:`REGISTRY`.

    Parameters
    ----------
    name
        Registry key.  Current slice-1 names:

        - ``"welford_diag"`` (default; reproduces ``window_adaptation`` exactly)
        - ``"welford_dense"``
        - ``"fisher_diag"``

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
            f"custom recipes not yet in the registry."
        )
    return REGISTRY[name]
