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
"""Post-warmup per-chain start-state safety (Issue#1064): stages 1 and 2a.

Stage 1 -- ensemble-calibrated typical-region gate (:func:`loo_state_gate`).
The multi-chain meta-adaptation controller already calibrates its ROUTE
decision from the ensemble (the between-chain magnitude/collinearity/
leave-one-out gates in :mod:`~blackjax.adaptation.meta._detection` decide
*whether the emitted metric should escalate*).  This module applies the
identical leave-one-out calibration principle one level down, to the
per-chain STATE the warmup hands off to sampling: each chain's post-warmup
potential energy (``-logdensity``, already cached on the warmup's final
``HMCState`` -- zero extra gradient cost) is compared against a reference
typical region built from the OTHER ``M-1`` chains only.  A chain whose
energy falls outside that region is flagged as having landed somewhere the
rest of the ensemble does not corroborate as typical under the emitted
products.  This is only possible because the routine is multi-chain -- the
ensemble supplies the reference distribution that a single chain cannot
supply for itself.  The gate is an empirical, operational stand-in for a
state-level calibration statistic; it is deliberately NOT the trigger for
:mod:`~blackjax.adaptation.staged_adaptation`'s reactive step-size-backoff
fallback (stage 2b there runs unconditionally regardless of this gate's
verdict), only for the proactive redraw in stage 2a below.

Stage 2a -- redraw near a healthy anchor (:func:`redraw_flagged_chains`).
A flagged chain's position is replaced by the ACTUAL position of the
non-flagged chain closest to ensemble consensus, plus a small
mass-matrix-consistent jitter (drawn through the emitted metric's own
``scale`` operator, so the perturbation respects the warmup's geometry
rather than an arbitrary isotropic nudge).  Anchoring on a genuine sampled
position (not a synthetic mean/median across chains) avoids landing outside
the typical set in high dimension, where averaging points is not guaranteed
to stay typical.

Both functions are pure (no kernel calls, no MCMC state threading) so they
can be tested and reasoned about independently of
:mod:`~blackjax.adaptation.staged_adaptation`, which owns stage 2b (the
per-chain probation scan) because that stage needs the built kernel and an
rng stream neither of these functions require.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from blackjax.mcmc.metrics import MetricTypes, default_metric
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "loo_logdensity_z_scores",
    "loo_state_gate",
    "select_healthy_anchor",
    "redraw_flagged_chains",
    "compute_chain_energies",
]


def loo_logdensity_z_scores(logdensities: Array) -> Array:
    """Leave-one-out z-score of each chain's logdensity vs. the other M-1 chains.

    For chain ``i``, the reference mean/std is computed from the ``M-1``
    OTHER chains only (never including chain ``i`` itself) -- the same
    leave-one-out convention used by
    :func:`~blackjax.adaptation.meta._detection._loo_detection_passes` for
    the route-level between-chain gate.  Excluding the chain under test from
    its own reference avoids the "swamping" failure mode where a genuine
    outlier inflates the very statistic meant to detect it.

    Parameters
    ----------
    logdensities
        Shape ``(M,)``, one scalar per chain.

    Returns
    -------
    Array
        Shape ``(M,)`` z-scores.  Degenerate (not meaningful) for ``M < 4``;
        callers should gate on :data:`~blackjax.adaptation.meta._calibration._STATE_GATE_MIN_CHAINS`.
    """
    M = logdensities.shape[0]
    ld = logdensities.astype(jnp.float32)
    total = jnp.sum(ld)
    total_sq = jnp.sum(ld**2)
    loo_sum = total - ld
    loo_mean = loo_sum / jnp.float32(M - 1)
    loo_sumsq = total_sq - ld**2
    # Sample variance of the OTHER M-1 values (ddof=1 -> divide by M-2).
    loo_var = (loo_sumsq - (M - 1) * loo_mean**2) / jnp.float32(max(M - 2, 1))
    loo_std = jnp.sqrt(jnp.maximum(loo_var, jnp.float32(1e-12)))
    return (ld - loo_mean) / loo_std


def loo_state_gate(logdensities: Array, threshold: float, min_chains: int) -> Array:
    """Ensemble-calibrated typical-region gate (stage 1).

    Parameters
    ----------
    logdensities
        Shape ``(M,)`` post-warmup potential-energy proxy (``-logdensity``
        sign convention is irrelevant here; only relative spread matters).
    threshold
        Absolute leave-one-out z-score above which a chain is flagged.
    min_chains
        Below this many chains the leave-one-out reference is too noisy
        (dof = M-2 too small); the gate is skipped and no chain is flagged.

    Returns
    -------
    Array
        Boolean mask, shape ``(M,)``. ``True`` = outside the calibrated
        typical region.
    """
    M = logdensities.shape[0]
    if M < min_chains:
        return jnp.zeros((M,), dtype=jnp.bool_)
    z = loo_logdensity_z_scores(logdensities)
    return jnp.abs(z) > threshold


def select_healthy_anchor(logdensities: Array, flagged_mask: Array) -> Array:
    """Index of the non-flagged chain closest to ensemble consensus.

    Flagged chains are excluded from consideration (their |z| is treated as
    infinite for the purposes of selection).  If every chain happens to be
    flagged, falls back to the least-extreme chain (defensive; not expected
    in practice given the LOO construction).
    """
    z = loo_logdensity_z_scores(logdensities)
    z_for_selection = jnp.where(flagged_mask, jnp.inf, jnp.abs(z))
    return jnp.argmin(z_for_selection)


def redraw_flagged_chains(
    position: ArrayLikeTree,
    flagged_mask: Array,
    anchor_idx: Array,
    metric: MetricTypes,
    rng_key: PRNGKey,
    jitter_scale: float,
) -> ArrayLikeTree:
    """Replace flagged chains' positions with a jittered copy of the anchor's (stage 2a).

    Works uniformly for a bare flat-array position or a structured PyTree
    position (e.g. a dict of per-site arrays), matching every other position
    shape blackjax supports elsewhere in the multi-chain path.

    Parameters
    ----------
    position
        PyTree with a leading ``(M, ...)`` per-chain batch dimension on every
        leaf.
    flagged_mask
        Shape ``(M,)`` boolean; ``True`` chains get replaced.
    anchor_idx
        Scalar index of the chain to anchor the redraw on (see
        :func:`select_healthy_anchor`).
    metric
        The emitted shared metric (e.g. a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`); the
        jitter is drawn through this metric's ``scale`` operator so the
        perturbation is mass-matrix-consistent rather than an arbitrary
        isotropic nudge.
    rng_key
        RNG key for the per-chain jitter draws.
    jitter_scale
        Fraction of the metric's scale used for the jitter (see
        :data:`~blackjax.adaptation.meta._calibration._STATE_GATE_REDRAW_JITTER_SCALE`).

    Returns
    -------
    ArrayLikeTree
        Same PyTree structure and leading ``(M, ...)`` shape as ``position``.
    """
    M = flagged_mask.shape[0]
    anchor_position = jax.tree.map(lambda x: x[anchor_idx], position)
    metric_obj = default_metric(metric)
    chain_keys = jax.random.split(rng_key, M)
    noise = jax.vmap(lambda k: generate_gaussian_noise(k, anchor_position))(chain_keys)
    jitter = jax.vmap(
        lambda n: metric_obj.scale(anchor_position, n, inv=False, trans=False)
    )(noise)

    def _select(orig_leaf, anchor_leaf, jitter_leaf):
        redrawn_leaf = anchor_leaf[None, ...] + jitter_scale * jitter_leaf
        mask_shape = (M,) + (1,) * (orig_leaf.ndim - 1)
        mask_b = flagged_mask.reshape(mask_shape)
        return jnp.where(mask_b, redrawn_leaf, orig_leaf)

    return jax.tree.map(_select, position, anchor_position, jitter)


def compute_chain_energies(
    position: ArrayLikeTree,
    logdensity: Array,
    metric: MetricTypes,
    rng_key: PRNGKey,
) -> Array:
    """Per-chain Hamiltonian energy ``-logdensity + kinetic(fresh momentum)``.

    NOT a gate trigger -- an OBSERVABLE recorded in the probation
    diagnostics (per TL amendment A) so the ensemble's energy spread is
    visible for future calibration work, presenting the empirical LOO probe
    above as one operational form of a broader state-level calibration idea
    (the same principle the controller already applies at the route level,
    one level down: route -> parameter -> state).  Momentum is freshly
    resampled per call, which is why this is an observable rather than a
    trigger: the LOO gate uses the noise-free potential energy already
    cached on the warmup state, not this stochastic quantity.

    Parameters
    ----------
    position
        PyTree with leading ``(M, ...)`` batch dimension.
    logdensity
        Shape ``(M,)``.
    metric
        The emitted shared metric.
    rng_key
        RNG key for the per-chain momentum draw.

    Returns
    -------
    Array
        Shape ``(M,)`` energies.
    """
    M = logdensity.shape[0]
    metric_obj = default_metric(metric)
    chain_keys = jax.random.split(rng_key, M)
    momentum = jax.vmap(metric_obj.sample_momentum)(chain_keys, position)
    kinetic = jax.vmap(metric_obj.kinetic_energy)(momentum)
    return -logdensity + kinetic
