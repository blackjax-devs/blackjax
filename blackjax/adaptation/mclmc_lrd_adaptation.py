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
"""Pilot-free (Scheme A) MCLMC adaptation with Low-Rank Diagonal preconditioning.

Overview
--------
This module implements the **Scheme A** warmup for MCLMC.  Phases 1–3 are
shared between the unadjusted and adjusted inner kernels; only Phase 4
switches:

1. **Pilot phase** — a single unadjusted MCLMC chain with diagonal
   preconditioning (via
   :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`)
   runs for ``pilot_num_warmup + pilot_num_samples`` steps to reach the
   typical set and collect geometry samples.

2. **LRD extraction** — the pilot draws are standardised and a thin SVD
   extracts the top-*k* principal directions of the correlation structure.
   The result is a
   :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` ``(sigma, U,
   lam)``.

   **Rank guard (non-negotiable):** before extraction, the effective sample
   size of the pilot chain is estimated via
   :func:`~blackjax.diagnostics.effective_sample_size` (the Geyer
   monotone-sequence estimator, operating on the ravelled flat-parameter
   vector; one chain, ``pilot_num_samples`` draws; basis is *not* the
   az-bulk ESS).  A safe rank bound ``k_safe = floor(n_eff / 2)`` is
   computed, and the requested ``k`` is hard-clamped to ``k_safe`` when it
   would exceed it.  Without this guard, under-mixed pilots produce
   rank-deficient SVDs and the LRD metric degrades sampling quality.

3. **Unadjusted LRD tuning** —
   :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
   is re-run with the LRD metric kernel
   (``diagonal_preconditioning=False``), so step size and trajectory length
   *L* are calibrated to the true posterior geometry.

4. **Inner-kernel dispatch** (controlled by ``inner_kernel``):

   * ``"mclmc"`` *(default)*: returns Phase-3 ``(L, step_size)`` directly.

   * ``"adjusted_mclmc"`` *(experimental)*: warm-starts
     :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
     from the unadjusted LRD-tuned parameters.  **Two hard constraints are
     enforced automatically** to avoid known failure modes:

     - ``params`` is set to ``MCLMCAdaptationState(L=L_init, step_size=...,
       inverse_mass_matrix=lrd_imm)`` so that the sqrt(dim) default L
       initialisation (which ignores the baked-in LRD metric) is never
       used.
     - ``frac_tune2=0.0`` disables the variance-based *L* estimator, which
       computes original-space ``trace(Σ)`` and is incompatible with an
       externally-baked LRD IMM.

     ``L_init`` is floored at ``floor_factor * step_unadj`` (default
     ``floor_factor=1.15``) to prevent degenerate one-step trajectories.

Stan-window analogy
-------------------
The staging mirrors Stan's two-phase warmup: a cheap diagonal (fast) phase
reaches the typical set first; the LRD (slow) phase then builds the metric
from draws that are already in the right region.  The key difference is that
Scheme A uses MCLMC (unadjusted, leapfrog-free) for the pilot, eliminating
the NUTS pilot cost that dominated total gradient expenditure in our
benchmarks:

* ill_cond_50 (d=50, κ=1000): k=40, n_pilot=10k → 101.1 % oracle step-size
  recovery, 2.42× ESS/total-grad vs NUTS-pilot, 72.5 % grad savings.
* german_credit (d=26): k=8, n_pilot=5k.
* mvn_10 (d=10): k=4, n_pilot=2k.

Limitations
-----------
* **Single-chain pilot.** The pilot is single-chain; overdispersed
  multi-chain initialisation would give stronger geometry coverage and
  tighter n_eff estimates.  This is a known limitation for the rank guard's
  tightness.
* **NUTS-pilot fallback.** When the target has funnel-class geometry (e.g.
  hierarchical models near unit-root), the MCLMC pilot may under-mix even
  with n_eff/2 clamping.  A NUTS-pilot fallback (replacing Phase 1 with
  :func:`~blackjax.adaptation.window_adaptation.window_adaptation`) is the
  recommended escape hatch but is out of scope here.
* **Adjusted path experimental.** The ``inner_kernel="adjusted_mclmc"``
  branch is certified on german_credit but has not yet been validated across
  a broad benchmark suite.  ill_cond_50 evidence is ongoing.  Treat as
  experimental; the unadjusted default is the stable path.
"""

import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.adjusted_mclmc as _adj_mclmc_mod
import blackjax.mcmc.mclmc as _mclmc_mod
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
)
from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

__all__ = [
    "MCLMCLRDAdaptationState",
    "mclmc_lrd_adaptation",
]

_VALID_INNER_KERNELS = frozenset({"mclmc", "adjusted_mclmc"})


class MCLMCLRDAdaptationState(NamedTuple):
    """Result of :func:`mclmc_lrd_adaptation`.

    L
        Adapted momentum decoherence length from the final tuning phase.
    step_size
        Adapted step size from the final tuning phase.
    inverse_mass_matrix
        The adapted LRD inverse mass matrix as a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple
        ``(sigma, U, lam)``.
    diagnostics
        A plain dict with provenance fields:

        ``inner_kernel``
            Which inner kernel was used (``"mclmc"`` or
            ``"adjusted_mclmc"``).
        ``n_eff``
            Effective sample size of the pilot chain (Geyer monotone-sequence
            estimator, blackjax basis, ravelled flat vector).
        ``k_safe``
            Floor of ``n_eff / 2`` — the maximum rank that is statistically
            supported by the pilot.
        ``k_used``
            The rank actually passed to SVD after clamping (``min(k, k_safe)``).
        ``pilot_num_grad_evals``
            Total gradient evaluations consumed by the pilot phase
            (warmup + samples; unadjusted MCLMC costs 2 grads/step).
    """

    L: float
    step_size: float
    inverse_mass_matrix: LowRankInverseMassMatrix
    diagnostics: dict


def _extract_lrd_from_samples(
    flat_positions: Any,
    k: int,
) -> tuple:
    """Extract LRD parameters from a ``(n_samples, d)`` array of pilot draws.

    Parameters
    ----------
    flat_positions
        Shape ``(n_samples, d)``.  Already ravelled to a 2-D float array.
    k
        Rank of the LRD approximation (after clamping by the caller).

    Returns
    -------
    sigma : shape ``(d,)``
    U     : shape ``(d, k)``
    lam   : shape ``(k,)``
    """
    mean = jnp.mean(flat_positions, axis=0)  # (d,)
    sigma = jnp.std(flat_positions, axis=0)  # (d,)
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)  # avoid div-by-zero in zero-var dims

    standardised = (flat_positions - mean[None, :]) / sigma[None, :]  # (n, d)
    n = flat_positions.shape[0]

    # Thin SVD: columns of V are the principal directions of standardised draws.
    # Eigenvalues of the sample correlation matrix: lam_i = s_i^2 / n.
    _, S, Vt = jnp.linalg.svd(standardised, full_matrices=False)
    V = Vt.T  # (d, min(n,d))
    lam = (S**2) / n  # (min(n,d),)

    # Select top-k by |λ - 1| (directions that deviate most from isotropic).
    sort_idx = jnp.argsort(jnp.abs(lam - 1.0))[::-1]
    top_idx = sort_idx[:k]

    lam_k = lam[top_idx]  # (k,)
    U_k = V[:, top_idx]  # (d, k)
    return sigma, U_k, lam_k


def mclmc_lrd_adaptation(
    logdensity_fn,
    position,
    rng_key,
    *,
    k: int = 10,
    pilot_num_warmup: int = 1000,
    pilot_num_samples: int = 5000,
    lrd_num_steps: int = 1000,
    inner_kernel: str = "mclmc",
    floor_factor: float = 1.15,
    adjusted_target: float = 0.9,
):
    """Scheme A (pilot-free) MCLMC adaptation with Low-Rank Diagonal preconditioning.

    Runs a cheap diagonal unadjusted MCLMC pilot to reach the typical set and
    collect geometry samples, extracts a low-rank diagonal (LRD) inverse mass
    matrix via thin SVD, then tunes step size and trajectory length L against
    the LRD metric kernel.  The inner kernel for the final tuning phase is
    controlled by ``inner_kernel``.

    Parameters
    ----------
    logdensity_fn
        Log-density of the target distribution.
    position
        Initial position (any pytree).
    rng_key
        JAX PRNG key.
    k
        Requested LRD rank.  Hard-clamped to ``floor(n_eff / 2)`` if the
        pilot chain is under-mixed; see the rank-guard note in the module
        docstring.
    pilot_num_warmup
        Number of steps for the diagonal pilot warmup phase (used by
        :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
        to adapt step size and L with a diagonal mass matrix).
    pilot_num_samples
        Number of unadjusted MCLMC draws collected *after* the pilot warmup,
        used for the SVD geometry estimate.
    lrd_num_steps
        Number of steps passed to the LRD tuning call(s) in Phase 3 (and
        Phase 4 for the adjusted path).
    inner_kernel
        Which inner kernel to use for the final tuning phase.  One of:

        * ``"mclmc"`` *(default, stable)*: unadjusted MCLMC.  Phase 3
          output ``(L, step_size)`` is returned directly.
        * ``"adjusted_mclmc"`` *(experimental)*: after Phase 3 unadjusted
          LRD tuning, warms-start
          :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
          with ``frac_tune2=0.0`` (variance-based *L* estimator disabled)
          and ``diagonal_preconditioning=False``.

    floor_factor
        For ``inner_kernel="adjusted_mclmc"`` only: the L initialisation
        floor is ``max(L_unadj, floor_factor * step_unadj)``.  Default
        ``1.15``; increase (e.g. to ``1.5``) for high-condition-number
        targets where the unadjusted trajectory length may be too short for
        the adjusted integrator.  Ignored when ``inner_kernel="mclmc"``.
    adjusted_target
        Target acceptance rate for the adjusted MCLMC tuning phase.
        Default ``0.9``.  Ignored when ``inner_kernel="mclmc"``.

    Returns
    -------
    MCLMCLRDAdaptationState
        A :class:`MCLMCLRDAdaptationState` NamedTuple with fields ``L``,
        ``step_size``, ``inverse_mass_matrix`` (a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`), and
        ``diagnostics`` (``inner_kernel``, ``n_eff``, ``k_safe``,
        ``k_used``, ``pilot_num_grad_evals``).

    Raises
    ------
    ValueError
        If ``inner_kernel`` is not one of ``"mclmc"`` or
        ``"adjusted_mclmc"``.

    Examples
    --------
    .. code-block:: python

        import jax
        import jax.numpy as jnp
        import blackjax
        from blackjax.adaptation.mclmc_lrd_adaptation import mclmc_lrd_adaptation

        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        position = jnp.zeros(10)
        rng_key = jax.random.key(42)

        # Unadjusted (stable default):
        result = mclmc_lrd_adaptation(
            logdensity_fn, position, rng_key,
            k=4, pilot_num_warmup=500, pilot_num_samples=2000,
            lrd_num_steps=1000,
        )

        # Adjusted (experimental):
        result_adj = mclmc_lrd_adaptation(
            logdensity_fn, position, rng_key,
            k=4, pilot_num_warmup=500, pilot_num_samples=2000,
            lrd_num_steps=1000, inner_kernel="adjusted_mclmc",
        )

        # Build the production kernel from the unadjusted result:
        import blackjax.mcmc.mclmc as mclmc_mod
        lrd_imm = result.inverse_mass_matrix
        kernel = mclmc_mod.build_kernel()
        init_state = mclmc_mod.init(position, logdensity_fn, jax.random.key(0))
        next_state, info = kernel(
            jax.random.key(1), init_state, logdensity_fn,
            lrd_imm, result.L, result.step_size,
        )
    """
    if inner_kernel not in _VALID_INNER_KERNELS:
        raise ValueError(
            f"inner_kernel must be one of {sorted(_VALID_INNER_KERNELS)!r}, "
            f"got {inner_kernel!r}."
        )

    # Five independent keys — no reuse across init / pilot warmup / pilot
    # sampling / LRD tuning / adjusted tuning.
    init_key, warmup_key, sample_key, lrd_key, adj_key = jax.random.split(rng_key, 5)

    # ------------------------------------------------------------------
    # Phase 1: diagonal pilot — reach typical set + collect geometry samples
    # ------------------------------------------------------------------
    base_kernel = _mclmc_mod.build_kernel()
    init_state = _mclmc_mod.init(position, logdensity_fn, init_key)

    state_after_warmup, pilot_params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=base_kernel,
        num_steps=pilot_num_warmup,
        state=init_state,
        rng_key=warmup_key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=True,
    )

    # Collect pilot_num_samples draws with the adapted diagonal kernel.
    def _pilot_step(state, key):
        next_state, _ = base_kernel(
            rng_key=key,
            state=state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=pilot_params.inverse_mass_matrix,
            L=pilot_params.L,
            step_size=pilot_params.step_size,
        )
        return next_state, next_state.position

    _, pilot_positions = jax.lax.scan(
        _pilot_step,
        state_after_warmup,
        jax.random.split(sample_key, pilot_num_samples),
    )

    # ------------------------------------------------------------------
    # Rank guard: n_eff via blackjax Geyer ESS on ravelled flat vector.
    # ESS basis: Geyer monotone-sequence estimator (blackjax.diagnostics.
    # effective_sample_size), single chain, pilot_num_samples draws.
    # Shape expected: (chains, draws, d) — we add the chains axis here.
    # ------------------------------------------------------------------
    flat_pilot = jax.vmap(lambda p: ravel_pytree(p)[0])(pilot_positions)  # (n, d)

    # effective_sample_size expects (chains, draws, d); add singleton chain axis.
    # Requires ≥2 draws; guard for degenerate pilots.
    if pilot_num_samples >= 2:
        ess_per_dim = effective_sample_size(flat_pilot[None, :, :])  # (d,)
        n_eff = float(jnp.min(ess_per_dim))  # conservative: use min over dims
    else:
        n_eff = 0.0  # degenerate: force k_safe=0 → k_used=1

    k_safe = int(n_eff / 2)  # floor(n_eff / 2)
    k_used = min(k, max(k_safe, 1))  # clamp; always use at least rank 1

    if k_used < k:
        n_eff_rounded = round(n_eff, 1)
        warnings.warn(
            f"mclmc_lrd_adaptation: requested k={k} exceeds the rank-safety "
            f"bound k_safe=floor(n_eff/2)={k_safe} "
            f"(n_eff={n_eff_rounded} from {pilot_num_samples} pilot draws). "
            f"Clamping to k_used={k_used}. "
            "Increase pilot_num_samples to raise n_eff, or reduce k.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Phase 2: SVD extraction → LowRankInverseMassMatrix
    # ------------------------------------------------------------------
    sigma, U_k, lam_k = _extract_lrd_from_samples(flat_pilot, k=k_used)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U_k, lam=lam_k)

    # ------------------------------------------------------------------
    # Phase 3: unadjusted LRD tuning — calibrate L and step_size with LRD
    # metric.  The kernel wrapper routes the inverse_mass_matrix argument
    # through lrd_imm so diagonal_preconditioning=False does not overwrite
    # it inside the tuner.
    # ------------------------------------------------------------------
    def lrd_kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):
        return base_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=lrd_imm,  # always route through LRD
            L=L,
            step_size=step_size,
        )

    # Warm-start from pilot L/step_size to skip unnecessary exploration.
    lrd_init_params = MCLMCAdaptationState(
        L=pilot_params.L,
        step_size=pilot_params.step_size,
        inverse_mass_matrix=pilot_params.inverse_mass_matrix,  # placeholder; overridden
    )
    _, lrd_params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=lrd_kernel,
        num_steps=lrd_num_steps,
        state=state_after_warmup,
        rng_key=lrd_key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=False,
        params=lrd_init_params,
    )

    # ------------------------------------------------------------------
    # Phase 4: inner-kernel dispatch
    # ------------------------------------------------------------------
    if inner_kernel == "mclmc":
        final_L = lrd_params.L
        final_step_size = lrd_params.step_size

    else:  # inner_kernel == "adjusted_mclmc"
        # Build the adjusted kernel wrapper — routes inverse_mass_matrix
        # through lrd_imm (same override pattern as Phase 3).
        adj_base_kernel = _adj_mclmc_mod.build_kernel()

        def adj_lrd_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            integration_steps_params,
            inverse_mass_matrix,
        ):
            return adj_base_kernel(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logdensity_fn,
                step_size=step_size,
                integration_steps_params=integration_steps_params,
                inverse_mass_matrix=lrd_imm,  # always route through LRD
            )

        # L_init floor: prevents degenerate one-step trajectories.
        # Hard constraints enforced here (see module docstring):
        #   1) params != None  →  no sqrt(dim) default L init
        #   2) frac_tune2=0.0  →  variance-based L estimator disabled
        L_init = max(float(lrd_params.L), floor_factor * float(lrd_params.step_size))
        # Note: inverse_mass_matrix is a placeholder — adj_lrd_kernel always
        # routes through lrd_imm; using the diagonal pilot IMM keeps the field
        # consistent with MCLMCAdaptationState's existing usage (jnp array).
        adj_init_params = MCLMCAdaptationState(
            L=L_init,
            step_size=float(lrd_params.step_size),
            inverse_mass_matrix=pilot_params.inverse_mass_matrix,
        )

        # adjusted_mclmc.init takes (position, logdensity_fn) — no rng_key.
        adj_init_state = _adj_mclmc_mod.init(
            position=state_after_warmup.position,
            logdensity_fn=logdensity_fn,
        )

        _, adj_params, _ = adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=adj_lrd_kernel,
            logdensity_fn=logdensity_fn,
            num_steps=lrd_num_steps,
            state=adj_init_state,
            rng_key=adj_key,
            target=adjusted_target,
            frac_tune2=0.0,  # variance-based L estimator incompatible with baked-in LRD IMM
            diagonal_preconditioning=False,  # don't overwrite LRD IMM
            params=adj_init_params,
        )
        final_L = adj_params.L
        final_step_size = adj_params.step_size

    # Gradient accounting: unadjusted MCLMC costs 2 grads/step.
    pilot_num_grad_evals = (pilot_num_warmup + pilot_num_samples) * 2

    diagnostics = {
        "inner_kernel": inner_kernel,
        "n_eff": n_eff,
        "k_safe": k_safe,
        "k_used": k_used,
        "pilot_num_grad_evals": pilot_num_grad_evals,
    }

    return MCLMCLRDAdaptationState(
        L=final_L,
        step_size=final_step_size,
        inverse_mass_matrix=lrd_imm,
        diagnostics=diagnostics,
    )
