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
"""Pilot-free (Scheme A) MCLMC warmup with Low-Rank Diagonal preconditioning.

Overview
--------
This module implements the **Scheme A** warmup for MCLMC via
:func:`mclmc_lrd_warmup`.  Phases 1–3 are shared between the unadjusted and
adjusted inner kernels; only Phase 4 switches:

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
   rank-deficient SVDs and the LRD metric degrades sampling quality.  In the
   extreme under-mix regime (n_eff < 5), the Geyer estimator may exceed
   az-bulk ESS on the minimum-ESS dimension; ``k_safe`` is still floored to
   0 → ``k_used=1``, so practical rank clamping is unaffected.

3. **Multi-chain unadjusted LRD tuning** —
   :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
   is run in parallel over ``num_chains`` independent chains (all starting
   from the pilot's final position, with per-chain PRNG keys) with the LRD
   metric kernel (``diagonal_preconditioning=False``).  The per-chain
   ``(L, step_size)`` values are averaged to give stable estimates of the
   trajectory length and step size in LRD geometry.

4. **Inner-kernel dispatch** (controlled by ``inner_kernel``):

   * ``"mclmc"`` *(default)*: returns the Phase-3 mean
     ``(L, step_size)`` directly.

   * ``"adjusted_mclmc"`` *(experimental)*: warm-starts
     :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
     across ``num_chains`` parallel chains from the Phase-3 mean parameters.
     **Two hard constraints are enforced automatically** to avoid known
     failure modes:

     - ``params`` is set to
       ``MCLMCAdaptationState(L=L_init, step_size=..., inverse_mass_matrix=lrd_imm)``
       so that the ``sqrt(dim)`` default L initialisation (which ignores the
       baked-in LRD metric) is never used.
     - ``frac_tune2=0.0`` disables the variance-based *L* estimator, which
       computes original-space ``trace(Σ)`` and is incompatible with an
       externally-baked LRD IMM.

     ``L_init`` is floored at ``floor_factor * step_unadj`` (default
     ``floor_factor=1.15``) to prevent the Dual-Averaging ceiling
     ``(L / 1.1)`` from binding below the oracle step size.  For geometry
     where the oracle step size exceeds the oracle L (stiff
     high-condition-number targets), the default 1.15 may leave the DA
     ceiling binding; raise to approximately 1.5 and set
     ``adjusted_num_steps`` to at least 5000 for those targets.

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
* **Adjusted path experimental.** ``inner_kernel="adjusted_mclmc"`` is
  certified 3/3 on german_credit at the default recipe (4-chain phase 3,
  ``frac_tune2=0``, ``floor_factor=1.15``, ``adjusted_num_steps=3000``).
  ill_cond_50 (stiff, κ=1000) is geometrically compatible (clamp-free at
  ``floor_factor=1.5``, step size 104–114% of oracle, zero DA divergences)
  but was NOT certified at ``adjusted_num_steps=3000`` (marginal R-hat
  1.010–1.011; DA not converged).  For stiff geometry, raise
  ``floor_factor`` to ~1.5 and ``adjusted_num_steps`` to ≥5000.
  The unadjusted default remains the stable, broadly validated path.
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
    "mclmc_lrd_warmup",
]

_VALID_INNER_KERNELS = frozenset({"mclmc", "adjusted_mclmc"})


class MCLMCLRDAdaptationState(NamedTuple):
    """Result of :func:`mclmc_lrd_warmup`.

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
        ``pilot_L``
            Trajectory length L adapted during the diagonal pilot warmup.
        ``pilot_step_size``
            Step size adapted during the diagonal pilot warmup.
        ``lrd_L``
            Mean trajectory length L across ``num_chains`` chains after
            Phase-3 unadjusted LRD tuning.
        ``lrd_step_size``
            Mean step size across ``num_chains`` chains after Phase-3
            unadjusted LRD tuning.
        ``L_init``
            *(adjusted path only)* The floor-guarded L initialisation value
            passed to the adjusted warm-start: ``max(lrd_L, floor_factor *
            lrd_step_size)``.  This is the value whose ``/ 1.1`` sets the DA
            ceiling for step-size tuning.
        ``floor_active``
            *(adjusted path only)* ``True`` when the floor guard was
            triggered (``floor_factor * lrd_step_size > lrd_L``), i.e. when
            ``L_init`` was raised above the unadjusted mean.
        ``N_sample``
            *(adjusted path only)* Effective number of leapfrog steps per
            trajectory at the final adapted parameters:
            ``round(L_init / final_step_size)``.  Provided as a bookkeeping
            aid for cert integration.
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


def _check_da_ceiling_warning(
    final_step_size: float,
    L_init: float,
    floor_factor: float,
) -> None:
    """Emit a UserWarning when the adapted step_size is at or near L_init/1.1.

    This is the Dual-Averaging (DA) ceiling signature: when
    ``step_size >= L_init / 1.1 * 0.999``, the step-size tuner may have been
    constrained by the ceiling rather than converged to the optimal value.

    Parameters
    ----------
    final_step_size
        Mean adapted step size across chains from the adjusted Phase-4 tuning.
    L_init
        The floor-guarded L_init value used for the adjusted warm-start.
    floor_factor
        Current ``floor_factor`` value; included in the warning message.
    """
    da_clamp = L_init / 1.1
    step_ratio = final_step_size / da_clamp
    if step_ratio >= 0.999:
        step_s = round(final_step_size, 4)
        clamp_s = round(da_clamp, 4)
        ratio_s = round(step_ratio, 3)
        warnings.warn(
            f"mclmc_lrd_warmup (adjusted_mclmc path): adapted step_size "
            f"({step_s}) is at or near the DA ceiling "
            f"L_init/1.1={clamp_s} (ratio={ratio_s}). "
            "The step-size tuner may have been constrained rather than "
            "converged. Consider raising `floor_factor` "
            f"(current value: {floor_factor}) — e.g. to 1.5 for "
            "high-condition-number targets.",
            UserWarning,
            stacklevel=3,  # points to caller of mclmc_lrd_warmup
        )


def mclmc_lrd_warmup(
    logdensity_fn,
    position,
    rng_key,
    *,
    k: int = 10,
    pilot_num_warmup: int = 1000,
    pilot_num_samples: int = 5000,
    lrd_num_steps: int = 1000,
    num_chains: int = 4,
    inner_kernel: str = "mclmc",
    floor_factor: float = 1.15,
    adjusted_num_steps: int = 3000,
    adjusted_target: float = 0.9,
):
    """Scheme A (pilot-free) MCLMC warmup with Low-Rank Diagonal preconditioning.

    Runs a cheap diagonal unadjusted MCLMC pilot to reach the typical set and
    collect geometry samples, extracts a low-rank diagonal (LRD) inverse mass
    matrix via thin SVD, then calibrates step size and trajectory length L
    across ``num_chains`` parallel chains with the LRD metric kernel.  The
    inner kernel for the final tuning phase is controlled by ``inner_kernel``.

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
    num_chains
        Number of parallel chains for Phase 3 (unadjusted LRD tuning) and
        Phase 4 (adjusted tuning when ``inner_kernel="adjusted_mclmc"``).
        All chains start from the pilot's final position; per-chain PRNG keys
        give independent trajectories.  Per-chain L and step_size are averaged
        for a stable multi-chain estimate.  Default ``4``.
    inner_kernel
        Which inner kernel to use for the final tuning phase.  One of:

        * ``"mclmc"`` *(default, stable)*: unadjusted MCLMC.  Phase-3
          mean ``(L, step_size)`` is returned directly.
        * ``"adjusted_mclmc"`` *(experimental)*: after Phase-3 unadjusted
          LRD tuning, warm-starts
          :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
          across ``num_chains`` chains with ``frac_tune2=0.0``
          (variance-based *L* estimator disabled) and
          ``diagonal_preconditioning=False``.

    floor_factor
        For ``inner_kernel="adjusted_mclmc"`` only: the L initialisation
        floor is ``max(L_unadj, floor_factor * step_unadj)``.  Default
        ``1.15``; certified on german_credit.  For stiff geometry where the
        oracle step size exceeds the oracle L, raise to approximately 1.5
        (the default leaves the DA ceiling ``L / 1.1`` binding for those
        targets).  Ignored when ``inner_kernel="mclmc"``.
    adjusted_num_steps
        Number of DA tuning steps for the adjusted Phase-4 path.  Default
        ``3000`` with ``frac_tune1=0.5`` → 1500 DA steps; the certified
        recipe for german_credit.  For stiff high-κ geometry, increase to
        at least 5000.  Ignored when ``inner_kernel="mclmc"``.
    adjusted_target
        Target acceptance rate for the adjusted MCLMC tuning phase.
        Default ``0.9``.  Ignored when ``inner_kernel="mclmc"``.

    Returns
    -------
    MCLMCLRDAdaptationState
        A :class:`MCLMCLRDAdaptationState` NamedTuple with fields ``L``,
        ``step_size``, ``inverse_mass_matrix`` (a
        :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`), and
        ``diagnostics`` (see :class:`MCLMCLRDAdaptationState` for keys).

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

        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        position = jnp.zeros(10)
        rng_key = jax.random.key(42)

        # Unadjusted (stable default):
        result = blackjax.mclmc_lrd_warmup(
            logdensity_fn, position, rng_key,
            k=4, pilot_num_warmup=500, pilot_num_samples=2000,
            lrd_num_steps=1000, num_chains=4,
        )

        # Adjusted (experimental):
        result_adj = blackjax.mclmc_lrd_warmup(
            logdensity_fn, position, rng_key,
            k=4, pilot_num_warmup=500, pilot_num_samples=2000,
            lrd_num_steps=1000, num_chains=4,
            inner_kernel="adjusted_mclmc",
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
    init_key, warmup_key, sample_key, lrd_subkey, adj_subkey = jax.random.split(
        rng_key, 5
    )

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

    pilot_L = float(pilot_params.L)
    pilot_step_size_val = float(pilot_params.step_size)

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
            f"mclmc_lrd_warmup: requested k={k} exceeds the rank-safety "
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
    # Phase 3: multi-chain unadjusted LRD tuning
    # num_chains independent chains, all starting from pilot's final position
    # with per-chain PRNG keys for independent trajectories.  L and step_size
    # are averaged across chains for a stable multi-chain estimate.
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

    # Split into 2*num_chains keys: first half for chain init, second for tuning.
    lrd_all_keys = jax.random.split(lrd_subkey, 2 * num_chains)
    lrd_init_keys = lrd_all_keys[:num_chains]
    lrd_tune_keys = lrd_all_keys[num_chains:]

    # Replicate position for all chains (same start, different momenta via keys).
    chain_positions = jax.tree.map(
        lambda x: jnp.stack([x] * num_chains),
        state_after_warmup.position,
    )

    @jax.vmap
    def _lrd_init_one(k, x0):
        return _mclmc_mod.init(x0, logdensity_fn, k)

    lrd_init_states = _lrd_init_one(lrd_init_keys, chain_positions)

    @jax.vmap
    def _lrd_tune_one(k, state):
        _, params, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=lrd_kernel,
            num_steps=lrd_num_steps,
            state=state,
            rng_key=k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
            params=lrd_init_params,
        )
        return params

    lrd_params_all = _lrd_tune_one(lrd_tune_keys, lrd_init_states)

    # Multi-chain mean for stable L and step_size estimates.
    lrd_L = float(jnp.mean(lrd_params_all.L))
    lrd_step_size = float(jnp.mean(lrd_params_all.step_size))

    # ------------------------------------------------------------------
    # Phase 4: inner-kernel dispatch
    # ------------------------------------------------------------------
    if inner_kernel == "mclmc":
        final_L = jnp.array(lrd_L)
        final_step_size = jnp.array(lrd_step_size)

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

        # L_init floor guard: prevents DA ceiling from binding below oracle.
        # Hard constraints enforced here (see module docstring):
        #   C1) params != None  →  no sqrt(dim) default L init
        #   C2) frac_tune2=0.0  →  variance-based L estimator disabled
        L_floor = floor_factor * lrd_step_size
        floor_active = bool(L_floor > lrd_L)
        L_init = float(max(lrd_L, L_floor))

        # Note: inverse_mass_matrix is a placeholder in adj_init_params —
        # adj_lrd_kernel always routes through lrd_imm; using the diagonal
        # pilot IMM keeps the field consistent with MCLMCAdaptationState's
        # existing dtype contract (jnp array).
        adj_init_params = MCLMCAdaptationState(
            L=jnp.array(L_init),
            step_size=jnp.array(lrd_step_size),
            inverse_mass_matrix=pilot_params.inverse_mass_matrix,
        )

        # Multi-chain adjusted tuning — same num_chains chains, starting
        # from the pilot's final position.
        adj_tune_keys = jax.random.split(adj_subkey, num_chains)

        @jax.vmap
        def _adj_init_one(x0):
            return _adj_mclmc_mod.init(x0, logdensity_fn)

        adj_init_states = _adj_init_one(chain_positions)

        @jax.vmap
        def _adj_tune_one(k, state):
            _, params, _ = adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=adj_lrd_kernel,
                logdensity_fn=logdensity_fn,
                num_steps=adjusted_num_steps,
                state=state,
                rng_key=k,
                target=adjusted_target,
                frac_tune1=0.5,  # certified recipe: 0.5 × adjusted_num_steps DA steps
                frac_tune2=0.0,  # REQUIRED: variance-based L estimator incompatible with LRD IMM
                diagonal_preconditioning=False,  # don't overwrite LRD IMM
                params=adj_init_params,
            )
            return params

        adj_params_all = _adj_tune_one(adj_tune_keys, adj_init_states)

        # frac_tune2=0.0 → L is fixed at L_init across all chains.
        # Average step_size across chains for a stable multi-chain estimate.
        final_step_size = jnp.mean(adj_params_all.step_size)
        final_L = jnp.array(L_init)  # L is fixed by construction

        # DA-ceiling diagnostic: warn if step_size is at or near L_init/1.1.
        _check_da_ceiling_warning(float(final_step_size), L_init, floor_factor)

    # Gradient accounting: unadjusted MCLMC costs 2 grads/step.
    pilot_num_grad_evals = (pilot_num_warmup + pilot_num_samples) * 2

    diagnostics = {
        "inner_kernel": inner_kernel,
        "n_eff": n_eff,
        "k_safe": k_safe,
        "k_used": k_used,
        "pilot_num_grad_evals": pilot_num_grad_evals,
        "pilot_L": pilot_L,
        "pilot_step_size": pilot_step_size_val,
        "lrd_L": lrd_L,
        "lrd_step_size": lrd_step_size,
    }

    # Adjusted-path-only provenance keys (available in scope only for that branch).
    if inner_kernel == "adjusted_mclmc":
        diagnostics["L_init"] = L_init
        diagnostics["floor_active"] = floor_active
        # N_sample = effective number of leapfrog steps per trajectory at the
        # final adapted params — the floor-guard inputs that tuningfork cert
        # integration uses to verify trajectory-length bookkeeping.
        diagnostics["N_sample"] = round(
            float(final_L) / max(float(final_step_size), 1e-10)
        )

    return MCLMCLRDAdaptationState(
        L=final_L,
        step_size=final_step_size,
        inverse_mass_matrix=lrd_imm,
        diagnostics=diagnostics,
    )
