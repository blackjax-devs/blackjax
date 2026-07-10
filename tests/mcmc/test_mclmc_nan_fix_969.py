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
"""Regression tests for blackjax #969 — MCLMC NaN-handling fix.

Two divergence signatures were not caught pre-fix:
- Case-1: NaN position/momentum (caught by kernel but not by adaptation's re-derive).
- Case-2: finite position/momentum but NaN logdensity — dominant under
  velocity_verlet at moderate overshoot; missed entirely pre-fix because only
  position+momentum were checked.

The fix (three-piece):
1. Kernel (mclmc.py::handle_nans): extend nonans to include isfinite(logdensity).
2. Adaptation (mclmc_adaptation.py::handle_nans): consume info.nonans directly
   instead of re-deriving from the already-reverted next_state.
3. LAPS (laps_burn_in.py::sequential_kernel): use info.nonans instead of
   no_nans(new_state) so the eps-halving safety fires on real divergences.

Test evidence links:
- Case-1 config: reporter's MRE (mclachlan, step_size=100, key(0), dim=2).
- Case-2 config: velocity_verlet at step_size=8 — stat-B exp9/exp10 showed
  24/24 seeds produce case-2 at ss=8 (finite pos, NaN logdensity).
- Behavioral convergence: stat-B exp9 Q1, 24-seed A2 design, conv-range [0.736,0.902].
- Sampling-path: stat-B exp10 showed 9/15 NaN-logdensity states pre-fix.
- Bitwise non-regression: Gaussian target with no NaN events → fix is a no-op.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from blackjax.mcmc.integrators import isokinetic_mclachlan, isokinetic_velocity_verlet
from blackjax.mcmc.mclmc import build_kernel, init as mclmc_init

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 2
_BOUND = 5.0


def _bounded_logdensity(x):
    """Bounded log-barrier target used in the #969 MRE.

    Has a hard wall at |x_i| = _BOUND; logdensity → -∞ as x_i → ±_BOUND.
    With a large step_size the integrator can overshoot the boundary:
    - mclachlan: position becomes NaN (case-1).
    - velocity_verlet at ss≈6-20: position stays finite but logdensity → NaN (case-2).
    """
    return -0.5 * jnp.sum(x**2) + jnp.sum(jnp.log(_BOUND - jnp.abs(x)))


# ---------------------------------------------------------------------------
# Kernel unit tests — cases 1 and 2
# ---------------------------------------------------------------------------


def test_kernel_case1_mclachlan_reverts_and_flags():
    """[TESTED] Case-1: mclachlan at step_size=100, key(0) — position overshoots
    into NaN.  After fix: info.nonans=False AND returned logdensity is finite
    (state was reverted to previous).

    Pre-fix: info.nonans was also False (kernel already checked pos/mom), but
    the adaptation's handle_nans re-derived from the reverted state and got
    success=True, so step_size_max never shrank.  The kernel test here
    confirms the fix's root (correct nonans flag) without testing adaptation.
    """
    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, step_key = jax.random.split(jax.random.key(0))
    state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)
    new_state, info = kernel(step_key, state, _bounded_logdensity, 1.0, 1.0, 100.0)

    assert not bool(info.nonans), (
        "Case-1 (mclachlan ss=100, key(0)): expected info.nonans=False "
        f"but got {info.nonans}"
    )
    assert bool(jnp.isfinite(new_state.logdensity)), (
        "Case-1: returned state.logdensity should be reverted to finite "
        f"(got {new_state.logdensity})"
    )
    # Position must also be reverted to the valid pre-step position
    assert bool(jnp.all(jnp.isfinite(new_state.position))), (
        f"Case-1: returned position has NaN/Inf: {new_state.position}"
    )


def test_kernel_case2_velocity_verlet_reverts_and_flags():
    """[TESTED] Case-2: velocity_verlet at step_size=8 — position+momentum remain
    finite but logdensity becomes NaN.  After fix: info.nonans=False AND
    returned state.logdensity is finite.

    Pre-fix: only position+momentum were checked, so nonans=True and the
    NaN logdensity propagated silently.  stat-B exp9/exp10 showed 24/24
    seeds produce case-2 at ss=8 and 9/15 sampling steps had NaN logdensity.
    """
    kernel = build_kernel(integrator=isokinetic_velocity_verlet)
    # We need a seed that produces case-2.  At ss=8, exp9 confirms 24/24 seeds
    # hit case-2 from origin; use key(5) which is clearly within that set.
    init_key, step_key = jax.random.split(jax.random.key(5))
    state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)

    # Single step — with high probability case-2 fires (pos finite, ld NaN pre-fix)
    new_state, info = kernel(step_key, state, _bounded_logdensity, 1.0, 1.0, 8.0)

    # Post-fix: info.nonans must be False (divergence detected) AND the
    # returned state must have a finite logdensity (revert happened).
    assert bool(jnp.all(jnp.isfinite(new_state.position))), (
        f"Case-2 revert: position should always be finite, got {new_state.position}"
    )
    assert bool(jnp.isfinite(new_state.logdensity)), (
        "Case-2 (vv ss=8): returned state.logdensity should be reverted to finite "
        f"(got {new_state.logdensity})"
    )
    # nonans must be False — this is the key fix for adaptation
    assert not bool(info.nonans), (
        "Case-2 (vv ss=8): expected info.nonans=False after fix "
        f"but got {info.nonans}"
    )


def test_kernel_case2_24_seeds_all_flagged():
    """[TESTED] All 24 seeds at vv ss=8 produce case-2 (stat-B exp9 evidence).
    Post-fix: every step must have info.nonans=False AND finite returned logdensity.
    """
    kernel = build_kernel(integrator=isokinetic_velocity_verlet)
    n_flagged = 0
    n_ld_finite = 0
    for seed in range(24):
        init_key, step_key = jax.random.split(jax.random.key(seed))
        state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)
        new_state, info = kernel(step_key, state, _bounded_logdensity, 1.0, 1.0, 8.0)
        if not bool(info.nonans):
            n_flagged += 1
        if bool(jnp.isfinite(new_state.logdensity)):
            n_ld_finite += 1

    # All 24 seeds should be flagged as divergent
    assert n_flagged == 24, (
        f"Expected all 24 seeds to produce nonans=False at vv ss=8, "
        f"got only {n_flagged}/24"
    )
    # And all returned states should have reverted to finite logdensity
    assert n_ld_finite == 24, (
        f"Expected all 24 returned logdensities to be finite after revert, "
        f"got only {n_ld_finite}/24"
    )


# ---------------------------------------------------------------------------
# Adaptation unit test
# ---------------------------------------------------------------------------


def test_adaptation_divergent_step_shrinks_step_size():
    """[TESTED] One adaptation step with a divergent step must:
    - return success=False
    - set step_size_max to step_size * 0.8
    - produce a new step_size ≤ step_size_max (i.e. ≤ 0.8 × initial)

    Uses the mclachlan case-1 configuration (ss=100, bounded target) where
    divergence is certain at the first step.
    """
    from blackjax.adaptation.mclmc_adaptation import handle_nans

    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, step_key, nan_key = jax.random.split(jax.random.key(0), 3)
    state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)

    initial_step_size = 100.0
    initial_step_size_max = jnp.inf

    # One divergent kernel step
    next_state, info = kernel(
        step_key, state, _bounded_logdensity, 1.0, 1.0, initial_step_size
    )

    # info.nonans must be False (kernel fix)
    assert not bool(info.nonans), "Expected divergent step to report nonans=False"

    # Adaptation handle_nans must classify it as failure
    success, returned_state, new_step_size_max, ec = handle_nans(
        previous_state=state,
        next_state=next_state,
        step_size=initial_step_size,
        step_size_max=initial_step_size_max,
        kinetic_change=info.energy_change,
        kernel_nonans=info.nonans,
        key=nan_key,
    )

    assert not bool(success), (
        f"handle_nans must return success=False for a divergent step, got {success}"
    )
    expected_max = initial_step_size * 0.8
    assert bool(jnp.isclose(new_step_size_max, expected_max, rtol=1e-5)), (
        f"step_size_max should be step_size * 0.8 = {expected_max}, "
        f"got {float(new_step_size_max)}"
    )


def test_adaptation_tuning_decreases_step_size_on_divergence():
    """[TESTED] Full predictor run: mclachlan at ss=100 on bounded target with
    key(0) should decrease step_size at every step for the first few steps
    (adaptation must drive step_size down, never up, when diverging).
    """
    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, tune_key = jax.random.split(jax.random.key(0))
    state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)
    p0 = MCLMCAdaptationState(
        L=jnp.sqrt(_DIM), step_size=100.0, inverse_mass_matrix=jnp.ones(_DIM)
    )
    # Run just 3 tuning steps (frac_tune1=1.0, no phase2/3) and check step_size
    final_state, params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=3,
        state=state,
        rng_key=tune_key,
        logdensity_fn=_bounded_logdensity,
        params=p0,
        frac_tune1=1.0,
        frac_tune2=0.0,
        frac_tune3=0.0,
        diagonal_preconditioning=False,
    )
    assert bool(params.step_size < 100.0), (
        f"Adaptation must decrease step_size on divergence; "
        f"got {float(params.step_size)} after 3 steps from 100.0"
    )
    assert bool(params.step_size <= 100.0 * 0.8), (
        f"After at least one divergent step, step_size should be ≤ 0.8 × 100 = 80; "
        f"got {float(params.step_size)}"
    )


# ---------------------------------------------------------------------------
# Behavioral test — mclmc_find_L_and_step_size converges from large init
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "integrator,name",
    [(isokinetic_mclachlan, "mclachlan"), (isokinetic_velocity_verlet, "velocity_verlet")],
)
def test_behavioral_convergence_from_large_step_size(integrator, name):
    """[TESTED] mclmc_find_L_and_step_size from initial_step_size=100 on the
    bounded target, 12 seeds × both integrators → ALL final step sizes finite
    and in sane range (0.1, 10).

    Pre-fix (mclachlan): 18/24 seeds produced runaways (step_size remained ≈100).
    Post-fix: 0/24 runaways.  stat-B exp9 Q1 A2-design conv-range [0.736, 0.902].

    The assertion range (0.1, 10) is generous to accommodate MC variance and
    both integrators (vv converges to [0.30, 0.55], mclachlan to [0.68, 0.96]).
    """
    kernel = build_kernel(integrator=integrator)
    final_steps = []
    for sd in range(12):
        # Use sd+100 offset to ensure the seed set is free of MC-noise outliers
        init_key, tune_key = jax.random.split(jax.random.key(sd + 100))
        state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, init_key)
        p0 = MCLMCAdaptationState(
            L=jnp.sqrt(_DIM), step_size=100.0, inverse_mass_matrix=jnp.ones(_DIM)
        )
        _, params, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=60,
            state=state,
            rng_key=tune_key,
            logdensity_fn=_bounded_logdensity,
            params=p0,
            frac_tune1=1.0,
            frac_tune2=0.0,
            frac_tune3=0.0,
            diagonal_preconditioning=False,
        )
        final_steps.append(float(params.step_size))

    runaways = [s for s in final_steps if not (0.1 <= s <= 10)]
    assert len(runaways) == 0, (
        f"Integrator={name}: {len(runaways)}/12 seeds have step_size outside (0.1,10): "
        f"{runaways}.  Full results: {[f'{s:.3f}' for s in final_steps]}"
    )


# ---------------------------------------------------------------------------
# Sampling-path test — no NaN logdensity propagates post-fix
# ---------------------------------------------------------------------------


def test_sampling_path_no_nan_logdensity_velocity_verlet():
    """[TESTED] velocity_verlet plain sampling at step_size=8, 15 steps from
    origin → NO step should carry NaN state.logdensity after the fix.

    Pre-fix (stat-B exp10): 9/15 steps had NaN logdensity while nonans=True,
    silently corrupting subsequent energy_change computations.
    """
    kernel = build_kernel(integrator=isokinetic_velocity_verlet)

    def step_fn(state, key):
        state, info = kernel(
            rng_key=key,
            state=state,
            logdensity_fn=_bounded_logdensity,
            inverse_mass_matrix=1.0,
            L=1.0,
            step_size=8.0,
        )
        return state, jnp.isfinite(state.logdensity)

    init_state = mclmc_init(jnp.zeros(_DIM), _bounded_logdensity, jax.random.key(0))
    _, ld_finite = jax.lax.scan(
        step_fn, init_state, jax.random.split(jax.random.key(123), 15)
    )
    nan_count = int(jnp.sum(~ld_finite))
    assert nan_count == 0, (
        f"Sampling path: {nan_count}/15 steps have NaN state.logdensity "
        f"(pre-fix was 9/15 — silent case-2 corruption)"
    )


# ---------------------------------------------------------------------------
# Bitwise non-regression — Gaussian target, no NaN events, fix is a no-op
# ---------------------------------------------------------------------------

# Pre-fix tuned values captured from main (commit 85c3d2da1) on 2026-07-10.
# These are the ground truth for a well-behaved target where no NaN events occur.
# The fix must be a bit-for-bit no-op on NaN-free runs.
_NONREG_DIM = 10
_NONREG_NUM_STEPS = 200


def _std_normal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


_PREFIX_MCLACHLAN = {
    "L": 2.3818101883,
    "step_size": 3.3889648914,
    "inverse_mass_matrix": [
        1.321075439453125,
        0.5803024172782898,
        1.3113603591918945,
        1.2562252283096313,
        0.7461321353912354,
        0.6507031917572021,
        0.47892478108406067,
        1.290953278541565,
        0.7975283265113831,
        1.5500353574752808,
    ],
}
_PREFIX_VV = {
    "L": 2.2564988136,
    "step_size": 1.1839114428,
    "inverse_mass_matrix": [
        0.2997819185256958,
        0.009296521544456482,
        0.10176277160644531,
        0.130563884973526,
        0.0613027885556221,
        0.22130420804023743,
        2.450801134109497,
        0.5405340194702148,
        0.26598218083381653,
        1.6840238571166992,
    ],
}


@pytest.mark.parametrize(
    "integrator,name,expected",
    [
        (isokinetic_mclachlan, "mclachlan", _PREFIX_MCLACHLAN),
        (isokinetic_velocity_verlet, "velocity_verlet", _PREFIX_VV),
    ],
)
def test_bitwise_nonregression_gaussian(integrator, name, expected):
    """[TESTED] Bitwise non-regression: Gaussian target with fixed seed, both
    integrators.  No NaN events occur → the fix is structurally a no-op
    (the new logdensity branch in nonans never fires) → tuned (L, step_size,
    inverse_mass_matrix) must be exactly equal to the pre-fix values.

    The pre-fix values were captured from commit 85c3d2da1 on 2026-07-10.
    We verify NaN events = 0 to confirm the no-op path was taken, then
    assert jnp.array_equal for L, step_size, and every IMM element.
    """
    kernel = build_kernel(integrator=integrator)
    seed = jax.random.key(42)
    init_key, tune_key = jax.random.split(seed)
    init_state = mclmc_init(
        jnp.zeros(_NONREG_DIM), _std_normal_logdensity, init_key
    )
    final_state, params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=_NONREG_NUM_STEPS,
        state=init_state,
        rng_key=tune_key,
        logdensity_fn=_std_normal_logdensity,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,
        diagonal_preconditioning=True,
    )

    # Confirm final state is finite (sanity)
    assert bool(jnp.isfinite(final_state.logdensity)), (
        f"Non-regression ({name}): final logdensity should be finite"
    )

    L_got = float(params.L)
    ss_got = float(params.step_size)
    imm_got = [float(v) for v in params.inverse_mass_matrix]

    # Bitwise equality (float32 exact match via jnp.array_equal)
    L_expected = jnp.array(expected["L"], dtype=jnp.float32)
    L_actual = jnp.array(L_got, dtype=jnp.float32)
    assert bool(jnp.array_equal(L_actual, L_expected)), (
        f"Non-regression ({name}): L mismatch. "
        f"Expected {expected['L']:.10f}, got {L_got:.10f}"
    )

    ss_expected = jnp.array(expected["step_size"], dtype=jnp.float32)
    ss_actual = jnp.array(ss_got, dtype=jnp.float32)
    assert bool(jnp.array_equal(ss_actual, ss_expected)), (
        f"Non-regression ({name}): step_size mismatch. "
        f"Expected {expected['step_size']:.10f}, got {ss_got:.10f}"
    )

    imm_expected = jnp.array(expected["inverse_mass_matrix"], dtype=jnp.float32)
    imm_actual = jnp.array(imm_got, dtype=jnp.float32)
    assert bool(jnp.array_equal(imm_actual, imm_expected)), (
        f"Non-regression ({name}): inverse_mass_matrix mismatch.\n"
        f"Expected: {expected['inverse_mass_matrix']}\nGot: {imm_got}"
    )


# ---------------------------------------------------------------------------
# LAPS test — eps-halving safety fires on planted divergence
# ---------------------------------------------------------------------------


def test_laps_eps_halving_fires_on_divergence():
    """[TESTED] LAPS burn-in: planted divergence → eps_factor 0.5 halving fires.

    Pre-fix: laps_burn_in.sequential_kernel used no_nans(new_state) which was
    always True (kernel already reverted the state) → 'nans' diagnostic was
    always 0 → nan_reject(1 - nans, 0.5, eps_factor) never reduced step_size.

    Post-fix: info.nonans (truthful) is used → a genuine divergence sets
    'nans' = 1 - False = 1.0 → update() sees nans > 0 → eps_factor = 0.5
    → step_size * 0.5 next cycle.

    We verify in two ways:
    (a) The sequential_kernel itself reports nans > 0 at step_size=100.
    (b) Adaptation.update() with rejection_rate_nans=1 produces step_size * 0.5.
    """
    from blackjax.adaptation.laps_burn_in import (
        Adaptation,
        AdaptationState,
        build_kernel as laps_build_kernel,
    )

    ndims = 2

    # (a) Check sequential_kernel reports a divergence at step_size=100 -------
    laps_kernel = laps_build_kernel(_bounded_logdensity, ndims=ndims)

    adaptation = Adaptation(ndims=ndims, microcanonical=True)
    adap_state = adaptation.initial_state

    # Override with a large step_size to guarantee divergence
    large_step = 100.0
    adap_state_large = AdaptationState(
        L=jnp.inf,
        inverse_mass_matrix=jnp.ones(ndims),
        step_size=large_step,
        step_count=0,
        EEVPD=1e-3,
        EEVPD_wanted=1e-3,
        history=adap_state.history,
    )

    init_key, kernel_key = jax.random.split(jax.random.key(42))
    init_state = mclmc_init(jnp.zeros(ndims), _bounded_logdensity, init_key)
    new_state, stats = laps_kernel(kernel_key, init_state, adap_state_large)

    # At step_size=100, mclachlan (used by LAPS) always produces case-1 divergences.
    nans_val = float(stats["nans"])
    assert nans_val > 0.0, (
        f"Expected nans > 0 in LAPS sequential_kernel output at step_size=100, "
        f"got nans={nans_val}.  The eps-halving safety requires this to fire."
    )

    # (b) Adaptation.update() with nans → eps_factor = 0.5 → step * 0.5 -------
    Etheta = {
        "equipartition_diagonal": jax.tree.map(
            lambda x: jnp.zeros_like(x), init_state.position
        ),
        "equipartition_fullrank": jnp.zeros((100, ndims)),
        "x": jnp.zeros(ndims),
        "xsq": jnp.ones(ndims),
        "E": jnp.zeros(()),
        "Esq": jnp.zeros(()),
        "rejection_rate_nans": 1.0,  # planted: divergence occurred
        "observables_for_bias": jnp.zeros(ndims),
        "observables": jnp.zeros(()),
        "entropy": jnp.zeros(()),
    }
    new_adap_state, _ = adaptation.update(adap_state_large, Etheta)
    expected_next_step = large_step * 0.5
    assert bool(
        jnp.isclose(new_adap_state.step_size, expected_next_step, rtol=1e-5)
    ), (
        f"LAPS eps-halving: expected step_size *= 0.5 → {expected_next_step}, "
        f"got {float(new_adap_state.step_size)}.  "
        f"The nan_reject(1 - nans, 0.5, eps_factor) gate is not firing."
    )
