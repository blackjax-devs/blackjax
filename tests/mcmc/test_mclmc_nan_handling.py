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
"""Tests for MCLMC/LAPS non-finite handling.

Covers the kernel's own divergence detection (position, momentum and
logdensity finiteness, plus kinetic/energy-change finiteness), the
adaptation layer's step-size response to a flagged kernel step, and the
LAPS burn-in's eps-halving safety when the reported acceptance statistics
are themselves non-finite.
"""

import jax
import jax.numpy as jnp
import pytest

from blackjax.adaptation.laps_burn_in import Adaptation, AdaptationState
from blackjax.adaptation.laps_burn_in import build_kernel as laps_build_kernel
from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    handle_nans,
    mclmc_find_L_and_step_size,
)
from blackjax.mcmc.integrators import (
    IntegratorState,
    isokinetic_mclachlan,
    isokinetic_velocity_verlet,
)
from blackjax.mcmc.mclmc import MCLMCInfo, build_kernel
from blackjax.mcmc.mclmc import handle_nans as kernel_handle_nans
from blackjax.mcmc.mclmc import init as mclmc_init

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIM = 2
_BOUND = 5.0

_INTEGRATORS = [
    (isokinetic_mclachlan, "mclachlan"),
    (isokinetic_velocity_verlet, "velocity_verlet"),
]


def _bounded_target(x):
    """Std-normal inside a hard box at ±_BOUND; overshoots produce NaN logdensity."""
    return -0.5 * jnp.sum(x**2) + jnp.sum(jnp.log(_BOUND - jnp.abs(x)))


def _gaussian_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _run_tuning(integrator, seed, ss_init, num_steps=60):
    """Run mclmc_find_L_and_step_size on the bounded target; return final params."""
    kernel = build_kernel(integrator=integrator)
    init_key, tune_key = jax.random.split(jax.random.key(seed))
    state = mclmc_init(jnp.zeros(_DIM), _bounded_target, init_key)
    p0 = MCLMCAdaptationState(
        L=jnp.sqrt(_DIM), step_size=ss_init, inverse_mass_matrix=jnp.ones(_DIM)
    )
    _, params, _ = mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=state,
        rng_key=tune_key,
        logdensity_fn=_bounded_target,
        params=p0,
        frac_tune1=1.0,
        frac_tune2=0.0,
        frac_tune3=0.0,
        diagonal_preconditioning=False,
    )
    return params


def _finite_state(dim=_DIM):
    return IntegratorState(
        position=jnp.zeros(dim),
        momentum=jnp.ones(dim) / jnp.sqrt(dim),
        logdensity=jnp.array(-1.0),
        logdensity_grad=jnp.zeros(dim),
    )


def _adaptation_state(step_size, ndims=_DIM):
    adaptation = Adaptation(ndims=ndims, microcanonical=True)
    state = AdaptationState(
        L=jnp.inf,
        inverse_mass_matrix=jnp.ones(ndims),
        step_size=step_size,
        step_count=0,
        EEVPD=1e-3,
        EEVPD_wanted=1e-3,
        history=adaptation.initial_state.history,
    )
    return state, adaptation


def _etheta(ndims=_DIM, E=0.0, Esq=0.0, rejection_rate_nans=0.0):
    return {
        "equipartition_diagonal": jnp.zeros(ndims),
        "equipartition_fullrank": jnp.zeros((100, ndims)),
        "x": jnp.zeros(ndims),
        "xsq": jnp.ones(ndims),
        "E": jnp.asarray(E),
        "Esq": jnp.asarray(Esq),
        "rejection_rate_nans": jnp.asarray(rejection_rate_nans),
        "observables_for_bias": jnp.full(
            (ndims,), 2.0
        ),  # non-zero: avoids 0/0 in contract_history
        "observables": jnp.zeros(()),
        "entropy": jnp.zeros(()),
    }


# ---------------------------------------------------------------------------
# Kernel tests — divergence detection
# ---------------------------------------------------------------------------


def test_kernel_reverts_and_flags_nonfinite_step():
    """mclachlan ss=100, key(0): position overshoots into NaN.

    info.nonans=False and the returned state is reverted to finite.
    """
    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, step_key = jax.random.split(jax.random.key(0))
    state = mclmc_init(jnp.zeros(_DIM), _bounded_target, init_key)
    new_state, info = kernel(step_key, state, _bounded_target, 1.0, 1.0, 100.0)

    assert not bool(info.nonans), f"expected nonans=False, got {info.nonans}"
    assert jnp.isfinite(new_state.logdensity), "reverted logdensity must be finite"
    assert jnp.all(jnp.isfinite(new_state.position)), "reverted position must be finite"


def test_kernel_flags_finite_state_with_nonfinite_logdensity():
    """velocity_verlet ss=8, 8 seeds: position and momentum stay finite but
    logdensity goes to NaN. All 8 seeds must be flagged (nonans=False) and
    reverted to a finite logdensity.
    """
    kernel = build_kernel(integrator=isokinetic_velocity_verlet)
    n_flagged = n_ld_finite = 0
    for seed in range(8):
        init_key, step_key = jax.random.split(jax.random.key(seed))
        state = mclmc_init(jnp.zeros(_DIM), _bounded_target, init_key)
        new_state, info = kernel(step_key, state, _bounded_target, 1.0, 1.0, 8.0)
        n_flagged += not bool(info.nonans)
        n_ld_finite += bool(jnp.isfinite(new_state.logdensity))

    assert n_flagged == 8, (
        f"expected all 8 seeds to flag nonans=False at vv ss=8, got {n_flagged}/8"
    )
    assert n_ld_finite == 8, (
        f"expected 8/8 reverted logdensities to be finite, got {n_ld_finite}/8"
    )


def test_kernel_handle_nans_catches_nan_kinetic_change_with_finite_state():
    """A NaN kinetic/energy change with an otherwise-finite state must be
    flagged and reverted; a gate that only checks position, momentum and
    logdensity would pass this route silently (nonans=True)."""
    previous_state = _finite_state()
    next_state = _finite_state()
    info = MCLMCInfo(
        logdensity=next_state.logdensity,
        kinetic_change=jnp.nan,
        energy_change=jnp.nan,
        nonans=True,
    )

    _, new_info = kernel_handle_nans(
        previous_state, next_state, info, jax.random.key(0)
    )

    assert not bool(new_info.nonans)
    assert jnp.isfinite(new_info.energy_change)
    assert jnp.isfinite(new_info.kinetic_change)


# ---------------------------------------------------------------------------
# MCLMC adaptation tests
# ---------------------------------------------------------------------------


def test_adaptation_divergent_step_shrinks_step_size():
    """Single step: handle_nans returns success=False and shrinks step_size_max to 0.8×ss.
    Three steps: adaptation must drive step_size to ≤ 80 from ss_init=100.
    """
    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, step_key, nan_key = jax.random.split(jax.random.key(0), 3)
    state = mclmc_init(jnp.zeros(_DIM), _bounded_target, init_key)

    next_state, info = kernel(step_key, state, _bounded_target, 1.0, 1.0, 100.0)
    assert not bool(info.nonans), "kernel must flag the divergent step"

    success, _, new_step_size_max, _ = handle_nans(
        previous_state=state,
        next_state=next_state,
        step_size=100.0,
        step_size_max=jnp.inf,
        kinetic_change=info.energy_change,
        kernel_nonans=info.nonans,
        key=nan_key,
    )
    assert not bool(success), (
        "handle_nans must return success=False on a divergent step"
    )
    assert jnp.isclose(new_step_size_max, 80.0, rtol=1e-5), (
        f"step_size_max should be 100*0.8=80, got {float(new_step_size_max)}"
    )

    # Multi-step: three tuning steps from ss=100 must decrease step_size to ≤ 80
    params = _run_tuning(isokinetic_mclachlan, seed=0, ss_init=100.0, num_steps=3)
    assert params.step_size <= 80.0, (
        f"After ≥1 divergent step, step_size must be ≤80 (got {float(params.step_size)})"
    )


@pytest.mark.parametrize("integrator,name", _INTEGRATORS)
def test_behavioral_convergence_from_large_step_size(integrator, name):
    """6 seeds × both integrators from ss=100: all final step_sizes in (0.1, 10)."""
    final_steps = [
        float(_run_tuning(integrator, seed=sd + 100, ss_init=100.0).step_size)
        for sd in range(6)
    ]
    runaways = [s for s in final_steps if not (0.1 <= s <= 10.0)]
    assert not runaways, f"{name}: {len(runaways)}/6 outside (0.1, 10): {runaways}"


# ---------------------------------------------------------------------------
# Sampling-path test
# ---------------------------------------------------------------------------


def test_sampling_path_no_nan_logdensity_velocity_verlet():
    """vv ss=8, 15 steps from origin: no NaN logdensity propagates (pre-fix: 9/15)."""
    kernel = build_kernel(integrator=isokinetic_velocity_verlet)

    def step_fn(state, key):
        state, _ = kernel(key, state, _bounded_target, 1.0, 1.0, 8.0)
        return state, jnp.isfinite(state.logdensity)

    init_state = mclmc_init(jnp.zeros(_DIM), _bounded_target, jax.random.key(0))
    _, ld_finite = jax.lax.scan(
        step_fn, init_state, jax.random.split(jax.random.key(123), 15)
    )
    nan_count = int(jnp.sum(~ld_finite))
    assert nan_count == 0, (
        f"{nan_count}/15 steps have NaN logdensity (pre-fix was 9/15)"
    )


# ---------------------------------------------------------------------------
# Structural no-op proof — Gaussian target, both integrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrator,name", _INTEGRATORS)
def test_structural_noop_gaussian(integrator, name):
    """On std-normal, the logdensity branch of the divergence check is algebraically a no-op.

    (a) info.nonans (pos∧mom∧ld) is True every step — the revert branch never fires.
    (b) Returned ld is finite every step; since no revert occurs (confirmed by (a)), this is
        the proposed ld, proving isfinite(ld) was always True.
    NOTE: this is a no-op prover, not a divergence detector —
    test_kernel_flags_finite_state_with_nonfinite_logdensity is the detector for that signature.
    """
    kernel = build_kernel(integrator=integrator)
    dim = 10
    init_state = mclmc_init(jnp.zeros(dim), _gaussian_logdensity, jax.random.key(42))

    def step_fn(state, key):
        new_state, info = kernel(
            key, state, _gaussian_logdensity, jnp.ones(dim), jnp.sqrt(dim), 1.0
        )
        return new_state, (
            info.nonans,
            jnp.isfinite(info.energy_change),
            jnp.isfinite(new_state.logdensity),
        )

    _, (all_nonans, all_ec_finite, all_ld_finite) = jax.lax.scan(
        step_fn, init_state, jax.random.split(jax.random.key(42), 200)
    )

    # (a) the revert branch never fires on a well-conditioned Gaussian target
    assert jnp.all(all_nonans), (
        f"{name}: revert branch fired on Gaussian — not a no-op here"
    )
    assert jnp.all(all_ec_finite), f"{name}: non-finite energy_change on Gaussian"
    # (b) Since no revert occurred, returned ld equals proposed ld; it is always finite,
    #     so isfinite(ld) was True at every proposed step.
    assert jnp.all(all_ld_finite), (
        f"{name}: non-finite ld on Gaussian — formulas would diverge"
    )


# ---------------------------------------------------------------------------
# LAPS tests
# ---------------------------------------------------------------------------


def test_laps_eps_halving_fires_on_divergence():
    """LAPS eps-halving safety fires on a divergent step."""
    ndims = 2
    laps_kernel = laps_build_kernel(_bounded_target, ndims=ndims)
    adaptation = Adaptation(ndims=ndims, microcanonical=True)
    adap_state = AdaptationState(
        L=jnp.inf,
        inverse_mass_matrix=jnp.ones(ndims),
        step_size=100.0,
        step_count=0,
        EEVPD=1e-3,
        EEVPD_wanted=1e-3,
        history=adaptation.initial_state.history,
    )

    init_key, kernel_key = jax.random.split(jax.random.key(42))
    init_state = mclmc_init(jnp.zeros(ndims), _bounded_target, init_key)
    _, stats = laps_kernel(kernel_key, init_state, adap_state)
    assert float(stats["nans"]) > 0.0, (
        f"LAPS nans must be >0 at ss=100 (got {stats['nans']})"
    )

    Etheta = {
        "equipartition_diagonal": jax.tree.map(
            lambda x: jnp.zeros_like(x), init_state.position
        ),
        "equipartition_fullrank": jnp.zeros((100, ndims)),
        "x": jnp.zeros(ndims),
        "xsq": jnp.ones(ndims),
        "E": jnp.zeros(()),
        "Esq": jnp.zeros(()),
        "rejection_rate_nans": 1.0,
        "observables_for_bias": jnp.zeros(ndims),
        "observables": jnp.zeros(()),
        "entropy": jnp.zeros(()),
    }
    new_adap_state, _ = adaptation.update(adap_state, Etheta)
    assert jnp.isclose(new_adap_state.step_size, 50.0, rtol=1e-5), (
        f"LAPS eps-halving: expected step_size=50.0, got {float(new_adap_state.step_size)}"
    )


def test_step_size_survives_undetected_nan_route():
    """Even on a route where rejection_rate_nans=0 does not itself signal a
    problem, a NaN E/Esq driving eps_factor non-finite must fall back to the
    previous step size rather than corrupt it forever."""
    adap_state, adaptation = _adaptation_state(step_size=0.05)
    etheta = _etheta(E=jnp.nan, Esq=jnp.nan, rejection_rate_nans=0.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isfinite(new_state.step_size)
    assert new_state.step_size == jnp.float32(0.05)


def test_step_size_still_halves_on_reported_nan():
    """Control: the pre-existing eps_factor=0.5 safety (rejection_rate_nans
    > 0) is untouched by the fallback above."""
    adap_state, adaptation = _adaptation_state(step_size=100.0)
    etheta = _etheta(rejection_rate_nans=1.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isclose(new_state.step_size, 50.0, rtol=1e-5)
