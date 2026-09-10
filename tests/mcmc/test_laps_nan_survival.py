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
"""Regression tests for the LAPS nightly NaN-death defect (two-piece fix).

D1 (blackjax/mcmc/mclmc.py, handle_nans): a NaN kinetic_change / energy_change
paired with an otherwise-finite proposed state (position, momentum,
logdensity) used to pass the kernel's own NaN gate silently. Root cause:
`momentum_proj` in the isokinetic integrator is a dot product of two unit
vectors that can round marginally below -1 in float32, driving
`jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)` negative. In an
ECA ensemble (LAPS burn-in), one such chain contaminates every chain's summary
statistics via `lax.psum`.

D2 (blackjax/adaptation/laps_burn_in.py, Adaptation.update): the adapted step
size is updated multiplicatively (`step_size * eps_factor`), so a single
non-finite `eps_factor` -- from D1 or from any other undetected route -- is
absorbing: once step_size is NaN it stays NaN for the rest of burn-in, and
LAPS silently returns a frozen, un-equilibrated ensemble with no error and no
warning.

Both tests below fail on pre-fix `main` and pass after the fix.
"""
import jax
import jax.numpy as jnp

from blackjax.adaptation.laps_burn_in import Adaptation, AdaptationState
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.mclmc import MCLMCInfo, handle_nans

_DIM = 2


def _finite_state(dim=_DIM):
    return IntegratorState(
        position=jnp.zeros(dim),
        momentum=jnp.ones(dim) / jnp.sqrt(dim),
        logdensity=jnp.array(-1.0),
        logdensity_grad=jnp.zeros(dim),
    )


# ---------------------------------------------------------------------------
# D1 -- the NaN gate must also cover kinetic_change / energy_change
# ---------------------------------------------------------------------------


def test_handle_nans_catches_nan_kinetic_change_with_finite_state():
    """A NaN kinetic_change/energy_change with an otherwise-finite proposed
    state must be flagged and reverted, exactly like a NaN position or a NaN
    logdensity. Pre-D1, `info.nonans` looked only at position/momentum/
    logdensity, so this route passed the gate silently (nonans=True) and a
    single such chain could poison an ECA ensemble average with no warning.
    """
    previous_state = _finite_state()
    # The proposed state itself is entirely finite -- only the energy
    # accounting is broken, exactly as measured in the LAPS nightly trace
    # (kinetic_change=NaN while position/momentum/logdensity stayed finite).
    next_state = _finite_state()
    info = MCLMCInfo(
        logdensity=next_state.logdensity,
        kinetic_change=jnp.nan,
        energy_change=jnp.nan,
        nonans=True,
    )

    new_state, new_info = handle_nans(
        previous_state, next_state, info, jax.random.key(0)
    )

    assert not bool(new_info.nonans), (
        "a NaN kinetic_change/energy_change must be flagged even when "
        "position, momentum and logdensity are all finite"
    )
    assert jnp.isfinite(new_info.energy_change), "reverted energy_change must be finite"
    assert jnp.isfinite(
        new_info.kinetic_change
    ), "reverted kinetic_change must be finite"
    assert jnp.array_equal(
        new_state.position, previous_state.position
    ), "a flagged step must revert position to the previous state"


def test_handle_nans_healthy_state_unaffected():
    """Control: a fully finite proposed step (including kinetic/energy
    change) is unaffected by the D1 change -- proves the extended gate is a
    strict-superset check, not a behavior change on healthy steps.
    """
    previous_state = _finite_state()
    next_state = _finite_state()
    info = MCLMCInfo(
        logdensity=next_state.logdensity,
        kinetic_change=jnp.array(0.1),
        energy_change=jnp.array(0.05),
        nonans=True,
    )

    new_state, new_info = handle_nans(
        previous_state, next_state, info, jax.random.key(0)
    )

    assert bool(new_info.nonans)
    assert new_info.energy_change == info.energy_change
    assert new_info.kinetic_change == info.kinetic_change
    assert jnp.array_equal(new_state.position, next_state.position)


# ---------------------------------------------------------------------------
# D2 -- the adapted step size must stay finite even if some other route
# produces a non-finite eps_factor
# ---------------------------------------------------------------------------


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
        # Non-zero so contract_history's r = (avg_sq - sq_avg) / sq_avg does
        # not divide 0/0 -- keeps the "healthy" control path itself finite so
        # it exercises the same eps_factor arithmetic as a real run.
        "observables_for_bias": jnp.full((ndims,), 2.0),
        "observables": jnp.zeros(()),
        "entropy": jnp.zeros(()),
    }


def test_step_size_survives_undetected_nan_route():
    """D2 must stand on its own: even when `rejection_rate_nans` is 0 (no
    chain was flagged -- simulating a route D1 does not cover), a NaN
    summary statistic that drives `eps_factor` non-finite must not poison the
    step size. This is the invariant that lets LAPS survive a poisoned
    iteration instead of freezing for the rest of its budget.
    """
    adap_state, adaptation = _adaptation_state(step_size=0.05)
    # E/Esq are NaN (as ECA's lax.psum would produce from one poisoned chain)
    # while rejection_rate_nans reports clean -- the D1 failure mode, tested
    # here in isolation from D1's own fix.
    etheta = _etheta(E=jnp.nan, Esq=jnp.nan, rejection_rate_nans=0.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isfinite(
        new_state.step_size
    ), f"step size must stay finite, got {float(new_state.step_size)}"
    assert new_state.step_size == jnp.float32(0.05), (
        "on an undetected non-finite route, step size should fall back to "
        "the previous (finite) value rather than being corrupted, got "
        f"{float(new_state.step_size)}"
    )


def test_step_size_healthy_update_unaffected():
    """Control: a normal, fully-finite update produces a finite, moved step
    size -- proves the D2 fallback changes nothing on a healthy iteration.
    """
    adap_state, adaptation = _adaptation_state(step_size=0.05)
    etheta = _etheta(E=0.1, Esq=0.02, rejection_rate_nans=0.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isfinite(new_state.step_size)
    assert float(new_state.step_size) != 0.05  # the update actually moved it


def test_step_size_still_halves_on_reported_nan():
    """Control: the pre-existing eps_factor=0.5 safety (rejection_rate_nans
    > 0) is untouched by the D2 fallback."""
    adap_state, adaptation = _adaptation_state(step_size=100.0)
    etheta = _etheta(rejection_rate_nans=1.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isclose(new_state.step_size, 50.0, rtol=1e-5)
