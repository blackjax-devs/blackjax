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
"""Regression tests for the LAPS silent NaN death (#1035).

D1: the NaN gate must cover kinetic/energy change, not just state.
D2: a non-finite eps_factor must not permanently kill the step size.
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


def test_handle_nans_catches_nan_kinetic_change_with_finite_state():
    """A NaN kinetic/energy change with an otherwise-finite state must be
    flagged and reverted; pre-D1 the gate never looked past position,
    momentum and logdensity, so this route passed silently (nonans=True)."""
    previous_state = _finite_state()
    next_state = _finite_state()
    info = MCLMCInfo(
        logdensity=next_state.logdensity,
        kinetic_change=jnp.nan,
        energy_change=jnp.nan,
        nonans=True,
    )

    _, new_info = handle_nans(previous_state, next_state, info, jax.random.key(0))

    assert not bool(new_info.nonans)
    assert jnp.isfinite(new_info.energy_change)
    assert jnp.isfinite(new_info.kinetic_change)


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


def test_step_size_survives_undetected_nan_route():
    """D2 must stand on its own: even with rejection_rate_nans=0 (a route D1
    does not itself close), a NaN E/Esq driving eps_factor non-finite must
    fall back to the previous step size rather than corrupt it forever."""
    adap_state, adaptation = _adaptation_state(step_size=0.05)
    etheta = _etheta(E=jnp.nan, Esq=jnp.nan, rejection_rate_nans=0.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isfinite(new_state.step_size)
    assert new_state.step_size == jnp.float32(0.05)


def test_step_size_still_halves_on_reported_nan():
    """Control: the pre-existing eps_factor=0.5 safety (rejection_rate_nans
    > 0) is untouched by the D2 fallback."""
    adap_state, adaptation = _adaptation_state(step_size=100.0)
    etheta = _etheta(rejection_rate_nans=1.0)

    new_state, _ = adaptation.update(adap_state, etheta)

    assert jnp.isclose(new_state.step_size, 50.0, rtol=1e-5)
