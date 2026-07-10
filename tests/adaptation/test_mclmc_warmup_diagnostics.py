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
"""Tests for the MCLMC warmup diagnostic warnings (blackjax#973 follow-up).

Two bespoke endpoint guards added in feat/mclmc-warmup-diagnostics:
  1. Frozen-outcome warning — fires when adapted step_size cannot traverse
     the posterior in the sampling budget (mixing infeasibility check).
  2. Gradient-finiteness init guard — fires when the init-state gradient is
     non-finite (value-finiteness is insufficient, e.g. lotka ODE stiffness).

Per-step divergence monitoring is delegated to jax-tap (>= 0.3.0) via the
scan ys enabling seam — see Notes in mclmc_find_L_and_step_size docstring.

All warnings are unreachable on healthy runs (zero behavior change).
"""
import warnings

import jax
import jax.numpy as jnp
import pytest

from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.mcmc.mclmc import build_kernel
from blackjax.mcmc.mclmc import init as mclmc_init

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIM = 2


def _cliff_target(x):
    """Only finite at x=0 in float32; any kernel step produces ld=-inf (not finite).

    Construction: log(max(1 - |x|^2 * 1e35, 0)).  At x=0 this is log(1)=0.
    For any |x|^2 > 0 in float32 the factor 1e35 drives the argument below 0,
    so max clips to 0 and log(0)=-inf.  Since jnp.isfinite(-inf)=False, the
    kernel flags nonans=False on every step and reverts — position stays at
    x=0, warmup_position_std accumulates to 0, triggering the frozen-outcome
    warning.
    """
    return jnp.log(jnp.maximum(1.0 - jnp.sum(x**2) * 1e35, 0.0))


def _gaussian_target(x):
    """Unbounded std-normal; should produce zero warmup warnings."""
    return -0.5 * jnp.sum(x**2)


def _nan_grad_target(x):
    """Finite logdensity at x=0 but NaN gradient (lotka-style init pathology).

    -sqrt(sum(x^2)) has gradient -x/|x|, which is 0/0 = NaN at x=0.
    This tests that the init guard catches value-finite but grad-non-finite
    targets — the same failure mode as lotka-volterra with a stiff ODE solver.
    """
    return -0.5 * jnp.sum(x**2) - jnp.sqrt(jnp.sum(x**2))


def _run_adaptation(target, ss_init, num_steps=200, frac1=0.5, frac2=0.5, frac3=0.0):
    kernel = build_kernel(integrator=isokinetic_mclachlan)
    init_key, tune_key = jax.random.split(jax.random.key(0))
    state = mclmc_init(jnp.zeros(_DIM), target, init_key)
    p0 = MCLMCAdaptationState(
        L=jnp.sqrt(_DIM), step_size=ss_init, inverse_mass_matrix=jnp.ones(_DIM)
    )
    return mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=state,
        rng_key=tune_key,
        logdensity_fn=target,
        params=p0,
        frac_tune1=frac1,
        frac_tune2=frac2,
        frac_tune3=frac3,
        diagonal_preconditioning=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cliff_replica_fires_frozen_warning():
    """Cliff target: every kernel step produces ld=-inf, so pos_std stays 0.

    Expected: frozen-outcome warning (pos_std < 1e-10 condition).
    No count warning (dropped in favour of the jaxtap y-tap route).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, params, _ = _run_adaptation(_cliff_target, ss_init=10.0, num_steps=200)

    user_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    frozen_warns = [m for m in user_msgs if "collapsed" in m.lower()]

    assert frozen_warns, f"expected frozen-outcome warning, got {user_msgs}"
    # Must not fire a divergence-count warning (machinery was removed)
    count_warns = [m for m in user_msgs if "divergent" in m.lower()]
    assert not count_warns, (
        f"divergence-count warning must not fire (use jaxtap recipe instead), "
        f"got {count_warns}"
    )


def test_healthy_gaussian_zero_warnings():
    """Std-normal target from reasonable init: all warnings must be silent.

    This is the structural no-op prover for the diagnostics layer: healthy runs
    produce zero UserWarnings (zero behavior change, zero output).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_adaptation(_gaussian_target, ss_init=1.0, num_steps=100)

    user_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not user_msgs, f"healthy Gaussian must emit zero warnings, got: {user_msgs}"


def test_grad_finiteness_init_guard():
    """Finite logdensity but NaN gradient at init: the init guard fires.

    Models like lotka-volterra with stiff ODE solvers can produce NaN gradients
    even when the log-density value is finite.  Value-checking alone at init is
    insufficient — this test verifies the gradient guard catches the pathology.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_adaptation(
            _nan_grad_target, ss_init=0.5, num_steps=20, frac1=1.0, frac2=0.0
        )

    grad_warns = [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and "gradient" in str(w.message).lower()
    ]
    all_user_msgs = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert grad_warns, f"expected gradient non-finite warning, got {all_user_msgs}"
    assert (
        "model/solver/support" in grad_warns[0]
    ), f"message should direct user to model, got: {grad_warns[0]}"


def test_warning_totality_under_w_error():
    """Warnings promoted to exceptions by -W error must not crash the adaptation.

    jax.debug.callback functions wrap warnings.warn in try/except to ensure a
    warnings.simplefilter("error") environment cannot raise through the JAX
    runtime.  This test verifies the adaptation run completes under that
    environment — the cliff target triggers warnings, but the run must finish.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote all warnings to exceptions
        _, final_params, total_steps = _run_adaptation(
            _cliff_target, ss_init=10.0, num_steps=200
        )
    # If we reach here the callbacks' try/except prevented any crash
    assert final_params is not None
    assert total_steps >= 0


def test_jaxtap_recipe_fires_on_cliff():
    """The documented jaxtap 0.3.0 recipe emits output events on the cliff target.

    Verifies that the Notes-paragraph recipe in mclmc_find_L_and_step_size
    actually fires alert_ys on the cliff replica (every step diverges there).
    Skipped when jax-tap is not installed — it is an optional monitoring layer.
    """
    jaxtap = pytest.importorskip(
        "jaxtap", minversion="0.3.0", reason="jax-tap>=0.3.0 not installed"
    )

    with jaxtap.record(
        select_ys=lambda ys: ys[0],  # the single divergence-flag leaf
        alert_ys=lambda e: "divergence" if e.value else None,
    ) as rec:
        _run_adaptation(_cliff_target, ss_init=10.0, num_steps=200)

    output_events = [e for e in rec.events if e.kind == "output"]
    divergent = [e for e in output_events if e.value]
    assert output_events, (
        "jaxtap recipe produced no output events from cliff target — "
        "check that the scan ys enabling seam is still in place"
    )
    assert divergent, (
        f"jaxtap recipe: no divergent steps detected on cliff target "
        f"({len(output_events)} output events, all non-divergent)"
    )
    # Spot-check: step index 0 should be divergent on the cliff
    first_output = output_events[0]
    assert first_output.step == 0, f"unexpected first step index: {first_output.step}"
