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
"""Tests for the MCLMC warmup diagnostic seam (blackjax#973 follow-up).

The scan ys enabling seam exposes a per-step divergence flag from the internal
tuning scan.  Per-step monitoring is delegated to jax-tap (>= 0.3.0) via this
seam — see Notes in mclmc_find_L_and_step_size docstring.
"""

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
    kernel flags nonans=False on every step and reverts — every step diverges.
    """
    return jnp.log(jnp.maximum(1.0 - jnp.sum(x**2) * 1e35, 0.0))


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
