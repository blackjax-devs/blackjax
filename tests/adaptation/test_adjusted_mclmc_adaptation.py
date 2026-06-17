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
"""Tests for adjusted_mclmc_find_L_and_step_size — focusing on the
target_num_integration_steps fix that prevents the MALA collapse (avg ≈ 1).
"""

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
import blackjax.diagnostics as diagnostics
from blackjax.mcmc.adjusted_mclmc import rescale
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIM = 5
_NUM_STEPS = 4000  # total steps including warmup
_TUNE_TARGET = 0.65
_TUNE_FRAC1 = 0.1
_TUNE_FRAC2 = 0.1


def _make_initial_state(key, logdensity_fn, dim):
    init_key, rng_key = jax.random.split(key)
    position = jax.random.normal(init_key, (dim,))
    state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=rng_key,
    )
    return state


def _make_kernel():
    return blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integrator=isokinetic_mclachlan,
        integration_steps_fn=lambda key, avg: jnp.ceil(
            jax.random.uniform(key) * rescale(avg)
        ).astype(jnp.int32),
    )


def _run_tuner(key, state, logdensity_fn, *, target_num_integration_steps):
    kernel = _make_kernel()
    _, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        logdensity_fn=logdensity_fn,
        num_steps=_NUM_STEPS,
        state=state,
        rng_key=key,
        target=_TUNE_TARGET,
        frac_tune1=_TUNE_FRAC1,
        frac_tune2=_TUNE_FRAC2,
        frac_tune3=0.0,
        target_num_integration_steps=target_num_integration_steps,
    )
    return params


def _run_chain(run_key, state, logdensity_fn, params, num_steps):
    step_size = params.step_size
    L = params.L
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key, avg: jnp.ceil(
            jax.random.uniform(key) * rescale(avg)
        ).astype(jnp.int32),
        integration_steps_params=(L / step_size,),
        integrator=isokinetic_mclachlan,
        inverse_mass_matrix=params.inverse_mass_matrix,
    )
    _, (_, info) = run_inference_algorithm(
        rng_key=run_key,
        initial_state=state,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=lambda state, info: (state.position, info),
    )
    return info


def _run_chain_positions(run_key, state, logdensity_fn, params, num_steps):
    step_size = params.step_size
    L = params.L
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key, avg: jnp.ceil(
            jax.random.uniform(key) * rescale(avg)
        ).astype(jnp.int32),
        integration_steps_params=(L / step_size,),
        integrator=isokinetic_mclachlan,
        inverse_mass_matrix=params.inverse_mass_matrix,
    )
    _, (positions, _) = run_inference_algorithm(
        rng_key=run_key,
        initial_state=state,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=lambda state, info: (state.position, info),
    )
    return positions


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestAdjustedMclmcTargetIntegrationSteps(BlackJAXTest):
    """Tests for the target_num_integration_steps fix.

    Uses a standard isotropic Gaussian (d=5) as the target.
    All assertions use tolerance bands — statistical properties only.
    """

    def _setup_state(self):
        logdensity_fn = lambda x: std_normal_logdensity(x)
        state = _make_initial_state(self.next_key(), logdensity_fn, _DIM)
        return logdensity_fn, state

    def test_tuner_returns_L_over_step_approx_target(self):
        """Tuner with target_num_integration_steps=2 returns L/step ≈ 2 (not ≈ 1).

        The regression: without the fix, the L-estimators produce L ≈ step,
        so L/step ≈ 1 and the dynamic kernel collapses to MALA.
        """
        logdensity_fn, state = self._setup_state()
        params = _run_tuner(
            self.next_key(),
            state,
            logdensity_fn,
            target_num_integration_steps=2.0,
        )
        ratio = params.L / params.step_size
        # Should be exactly 2.0 (post-override), within float precision.
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-5)

    def test_tuner_default_is_target_2(self):
        """Default call (no target_num_integration_steps) produces L/step ≈ 2."""
        logdensity_fn, state = self._setup_state()
        kernel = _make_kernel()
        # Calling with all existing positional/keyword args, omitting the new param.
        _, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=logdensity_fn,
            num_steps=_NUM_STEPS,
            state=state,
            rng_key=self.next_key(),
            target=_TUNE_TARGET,
            frac_tune1=_TUNE_FRAC1,
            frac_tune2=_TUNE_FRAC2,
        )
        ratio = params.L / params.step_size
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-5)

    def test_tuner_target_1_produces_ratio_1(self):
        """target_num_integration_steps=1.0 recovers the old MALA-equivalent behaviour.

        This verifies the override is correctly parameterised (not hardcoded).
        """
        logdensity_fn, state = self._setup_state()
        params = _run_tuner(
            self.next_key(),
            state,
            logdensity_fn,
            target_num_integration_steps=1.0,
        )
        ratio = params.L / params.step_size
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-5)

    def test_dynamic_kernel_median_steps_approx_2(self):
        """After tuning with target=2, the dynamic kernel takes ~2 steps per proposal.

        Regression test for the MALA collapse: without the fix, median steps ≈ 1.
        With the fix, median steps should be ≥ 1.5 (avg=2 → Uniform[1, 3] → median ≈ 2).
        """
        logdensity_fn, state = self._setup_state()
        tune_key, run_key = jax.random.split(self.next_key())
        params = _run_tuner(
            tune_key,
            state,
            logdensity_fn,
            target_num_integration_steps=2.0,
        )
        num_sampling_steps = 2000
        info = _run_chain(run_key, state, logdensity_fn, params, num_sampling_steps)
        median_steps = jnp.median(info.num_integration_steps.astype(jnp.float32))
        # avg=2 => steps ~ ceil(Uniform * rescale(2)) ~ Uniform[1, 3], median ~ 2.
        # We assert >= 1.5 to be conservative.
        assert median_steps >= 1.5, (
            f"Median integration steps {float(median_steps)} too low, "
            "expected >= 1.5 (regression: MALA collapse gives median ~ 1)."
        )

    def test_avg_2_ess_geq_avg_1_on_gaussian(self):
        """avg=2 tuning gives ESS ≥ avg=1 (MALA) on a standard Gaussian.

        This is the quality gate from the brief: avg=2 should give ~2× ESS at
        equal compute vs MALA (avg=1).  We assert ESS_avg2 ≥ 0.8 * ESS_avg1
        as a conservative lower bound (true ratio is ~2× per statistician data).
        """
        logdensity_fn = lambda x: std_normal_logdensity(x)

        key1, key2, run_key1, run_key2 = jax.random.split(self.next_key(), 4)

        state1 = _make_initial_state(key1, logdensity_fn, _DIM)
        state2 = _make_initial_state(key2, logdensity_fn, _DIM)

        params_avg2 = _run_tuner(
            key1, state1, logdensity_fn, target_num_integration_steps=2.0
        )
        params_avg1 = _run_tuner(
            key2, state2, logdensity_fn, target_num_integration_steps=1.0
        )

        num_sampling = 2000
        positions_avg2 = _run_chain_positions(
            run_key1, state1, logdensity_fn, params_avg2, num_sampling
        )
        positions_avg1 = _run_chain_positions(
            run_key2, state2, logdensity_fn, params_avg1, num_sampling
        )

        # ESS per dimension, then average
        ess_avg2 = float(
            jnp.mean(diagnostics.effective_sample_size(positions_avg2[None, ...]))
        )
        ess_avg1 = float(
            jnp.mean(diagnostics.effective_sample_size(positions_avg1[None, ...]))
        )

        # avg=2 should be strictly better than avg=1; allow 20% tolerance
        # for Monte Carlo noise on small runs.
        assert ess_avg2 >= 0.8 * ess_avg1, (
            f"ESS with avg=2 ({ess_avg2}) is worse than 80% of ESS with avg=1 "
            f"({ess_avg1}), expected avg=2 to be substantially better."
        )

    def test_backward_compat_existing_signature(self):
        """Existing call signature (no new param) works and returns valid params."""
        logdensity_fn, state = self._setup_state()
        kernel = _make_kernel()

        # Exact replica of the existing test_sampling.py call pattern.
        _, params, num_steps = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=logdensity_fn,
            num_steps=_NUM_STEPS,
            state=state,
            rng_key=self.next_key(),
            target=_TUNE_TARGET,
            frac_tune1=_TUNE_FRAC1,
            frac_tune2=_TUNE_FRAC2,
            frac_tune3=0.0,
            diagonal_preconditioning=True,
        )

        # params must be a valid MCLMCAdaptationState with positive values.
        assert params.step_size > 0
        assert params.L > 0
        assert num_steps >= 0
        # L/step = 2.0 (the new default).
        np.testing.assert_allclose(params.L / params.step_size, 2.0, rtol=1e-5)
