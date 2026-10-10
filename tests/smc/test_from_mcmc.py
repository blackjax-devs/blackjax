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
"""Unit tests for blackjax.smc.from_mcmc."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc.base import SMCState, init
from blackjax.smc.from_mcmc import build_kernel, unshared_parameters_and_step_fn
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# unshared_parameters_and_step_fn
# ---------------------------------------------------------------------------


class UnsharedParametersTest(BlackJAXTest):
    """Tests for unshared_parameters_and_step_fn."""

    def _dummy_step_fn(self, **kwargs):
        return kwargs

    def test_shared_parameter_removed_from_unshared(self):
        """A parameter with shape[0]==1 is shared and not in unshared dict."""
        params = {
            "step_size": jnp.array([[1e-2]]),  # shape (1, 1) → shared
            "num_steps": jnp.array([[10], [20], [30]]),  # shape (3, 1) → unshared
        }
        unshared, _ = unshared_parameters_and_step_fn(params, self._dummy_step_fn)
        assert "step_size" not in unshared
        assert "num_steps" in unshared

    def test_unshared_parameter_shape_preserved(self):
        """Unshared parameters keep their original shape (minus the leading dim of 1)."""
        step_size = jnp.ones((1,)) * 1e-2  # shape (1,) → shared
        inv_mass = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # shape (2, 2)
        # shape[0] != 1 so it's unshared
        params = {"step_size": step_size, "inv_mass": inv_mass}
        unshared, _ = unshared_parameters_and_step_fn(params, self._dummy_step_fn)
        assert "inv_mass" in unshared
        np.testing.assert_array_equal(unshared["inv_mass"], inv_mass)

    def test_shared_parameter_bound_in_step_fn(self):
        """Shared parameters are bound into the returned step function."""
        calls = []

        def recording_step_fn(**kwargs):
            calls.append(kwargs)
            return {}

        step_size = jnp.array([5.0])  # shape (1,) → shared
        unshared_val = jnp.array([1.0, 2.0])  # shape (2,) → unshared

        params = {"step_size": step_size, "vals": unshared_val}
        _, bound_step_fn = unshared_parameters_and_step_fn(params, recording_step_fn)

        # Call the bound step function (no shared params needed)
        bound_step_fn()
        assert len(calls) == 1
        assert "step_size" in calls[0]
        # step_size was scalar-indexed (v[0, ...]) from shape (1,)
        np.testing.assert_array_equal(calls[0]["step_size"], jnp.array(5.0))

    def test_all_shared(self):
        """When all parameters are shared, unshared dict is empty."""
        params = {
            "a": jnp.ones((1, 3)),
            "b": jnp.ones((1,)),
        }
        unshared, _ = unshared_parameters_and_step_fn(params, self._dummy_step_fn)
        assert len(unshared) == 0

    def test_all_unshared(self):
        """When no parameter has shape[0]==1, unshared dict equals input."""
        params = {
            "a": jnp.ones((3, 3)),
            "b": jnp.ones((5,)),
        }
        unshared, _ = unshared_parameters_and_step_fn(params, self._dummy_step_fn)
        assert set(unshared.keys()) == set(params.keys())


# ---------------------------------------------------------------------------
# build_kernel (integration test with a simple Gaussian target)
# ---------------------------------------------------------------------------


class BuildKernelTest(BlackJAXTest):
    """Integration tests for build_kernel."""

    def test_step_returns_smc_state_and_info(self):
        """A single build_kernel step returns SMCState and SMCInfo."""
        num_particles = 50

        hmc_init = blackjax.hmc.init
        hmc_step = blackjax.hmc.build_kernel()

        hmc_parameters = {
            "step_size": jnp.ones((1,)) * 1e-2,
            "inverse_mass_matrix": jnp.eye(2)[None],  # shape (1, 2, 2)
            "num_integration_steps": jnp.ones((1,), dtype=int) * 5,
        }

        smc_step = build_kernel(hmc_step, hmc_init, resampling.systematic)

        init_key, step_key = jax.random.split(self.next_key())
        init_particles = jax.random.normal(init_key, (num_particles, 2))
        state = init(init_particles, {})

        new_state, info = smc_step(
            step_key,
            state,
            num_mcmc_steps=3,
            mcmc_parameters=hmc_parameters,
            logposterior_fn=std_normal_logdensity,
            log_weights_fn=std_normal_logdensity,
        )
        self.assertIsInstance(new_state, SMCState)
        self.assertEqual(new_state.particles.shape, (num_particles, 2))
        self.assertEqual(new_state.weights.shape, (num_particles,))

    def test_particles_change_after_step(self):
        """Particles should generally change after an SMC step."""
        num_particles = 30

        hmc_init = blackjax.hmc.init
        hmc_step = blackjax.hmc.build_kernel()

        hmc_parameters = {
            "step_size": jnp.ones((1,)) * 1e-2,
            "inverse_mass_matrix": jnp.eye(2)[None],
            "num_integration_steps": jnp.ones((1,), dtype=int) * 5,
        }

        smc_step = build_kernel(hmc_step, hmc_init, resampling.systematic)

        init_key, step_key = jax.random.split(self.next_key())
        init_particles = jax.random.normal(init_key, (num_particles, 2))
        state = init(init_particles, {})

        new_state, _ = smc_step(
            step_key,
            state,
            num_mcmc_steps=3,
            mcmc_parameters=hmc_parameters,
            logposterior_fn=std_normal_logdensity,
            log_weights_fn=std_normal_logdensity,
        )
        # At least some particles should have moved
        assert not jnp.allclose(new_state.particles, init_particles)

    def test_jit_compatible(self):
        """build_kernel step is JIT-compilable."""
        num_particles = 20

        hmc_init = blackjax.hmc.init
        hmc_step = blackjax.hmc.build_kernel()

        hmc_parameters = {
            "step_size": jnp.ones((1,)) * 1e-2,
            "inverse_mass_matrix": jnp.eye(2)[None],
            "num_integration_steps": jnp.ones((1,), dtype=int) * 3,
        }

        smc_step = jax.jit(
            build_kernel(hmc_step, hmc_init, resampling.systematic),
            static_argnums=(2, 4, 5),
        )

        init_key, step_key = jax.random.split(self.next_key())
        init_particles = jax.random.normal(init_key, (num_particles, 2))
        state = init(init_particles, {})

        new_state, _ = smc_step(
            step_key,
            state,
            3,  # num_mcmc_steps (static)
            hmc_parameters,
            std_normal_logdensity,  # static
            std_normal_logdensity,  # static
        )
        self.assertEqual(new_state.particles.shape, (num_particles, 2))


if __name__ == "__main__":
    absltest.main()
