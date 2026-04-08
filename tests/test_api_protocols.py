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
"""Test that every public sampler's top-level API conforms to the declared
protocols in ``blackjax.base``.

The ``InitFn`` protocol is ``(position, rng_key=None) -> State`` and
``UpdateFn`` is ``(rng_key, state) -> (State, Info)``.  These tests verify
that each algorithm can be called through the ``SamplingAlgorithm`` interface
without errors.
"""
import inspect

import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.base import SamplingAlgorithm
from tests.fixtures import std_normal_logdensity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DIM = 2
_POSITION = jnp.ones(_DIM)


def _make_algorithm(name):
    """Instantiate a top-level ``SamplingAlgorithm`` for each public sampler."""
    cov = jnp.eye(_DIM)
    inv_mass = jnp.ones(_DIM)

    factories = {
        # --- MCMC ---
        "hmc": lambda: blackjax.hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=inv_mass,
            num_integration_steps=10,
        ),
        "nuts": lambda: blackjax.nuts(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=inv_mass,
        ),
        "mala": lambda: blackjax.mala(std_normal_logdensity, step_size=0.1),
        "mclmc": lambda: blackjax.mclmc(
            std_normal_logdensity,
            L=1.0,
            step_size=0.1,
        ),
        "adjusted_mclmc": lambda: blackjax.adjusted_mclmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=inv_mass,
            num_integration_steps=10,
        ),
        "adjusted_mclmc_dynamic": lambda: blackjax.adjusted_mclmc_dynamic(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=inv_mass,
        ),
        "barker": lambda: blackjax.barker(std_normal_logdensity, step_size=0.1),
        "dynamic_hmc": lambda: blackjax.dynamic_hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=inv_mass,
        ),
        "rmhmc": lambda: blackjax.rmhmc(
            std_normal_logdensity,
            step_size=0.1,
            mass_matrix=cov,
            num_integration_steps=10,
        ),
        "ghmc": lambda: blackjax.ghmc(
            std_normal_logdensity,
            step_size=0.1,
            momentum_inverse_scale=inv_mass,
            alpha=0.5,
            delta=0.5,
        ),
        "elliptical_slice": lambda: blackjax.elliptical_slice(
            std_normal_logdensity,
            mean=jnp.zeros(_DIM),
            cov=cov,
        ),
        "additive_step_random_walk": lambda: (
            blackjax.additive_step_random_walk.normal_random_walk(
                std_normal_logdensity,
                sigma=0.1 * cov,
            )
        ),
        "rmh": lambda: blackjax.rmh(
            std_normal_logdensity,
            proposal_generator=lambda key, x: x + 0.1 * jax.random.normal(key, x.shape),
        ),
        "irmh": lambda: blackjax.irmh(
            std_normal_logdensity,
            proposal_distribution=lambda key: jax.random.normal(key, (_DIM,)),
        ),
    }

    if name not in factories:
        return None
    return factories[name]()


# Algorithms whose init requires rng_key (not None)
_NEEDS_RNG_KEY = {"mclmc", "ghmc", "adjusted_mclmc_dynamic", "dynamic_hmc"}

# All MCMC algorithms we test
_MCMC_ALGORITHMS = [
    "hmc",
    "nuts",
    "mala",
    "mclmc",
    "adjusted_mclmc",
    "adjusted_mclmc_dynamic",
    "barker",
    "dynamic_hmc",
    "rmhmc",
    "ghmc",
    "elliptical_slice",
    "additive_step_random_walk",
    "rmh",
    "irmh",
]


@pytest.mark.parametrize("name", _MCMC_ALGORITHMS)
class TestSamplingAlgorithmProtocol:
    """Verify that each MCMC algorithm's top-level API produces a valid
    SamplingAlgorithm and conforms to InitFn / UpdateFn protocols."""

    def test_returns_sampling_algorithm(self, name):
        """The factory should return a SamplingAlgorithm NamedTuple."""
        alg = _make_algorithm(name)
        if alg is None:
            pytest.skip(f"{name} not covered by generic factory")
        assert isinstance(
            alg, SamplingAlgorithm
        ), f"{name} factory did not return a SamplingAlgorithm"

    def test_init_step_roundtrip(self, name):
        """init -> step should execute without error and return (state, info)."""
        alg = _make_algorithm(name)
        if alg is None:
            pytest.skip(f"{name} not covered by generic factory")

        init_key, step_key = jax.random.split(jax.random.key(0))
        if name in _NEEDS_RNG_KEY:
            state = alg.init(_POSITION, init_key)
        else:
            state = alg.init(_POSITION)

        new_state, info = alg.step(step_key, state)
        assert new_state is not None
        assert info is not None

    def test_init_fn_first_param_is_position(self, name):
        """The init function's first parameter should be named 'position'."""
        alg = _make_algorithm(name)
        if alg is None:
            pytest.skip(f"{name} not covered by generic factory")

        sig = inspect.signature(alg.init)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, f"{name}.init has no parameters"
        assert (
            params[0] == "position"
        ), f"{name}.init first param is '{params[0]}', expected 'position'"

    def test_step_fn_first_two_params(self, name):
        """The step function should accept (rng_key, state) as first two params."""
        alg = _make_algorithm(name)
        if alg is None:
            pytest.skip(f"{name} not covered by generic factory")

        sig = inspect.signature(alg.step)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, f"{name}.step has fewer than 2 parameters"
        assert (
            params[0] == "rng_key"
        ), f"{name}.step first param is '{params[0]}', expected 'rng_key'"
        assert (
            params[1] == "state"
        ), f"{name}.step second param is '{params[1]}', expected 'state'"
