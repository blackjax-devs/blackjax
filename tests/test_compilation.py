"""Make sure that the log probability function is only compiled/traced once.
"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from absl.testing import absltest

import blackjax.hmc as hmc
import blackjax.nuts as nuts


class CompilationTest(chex.TestCase):
    def test_hmc(self):
        @chex.assert_max_traces(n=1)
        def logprob_fn(x):
            return jscipy.stats.norm.logpdf(x)

        rng_key = jax.random.PRNGKey(0)
        state = hmc.new_state(1.0, logprob_fn)

        chex.clear_trace_counter()

        kernel = jax.jit(
            hmc.kernel(
                logprob_fn,
                step_size=1e-2,
                inverse_mass_matrix=jnp.array([1.0]),
                num_integration_steps=10,
            )
        )

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = kernel(sample_key, state)

    def test_nuts(self):
        # Log probability function was traced twice as we call it
        # at Step 0 when building a new trajectory in tree doubling.
        @chex.assert_max_traces(n=2)
        def logprob_fn(x):
            return jscipy.stats.norm.logpdf(x)

        rng_key = jax.random.PRNGKey(0)
        state = hmc.new_state(1.0, logprob_fn)

        chex.clear_trace_counter()

        kernel = jax.jit(
            nuts.kernel(
                logprob_fn, step_size=1e-2, inverse_mass_matrix=jnp.array([1.0])
            )
        )

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = kernel(sample_key, state)


if __name__ == "__main__":
    absltest.main()
