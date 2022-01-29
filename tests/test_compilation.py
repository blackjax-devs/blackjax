"""Make sure that the log probability function is only compiled/traced once.
"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from absl.testing import absltest

import blackjax


class CompilationTest(chex.TestCase):
    def test_hmc(self):
        @chex.assert_max_traces(n=1)
        def logprob_fn(x):
            return jscipy.stats.norm.logpdf(x)

        rng_key = jax.random.PRNGKey(0)
        state = blackjax.hmc.init(1.0, logprob_fn)

        chex.clear_trace_counter()

        kernel = blackjax.hmc(
            logprob_fn,
            step_size=1e-2,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        step = jax.jit(kernel.step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)

    def test_nuts(self):
        # Log probability function was traced twice as we call it
        # at Step 0 when building a new trajectory in tree doubling.
        @chex.assert_max_traces(n=1)
        def logprob_fn(x):
            return jscipy.stats.norm.logpdf(x)

        rng_key = jax.random.PRNGKey(0)
        state = blackjax.nuts.init(1.0, logprob_fn)

        chex.clear_trace_counter()

        kernel = blackjax.nuts(
            logprob_fn, step_size=1e-2, inverse_mass_matrix=jnp.array([1.0])
        )
        step = jax.jit(kernel.step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)

    def test_warmup(self):
        # Log probability function was traced twice as we call it
        # at Step 0 when building a new trajectory in tree doubling.
        @chex.assert_max_traces(n=4)
        def logprob_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.PRNGKey(0)

        warmup = blackjax.window_adaptation(
            algorithm=blackjax.nuts,
            logprob_fn=logprob_fn,
            target_acceptance_rate=0.8,
        )
        state, kernel, _ = warmup.run(rng_key, 1.0, 1000)
        step = jax.jit(kernel)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)


if __name__ == "__main__":
    absltest.main()
