"""Make sure that the log probability function is only compiled/traced once.

These are very important regression tests! JIT-compilation dominates the
total sampling time in many situations, and we need to make sure that
internal changes do not trigger more compilations than is necessary.

"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from absl.testing import absltest

import blackjax


class CompilationTest(chex.TestCase):
    def test_hmc(self):
        """Count the number of times the logdensity is compiled when using HMC.

        The logdensity is compiled twice: when initializing the state and when
        compiling the kernel.

        """

        @chex.assert_max_traces(n=2)
        def logdensity_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.key(0)
        state = blackjax.hmc.init(1.0, logdensity_fn)

        kernel = blackjax.hmc(
            logdensity_fn,
            step_size=1e-2,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        step = jax.jit(kernel.step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)

    def test_nuts(self):
        """Count the number of times the logdensity is compiled when using NUTS.

        The logdensity is compiled twice: when initializing the state and when
        compiling the kernel.

        """

        @chex.assert_max_traces(n=2)
        def logdensity_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.key(0)
        state = blackjax.nuts.init(1.0, logdensity_fn)

        kernel = blackjax.nuts(
            logdensity_fn, step_size=1e-2, inverse_mass_matrix=jnp.array([1.0])
        )
        step = jax.jit(kernel.step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)

    def test_hmc_warmup(self):
        """Count the number of times the logdensity is compiled when using window
        adaptation to adapt the value of the step size and the inverse mass
        matrix for the HMC algorithm.

        """

        @chex.assert_max_traces(n=3)
        def logdensity_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.key(0)

        warmup = blackjax.window_adaptation(
            algorithm=blackjax.hmc,
            logdensity_fn=logdensity_fn,
            target_acceptance_rate=0.8,
            num_integration_steps=10,
        )
        (state, parameters), _ = warmup.run(rng_key, 1.0, num_steps=100)
        kernel = jax.jit(blackjax.hmc(logdensity_fn, **parameters).step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = kernel(sample_key, state)

    def test_nuts_warmup(self):
        """Count the number of times the logdensity is compiled when using window
        adaptation to adapt the value of the step size and the inverse mass
        matrix for the NUTS algorithm.

        """

        @chex.assert_max_traces(n=3)
        def logdensity_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.key(0)

        warmup = blackjax.window_adaptation(
            algorithm=blackjax.nuts,
            logdensity_fn=logdensity_fn,
            target_acceptance_rate=0.8,
        )
        (state, parameters), _ = warmup.run(rng_key, 1.0, num_steps=100)
        step = jax.jit(blackjax.nuts(logdensity_fn, **parameters).step)

        for _ in range(10):
            rng_key, sample_key = jax.random.split(rng_key)
            state, _ = step(sample_key, state)


if __name__ == "__main__":
    absltest.main()
