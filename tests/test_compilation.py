"""Make sure that the log probability function is only compiled/traced once.

These are very important regression tests! JIT-compilation dominates the
total sampling time in many situations, and we need to make sure that
internal changes do not trigger more compilations than is necessary.

"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.stats as jstats
from absl.testing import absltest

import blackjax
from blackjax.mcmc.hmc import multinomial_hmc_proposal
from blackjax.util import run_inference_algorithm


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

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
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

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
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

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = kernel(sample_key, state)

    def test_multinomial_hmc(self):
        """Count the number of times the logdensity is compiled when using
        Multinomial HMC via hmc.build_kernel with proposal_generator.

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
            build_proposal=multinomial_hmc_proposal,
        )
        step = jax.jit(kernel.step)

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = step(sample_key, state)

    def test_multinomial_hmc_warmup(self):
        """Count the number of times the logdensity is compiled when using
        window adaptation for the Multinomial HMC algorithm via the
        top-level blackjax.multinomial_hmc alias.

        The logdensity is compiled three times: once during init, once
        for the warmup kernel inside window_adaptation.run, and once
        for the post-warmup sampling kernel.

        """

        @chex.assert_max_traces(n=3)
        def logdensity_fn(x):
            return jscipy.stats.norm.logpdf(x)

        chex.clear_trace_counter()

        rng_key = jax.random.key(0)

        warmup = blackjax.window_adaptation(
            algorithm=blackjax.multinomial_hmc,
            logdensity_fn=logdensity_fn,
            target_acceptance_rate=0.8,
            num_integration_steps=10,
        )
        (state, parameters), _ = warmup.run(rng_key, 1.0, num_steps=100)
        kernel = jax.jit(blackjax.multinomial_hmc(logdensity_fn, **parameters).step)

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
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

        for i in range(10):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = step(sample_key, state)


def _regression_logdensity_fn():
    """Tiny 2-parameter regression model, reused by the jit-boundary guards
    below (shape-wise representative of the benchmark model in
    tests/test_benchmarks.py without its 100_000-row cost)."""
    x_data = jax.random.normal(jax.random.key(0), (100, 1))
    y_data = 3 * x_data

    def regression_logprob(log_scale, coefs, preds, x):
        scale = jnp.exp(log_scale)
        scale_prior = jstats.expon.logpdf(scale, 0, 1) + log_scale
        coefs_prior = jstats.norm.logpdf(coefs, 0, 5)
        y = jnp.dot(x, coefs)
        logpdf = jstats.norm.logpdf(preds, y, scale)
        return sum(v.sum() for v in [scale_prior, coefs_prior, logpdf])

    logdensity_fn_ = functools.partial(regression_logprob, x=x_data, preds=y_data)
    return lambda position: logdensity_fn_(**position)


_JIT_PRIMITIVE_NAMES = {"jit", "pjit", "closed_call"}
_LOOP_PRIMITIVE_NAMES = {"scan", "while"}


def _jaxpr_contains_loop(jaxpr) -> bool:
    """Recursively search a (Closed)Jaxpr's equations -- including inside
    nested jit/pjit/closed_call sub-jaxprs -- for a scan/while primitive."""
    jaxpr = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in _LOOP_PRIMITIVE_NAMES:
            return True
        for value in eqn.params.values():
            candidates = value if isinstance(value, (list, tuple)) else (value,)
            for candidate in candidates:
                sub_jaxpr = getattr(candidate, "jaxpr", None)
                if sub_jaxpr is not None and _jaxpr_contains_loop(sub_jaxpr):
                    return True
    return False


def _has_jit_wrapped_loop(jaxpr) -> bool:
    """True iff the loop was dispatched THROUGH an explicit `jax.jit`
    boundary, not eagerly.

    A bare `jit`/`pjit` equation ANYWHERE in the jaxpr is not sufficient --
    plenty of unrelated internal helpers (e.g. value_and_grad of the
    logdensity) are themselves jitted and would make a naive "any jit
    equation present" check pass vacuously even on the un-jitted code. This
    checks the TOP-LEVEL equations specifically for:
      (a) no bare top-level `scan`/`while` equation (that would mean the
          loop itself is dispatched eagerly, outside any jit), and
      (b) at least one top-level `jit`/`pjit`/`closed_call` equation whose
          (possibly nested) body contains a `scan`/`while` primitive.
    """
    top = jaxpr.jaxpr
    if any(eqn.primitive.name in _LOOP_PRIMITIVE_NAMES for eqn in top.eqns):
        return False
    for eqn in top.eqns:
        if eqn.primitive.name in _JIT_PRIMITIVE_NAMES:
            sub_jaxpr = eqn.params.get("jaxpr")
            if sub_jaxpr is not None and _jaxpr_contains_loop(sub_jaxpr):
                return True
    return False


class EagerScanJitBoundaryTest(chex.TestCase):
    """Regression guard for the eager-`lax.scan`-dispatch CPU slowdown on
    jax>=0.11 (jax-ml/jax#37465, ~2-3x per call): several public BlackJAX
    entry points run their own internal `lax.scan` / `lax.fori_loop`
    eagerly (no enclosing `jax.jit`) when a caller invokes them outside
    their own `jax.jit`. The fix wraps each such loop in an explicit
    `jax.jit`-decorated closure -- see blackjax/util.py,
    blackjax/adaptation/staged_adaptation.py, and the other sites listed in
    the PR that introduced this test.

    These tests are intentionally NOT timing-based (timing assertions are
    flaky in CI); instead they inspect the jaxpr produced by calling the
    entry point and assert the loop was dispatched through an explicit
    `jax.jit` boundary -- see `_has_jit_wrapped_loop`.
    """

    def test_run_inference_algorithm_is_jit_wrapped(self):
        logdensity_fn = _regression_logdensity_fn()
        nuts = blackjax.nuts(
            logdensity_fn, step_size=0.1, inverse_mass_matrix=jnp.eye(2)
        )
        state = nuts.init({"log_scale": 0.0, "coefs": 2.0})

        def call():
            return run_inference_algorithm(
                rng_key=jax.random.key(1),
                initial_state=state,
                inference_algorithm=nuts,
                num_steps=10,
            )

        jaxpr = jax.make_jaxpr(call)()
        self.assertTrue(
            _has_jit_wrapped_loop(jaxpr),
            "run_inference_algorithm's internal lax.scan must stay wrapped "
            "in jax.jit (blackjax/util.py) -- un-jitted lax.scan dispatch "
            "is ~2-3x slower per call on jax>=0.11 (jax-ml/jax#37465).",
        )

    def test_window_adaptation_run_is_jit_wrapped(self):
        logdensity_fn = _regression_logdensity_fn()

        def call():
            warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
            return warmup.run(
                jax.random.key(0), {"log_scale": 0.0, "coefs": 2.0}, num_steps=10
            )

        jaxpr = jax.make_jaxpr(call)()
        self.assertTrue(
            _has_jit_wrapped_loop(jaxpr),
            "window_adaptation(...).run's internal lax.scan must stay "
            "wrapped in jax.jit "
            "(blackjax/adaptation/staged_adaptation.py) -- un-jitted "
            "lax.scan dispatch is ~2-3x slower per call on jax>=0.11 "
            "(jax-ml/jax#37465).",
        )

    def test_mclmc_find_l_and_step_size_is_jit_wrapped(self):
        logdensity_fn = _regression_logdensity_fn()
        kernel = blackjax.mcmc.mclmc.build_kernel(
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan
        )
        init_key, tune_key = jax.random.split(jax.random.key(0))
        state = blackjax.mcmc.mclmc.init(
            {"log_scale": 0.0, "coefs": 2.0}, logdensity_fn, init_key
        )

        def call():
            return blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                logdensity_fn=logdensity_fn,
                num_steps=10,
                state=state,
                rng_key=tune_key,
            )

        jaxpr = jax.make_jaxpr(call)()
        self.assertTrue(
            _has_jit_wrapped_loop(jaxpr),
            "mclmc_find_L_and_step_size's internal lax.scan (run_steps) "
            "must stay wrapped in jax.jit "
            "(blackjax/adaptation/mclmc_adaptation.py) -- un-jitted "
            "lax.scan dispatch is ~2-3x slower per call on jax>=0.11 "
            "(jax-ml/jax#37465).",
        )


if __name__ == "__main__":
    absltest.main()
