import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc import metrics
from blackjax.mcmc.barker import _barker_pdf, _barker_sample, _log1pexp
from blackjax.util import run_inference_algorithm


class BarkerSamplingTest(chex.TestCase):
    @parameterized.parameters([1234, 5678])
    def test_nd(self, seed):
        n_samples = 100_000

        key = jax.random.key(seed)
        m, a, scale = (
            jnp.array([1.0, 0.5, 0.0, 0.0]),
            jnp.array([1.0, -2.0, 10.0, 0.0]),
            0.5,
        )

        metric = metrics.default_metric(jnp.eye(4))
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: _barker_sample(k, m, a, scale, metric))(keys)
        # Check that the emprical mean and the mean computed as sum(x * p(x) dx) are close
        _test_samples_vs_pdf(samples, lambda x: _barker_pdf(x, m, a, scale))


def _test_samples_vs_pdf(samples, pdf):
    samples_mean = jnp.mean(samples, 0)
    samples_squrared_mean = jnp.mean(samples**2, 0)
    linspace = jnp.linspace(-10, 10, 50_000)

    diff = jnp.diff(linspace, axis=0)

    # trapezoidal rule
    pdf_mean = 0.5 * jnp.sum(
        linspace[1:, None] * pdf(linspace[1:, None]) * diff[:, None], 0
    )
    pdf_mean += 0.5 * jnp.sum(
        linspace[:-1, None] * pdf(linspace[:-1, None]) * diff[:, None], 0
    )
    pdf_squared_mean = 0.5 * jnp.sum(
        linspace[1:, None] ** 2 * pdf(linspace[1:, None]) * diff[:, None], 0
    )
    pdf_squared_mean += 0.5 * jnp.sum(
        linspace[:-1, None] ** 2 * pdf(linspace[:-1, None]) * diff[:, None], 0
    )

    chex.assert_trees_all_close(samples_mean, pdf_mean, atol=1e-2, rtol=1e-2)
    chex.assert_trees_all_close(
        samples_squrared_mean, pdf_squared_mean, atol=1e-2, rtol=1e-2
    )


class BarkerPreconditioiningTest(chex.TestCase):
    @parameterized.parameters([1234, 5678])
    def test_preconditioning_matrix(self, seed):
        """Test two different ways of using pre-conditioning matrix has exactly same effect.

        We follow the discussion in Appendix G of the Barker 2020 paper.
        """

        key = jax.random.key(seed)
        init_key, inference_key = jax.random.split(key, 2)

        # setup some 2D multivariate normal model
        # setup sampling mean and cov
        true_x = jnp.array([0.0, 1.0])
        data = jax.random.normal(init_key, shape=(1000,)) * true_x[1] + true_x[0]
        assert data.shape == (1000,)

        # some non-diagonal positive-defininte matrix for pre-conditioning
        inv_mass_matrix = jnp.array([[1, 0.1], [0.1, 1]])
        metric = metrics.default_metric(inv_mass_matrix)

        # define barker kernel two ways
        # non-scaled, use pre-conditioning
        def logdensity(x, data):
            mu_prior = stats.norm.logpdf(x[0], loc=0, scale=1)
            sigma_prior = stats.uniform.logpdf(x[1], 0.0, 3.0)
            return mu_prior + sigma_prior + jnp.sum(stats.norm.logcdf(data, x[0], x[1]))

        logposterior_fn1 = functools.partial(logdensity, data=data)
        barker1 = blackjax.barker(logposterior_fn1, 1e-1, inv_mass_matrix)
        state1 = barker1.init(true_x)

        # scaled, trivial pre-conditioning
        def scaled_logdensity(x_scaled, data, metric):
            x = metric.scale(x_scaled, x_scaled, inv=False, trans=False)
            return logdensity(x, data)

        logposterior_fn2 = functools.partial(
            scaled_logdensity, data=data, metric=metric
        )
        barker2 = blackjax.barker(logposterior_fn2, 1e-1, jnp.eye(2))

        true_x_trans = metric.scale(true_x, true_x, inv=True, trans=True)
        state2 = barker2.init(true_x_trans)

        n_steps = 10
        _, states1 = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=state1,
            inference_algorithm=barker1,
            transform=lambda state, info: state.position,
            num_steps=n_steps,
        )

        _, states2 = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=state2,
            inference_algorithm=barker2,
            transform=lambda state, info: state.position,
            num_steps=n_steps,
        )

        # states should be the exact same with same random key after transforming
        states2_trans = []
        for ii in range(n_steps):
            s = states2[ii]
            states2_trans.append(metric.scale(s, s, inv=False, trans=False))
        states2_trans = jnp.array(states2_trans)
        assert jnp.allclose(states1, states2_trans)

    @parameterized.parameters(
        itertools.product([1234, 5678], ["gaussian", "riemannian"])
    )
    def test_invariance(self, seed, metric):
        logpdf = lambda x: -0.5 * jnp.sum(x**2)

        n_samples, m_steps = 10_000, 50

        key = jax.random.key(seed)
        init_key, inference_key = jax.random.split(key, 2)
        inference_keys = jax.random.split(inference_key, n_samples)
        if metric == "gaussian":
            inv_mass_matrix = jnp.ones((2,))
            metric = metrics.default_metric(inv_mass_matrix)
        else:
            # bit of a random metric but we are testing invariance, not efficiency
            metric = metrics.gaussian_riemannian(
                lambda x: 1 / jnp.sum(1 + jnp.sum(x**2)) * jnp.eye(2)
            )

        barker = blackjax.barker(logpdf, 0.5, metric)
        init_samples = jax.random.normal(init_key, shape=(n_samples, 2))

        def loop(carry, key_):
            state, accepted = carry
            state, info = barker.step(key_, state)
            accepted += info.is_accepted
            return (state, accepted), None

        def get_samples(init_sample, key_):
            init = (barker.init(init_sample), 0)
            (out, n_accepted), _ = jax.lax.scan(
                loop, init, jax.random.split(key_, m_steps)
            )
            return out.position, n_accepted / m_steps

        samples, total_accepted = jax.vmap(get_samples)(init_samples, inference_keys)
        # now we test the distance versus a Gaussian
        chex.assert_trees_all_close(
            jnp.mean(samples, 0), jnp.zeros((2,)), atol=1e-1, rtol=1e-1
        )
        chex.assert_trees_all_close(
            jnp.cov(samples.T), jnp.eye(2), atol=1e-1, rtol=1e-1
        )


class BarkerNumericalStabilityTest(chex.TestCase):
    @parameterized.product(dtype=[jnp.float32, jnp.float64], use_jit=[False, True])
    def test_log1pexp_values_and_gradients(self, dtype, use_jit):
        with jax.enable_x64(dtype == jnp.float64):
            values = jnp.array(
                [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0], dtype=dtype
            )
            value_and_grad = jax.value_and_grad(lambda x: _log1pexp(x).sum())
            if use_jit:
                value_and_grad = jax.jit(value_and_grad)
            actual, gradient = value_and_grad(values)
            host = np.asarray(values, dtype=np.float64)
            expected = np.logaddexp(0.0, host).sum()
            # Stable sigmoid, including both saturated tails and its derivative at zero.
            expected_grad = np.exp(-np.logaddexp(0.0, -host))
            chex.assert_trees_all_close(actual, expected, rtol=1e-6)
            chex.assert_trees_all_close(gradient, expected_grad, atol=1e-7, rtol=1e-6)

    @parameterized.product(
        dtype=[jnp.float32, jnp.float64], use_jit=[False, True], pytree=[False, True]
    )
    def test_accept_improving_proposal_with_large_gradient(
        self, dtype, use_jit, pytree
    ):
        with jax.enable_x64(dtype == jnp.float64):
            # A finite Gaussian state can have logistic corrections beyond exp's range.
            # Use a larger initial position in float64 to exercise that dtype's range too.
            initial = 20.0 if dtype == jnp.float32 else 60.0
            position = jnp.array([initial], dtype=dtype)
            if pytree:
                position = {"x": position}
            logdensity = lambda x: -0.5 * sum(jnp.sum(v**2) for v in jax.tree.leaves(x))
            algorithm = blackjax.barker(logdensity, step_size=initial / 2)
            state = algorithm.init(position)
            step = jax.jit(algorithm.step) if use_jit else algorithm.step
            new_state, info = step(jax.random.key(2), state)
            proposed = float(jax.tree.leaves(info.proposal.position)[0][0])
            delta = proposed - initial
            # Independent Gaussian target and forward/reverse Barker proposal ratio.
            log_accept = (
                0.5 * (initial**2 - proposed**2)
                + np.logaddexp(0.0, initial * delta)
                - np.logaddexp(0.0, -proposed * delta)
            )
            self.assertGreater(log_accept, 0.0)
            self.assertEqual(float(info.acceptance_rate), 1.0)
            self.assertTrue(bool(info.is_accepted))
            chex.assert_trees_all_equal(new_state.position, info.proposal.position)


if __name__ == "__main__":
    absltest.main()
