"""Test the ess function"""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.scipy.stats.multivariate_normal import logpdf as multivariate_logpdf
from jax.scipy.stats.norm import logpdf as univariate_logpdf

import blackjax.smc.ess as ess
import blackjax.smc.solver as solver


class SMCEffectiveSampleSizeTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([1000, 5000])
    def test_ess(self, N):
        log_ess_fn = self.variant(functools.partial(ess.ess, log=True))
        ess_fn = self.variant(functools.partial(ess.ess, log=False))

        w = np.random.rand(N)
        log_w = np.log(w)
        log_ess_val = log_ess_fn(log_w)
        ess_val = ess_fn(log_w)
        np.testing.assert_almost_equal(np.log(ess_val), log_ess_val, decimal=3)

        normalized_w = w / w.sum()
        log_normalized_w = np.log(normalized_w)
        log_normalized_ess_val = log_ess_fn(log_normalized_w)
        normalized_ess_val = ess_fn(log_normalized_w)

        np.testing.assert_almost_equal(log_ess_val, log_normalized_ess_val, decimal=3)
        np.testing.assert_almost_equal(ess_val, normalized_ess_val, decimal=3)
        np.testing.assert_almost_equal(
            ess_val, 1 / np.sum(normalized_w**2), decimal=3
        )

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product([0.25, 0.5], [1000, 5000]))
    def test_ess_solver(self, target_ess, N):
        potential_fn = lambda pytree: -univariate_logpdf(pytree, scale=0.1)
        potential = jax.vmap(lambda x: potential_fn(x), in_axes=[0])
        particles = np.random.normal(0, 1, size=(N, 1))
        self.ess_solver_test_case(potential, particles, target_ess, N, 1.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product([0.25, 0.5], [1000, 5000]))
    def test_ess_solver_multivariate(self, target_ess, N):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x) x ~ N(mean, cov) x in R^{2}
        """
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))
        potential_fn = lambda pytree: -multivariate_logpdf(pytree, mean=mean, cov=cov)
        potential = jax.vmap(lambda x: potential_fn(x), in_axes=[0], out_axes=0)
        particles = np.random.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=N
        )
        self.ess_solver_test_case(potential, particles, target_ess, N, 10.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product([0.25, 0.5], [1000, 5000]))
    def test_ess_solver_posterior_signature(self, target_ess, N):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
        """
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))

        def potential_fn(pytree):
            return -multivariate_logpdf(
                pytree[0], mean=mean, cov=cov
            ) - multivariate_logpdf(pytree[1], mean=mean, cov=cov)

        potential = jax.vmap(potential_fn, in_axes=[0], out_axes=0)
        particles = [
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=N
            ),
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=N
            ),
        ]
        self.ess_solver_test_case(potential, particles, target_ess, N, 10.0)

    def ess_solver_test_case(self, potential, particles, target_ess, N, max_delta):
        ess_solver_fn = functools.partial(
            ess.ess_solver,
            potential,
            target_ess=target_ess,
            max_delta=max_delta,
            root_solver=solver.dichotomy,
        )

        log_ess_solver = self.variant(
            functools.partial(ess_solver_fn, use_log_ess=True)
        )
        ess_solver = self.variant(functools.partial(ess_solver_fn, use_log_ess=False))
        delta_log = log_ess_solver(particles)
        delta = ess_solver(particles)
        assert delta_log > 0
        np.testing.assert_allclose(delta_log, delta, atol=1e-3, rtol=1e-3)
        log_ess = ess.ess(-delta_log * potential(particles), log=True)
        np.testing.assert_allclose(
            np.exp(log_ess), target_ess * N, atol=1e-1, rtol=1e-2
        )


if __name__ == "__main__":
    absltest.main()
