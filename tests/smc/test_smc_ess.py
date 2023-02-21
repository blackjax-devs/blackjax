"""Test the ess function"""
import functools

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
    def test_ess(self):
        # All particles have zero weight but one
        weights = jnp.array([-jnp.inf, -jnp.inf, 0, -jnp.inf])
        ess_val = self.variant(ess.ess)(weights)
        assert ess_val == 1.0

        weights = jnp.ones(12)
        ess_val = self.variant(ess.ess)(weights)
        assert ess_val == 12

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver(self, target_ess):
        num_particles = 1000
        potential_fn = lambda pytree: -univariate_logpdf(pytree, scale=0.1)
        potential = jax.vmap(lambda x: potential_fn(x), in_axes=[0])
        particles = np.random.normal(0, 1, size=(num_particles, 1))
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 1.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver_multivariate(self, target_ess):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x) x ~ N(mean, cov) x in R^{2}
        """
        num_particles = 1000
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))
        _logdensity_fn = lambda pytree: multivariate_logpdf(pytree, mean=mean, cov=cov)
        potential = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = np.random.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
        )
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 10.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0.2, 0.95])
    def test_ess_solver_posterior_signature(self, target_ess):
        """
        Posterior with more than one variable. Let's assume we want to
        sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
        """
        num_particles = 1000
        mean = jnp.zeros((1, 2))
        cov = jnp.diag(jnp.array([1, 1]))

        def _logdensity_fn(pytree):
            return multivariate_logpdf(
                pytree[0], mean=mean, cov=cov
            ) + multivariate_logpdf(pytree[1], mean=mean, cov=cov)

        potential = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = [
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
        ]
        self.ess_solver_test_case(potential, particles, target_ess, num_particles, 10.0)

    def ess_solver_test_case(self, potential, particles, target_ess, N, max_delta):
        ess_solver_fn = functools.partial(
            ess.ess_solver,
            potential,
            target_ess=target_ess,
            max_delta=max_delta,
            root_solver=solver.dichotomy,
        )

        delta = self.variant(ess_solver_fn)(particles)
        assert delta > 0

        ess_val = ess.ess(-delta * potential(particles))
        np.testing.assert_allclose(ess_val, target_ess * N, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    absltest.main()
