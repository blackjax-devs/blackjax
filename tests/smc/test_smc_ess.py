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
        # NOTE: ``ess_solver`` expects a log-density (positive sign — same as
        # the log-likelihood passed by ``adaptive_tempered_smc``), NOT a
        # potential. Passing ``-logpdf`` here would silently work for
        # symmetric distributions and mask the sign bug in #914.
        num_particles = 1000
        logdensity_fn = jax.vmap(lambda x: univariate_logpdf(x, scale=0.1), in_axes=[0])
        particles = np.random.normal(0, 1, size=(num_particles, 1))
        self.ess_solver_test_case(
            logdensity_fn, particles, target_ess, num_particles, 1.0
        )

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
        logdensity_fn = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = np.random.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
        )
        self.ess_solver_test_case(
            logdensity_fn, particles, target_ess, num_particles, 10.0
        )

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

        logdensity_fn = jax.vmap(_logdensity_fn, in_axes=[0], out_axes=0)
        particles = [
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
            np.random.multivariate_normal(
                mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_particles
            ),
        ]
        self.ess_solver_test_case(
            logdensity_fn, particles, target_ess, num_particles, 10.0
        )

    def ess_solver_test_case(self, logdensity_fn, particles, target_ess, N, max_delta):
        ess_solver_fn = functools.partial(
            ess.ess_solver,
            logdensity_fn,
            target_ess=target_ess,
            max_delta=max_delta,
            root_solver=solver.dichotomy,
        )

        delta = self.variant(ess_solver_fn)(particles)
        assert delta > 0

        # Verify the solver's solution against the same weight expression
        # the SMC kernel uses (``delta * loglikelihood``, see
        # blackjax/smc/tempered.py:log_weights_fn). Using the wrong sign
        # here would re-introduce the silent #914 cancellation.
        ess_val = ess.ess(delta * logdensity_fn(particles))
        np.testing.assert_allclose(ess_val, target_ess * N, atol=1e-1, rtol=1e-2)

    @chex.all_variants(with_pmap=False)
    def test_ess_solver_asymmetric_loglikelihood_issue_914(self):
        """Regression test for the sign bug in #914.

        With a Cauchy prior and a sharply concentrated Gaussian likelihood
        centred away from 0, the prior-IS estimator already achieves an ESS
        well above the target with ``delta=1.0`` (one-step IS suffices, no
        tempering needed). The bisection must therefore return
        ``delta = max_delta = 1.0``. Before the #914 fix, the wrong sign
        made the bisection report ``delta ~ 5e-8``, which caused
        ``adaptive_tempered_smc`` to stall at ``lambda ~ 0``.

        We choose ``max_delta = 1.0`` so the boundary case ``delta == 1.0``
        is in-range; the asymmetric log-likelihood values (chi-squared-like,
        not invariant under sign flip) ensure the bug cannot hide.
        """
        N = 8192
        target_ess = 0.9 * 1024 / N  # ~0.1125

        key = jax.random.key(0)
        u = jax.random.uniform(key, (N,))
        # Cauchy prior via inverse-CDF.
        particles = jnp.tan(jnp.pi * (u - 0.5))
        # Gaussian log-likelihood centred at mu=2, sigma=0.5 — asymmetric in
        # the particle index, with a long left tail of very negative values.
        loglikelihood_fn = lambda x: -0.5 * ((x - 2.0) / 0.5) ** 2

        delta = self.variant(
            functools.partial(
                ess.ess_solver,
                lambda _particles: loglikelihood_fn(_particles),
                target_ess=target_ess,
                max_delta=1.0,
                root_solver=solver.dichotomy,
            )
        )(particles)

        # Cross-check via the closed-form posterior IS ESS estimator
        # (one-step reweighting from prior to posterior).
        from jax.scipy.special import logsumexp

        ll = loglikelihood_fn(particles)
        ess_posterior = float(jnp.exp(2 * logsumexp(ll) - logsumexp(2 * ll)))
        assert (
            ess_posterior > target_ess * N
        ), "Test premise broken: prior-IS ESS must already exceed target."

        # The bisection should return (close to) max_delta.
        np.testing.assert_allclose(float(delta), 1.0, atol=1e-2)


if __name__ == "__main__":
    absltest.main()
