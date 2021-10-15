"""Test the tempered SMC steps and routine"""
import functools
import itertools
from typing import List

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.hmc as hmc
import blackjax.inference.smc.resampling as resampling
import blackjax.inference.smc.solver as solver
from blackjax.tempered_smc import TemperedSMCState, adaptive_tempered_smc, tempered_smc


def inference_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.lmbda < 1

    def body(carry):
        i, state, op_key, curr_loglikelihood = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        return i + 1, state, op_key, curr_loglikelihood + info.log_likelihood_increment

    total_iter, final_state, _, log_likelihood = jax.lax.while_loop(
        cond, body, (0, initial_state, rng_key, 0.0)
    )

    return total_iter, final_state, log_likelihood


class TemperedSMCTest(chex.TestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def logprob_fn(self, scale, coefs, preds, x):
        """Linear regression"""
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    @chex.all_variants(without_jit=False, with_pmap=False)
    @parameterized.parameters(itertools.product([100, 5000], [True, False]))
    def test_adaptive_tempered_smc(self, N, use_log):
        x_data = np.random.normal(0, 1, size=(1000, 1))
        y_data = 3 * x_data + np.random.normal(size=x_data.shape)
        observations = {"x": x_data, "preds": y_data}

        conditioned_logprob = lambda x: self.logprob_fn(*x, **observations)

        prior = lambda x: stats.expon.logpdf(x[0], 1, 1) + stats.norm.logpdf(x[1])
        scale_init = 1 + np.random.exponential(1, N)
        coeffs_init = 3 + 2 * np.random.randn(N)
        smc_state_init = [scale_init, coeffs_init]

        iterates = []
        results = []  # type: List[TemperedSMCState]
        mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 10e-2, jnp.eye(2), 50)

        for target_ess in [0.5, 0.75]:
            tempering_kernel = adaptive_tempered_smc(
                prior,
                conditioned_logprob,
                mcmc_kernel_factory,
                hmc.new_state,
                resampling.systematic,
                target_ess,
                solver.dichotomy,
                use_log,
                5,
            )
            tempered_smc_state_init = TemperedSMCState(smc_state_init, 0.0)

            n_iter, result, log_likelihood = self.variant(
                functools.partial(inference_loop, tempering_kernel)
            )(self.key, tempered_smc_state_init)
            iterates.append(n_iter)
            results.append(result)

            np.testing.assert_allclose(np.mean(result.particles[0]), 1.0, rtol=1e-1)
            np.testing.assert_allclose(np.mean(result.particles[1]), 3.0, rtol=1e-1)

        assert iterates[1] >= iterates[0]

    @chex.all_variants(without_jit=False, with_pmap=False)
    @parameterized.parameters(itertools.product([100, 1000], [10, 100]))
    def test_fixed_schedule_tempered_smc(self, N, n_schedule):
        x_data = np.random.normal(0, 1, size=(1000, 1))
        y_data = 3 * x_data + np.random.normal(size=x_data.shape)
        observations = {"x": x_data, "preds": y_data}

        conditionned_logprob = lambda x: self.logprob_fn(*x, **observations)
        prior = lambda x: stats.norm.logpdf(jnp.log(x[0])) + stats.norm.logpdf(x[1])
        scale_init = np.exp(np.random.randn(N))
        coeffs_init = np.random.randn(N)
        smc_state_init = [scale_init, coeffs_init]

        lambda_schedule = np.logspace(-5, 0, n_schedule)
        mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 10e-2, jnp.eye(2), 50)

        tempering_kernel = self.variant(
            tempered_smc(
                prior,
                conditionned_logprob,
                mcmc_kernel_factory,
                hmc.new_state,
                resampling.systematic,
                10,
            )
        )
        tempered_smc_state_init = TemperedSMCState(smc_state_init, 0.0)

        def body_fn(carry, lmbda):
            rng_key, state = carry
            _, rng_key = jax.random.split(rng_key)
            new_state, info = tempering_kernel(rng_key, state, lmbda)
            return (rng_key, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(
            body_fn, (self.key, tempered_smc_state_init), lambda_schedule
        )
        np.testing.assert_allclose(np.mean(result.particles[0]), 1.0, rtol=1e-1)
        np.testing.assert_allclose(np.mean(result.particles[1]), 3.0, rtol=1e-1)


def normal_logprob_fn(x, chol_cov):
    """minus log-density of a centered multivariate normal distribution"""
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        np.sum(np.log(np.abs(np.diag(chol_cov)))) + dim * np.log(2 * np.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -(0.5 * norm_y + normalizing_constant)


class NormalizingConstantTest(chex.TestCase):
    """Test normalizing constant estimate."""

    @chex.all_variants(without_jit=False, with_pmap=False)
    @parameterized.parameters(itertools.product([500, 1_000], [2, 10]))
    def test_normalizing_constant(self, N, dim):
        rng_key = jax.random.PRNGKey(2356)
        rng_key, cov_key = jax.random.split(rng_key, 2)
        chol_cov = jax.random.uniform(cov_key, shape=(dim, dim))
        iu = np.triu_indices(dim, 1)
        chol_cov = chol_cov.at[iu].set(0.0)
        cov = chol_cov @ chol_cov.T
        conditionned_logprob = lambda x: normal_logprob_fn(x, chol_cov)

        prior = lambda x: stats.multivariate_normal.logpdf(
            x, jnp.zeros((dim,)), jnp.eye(dim)
        )

        rng_key, init_key = jax.random.split(rng_key, 2)
        x_init = jax.random.normal(init_key, shape=(N, dim))

        mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 1e-2, jnp.eye(dim), 50)

        tempering_kernel = adaptive_tempered_smc(
            prior,
            conditionned_logprob,
            mcmc_kernel_factory,
            hmc.new_state,
            resampling.systematic,
            0.9,
            solver.dichotomy,
            True,
            10,
        )
        tempered_smc_state_init = TemperedSMCState(x_init, 0.0)
        n_iter, result, log_likelihood = self.variant(
            functools.partial(inference_loop, tempering_kernel)
        )(rng_key, tempered_smc_state_init)

        expected_log_likelihood = -0.5 * np.linalg.slogdet(np.eye(dim) + cov)[
            1
        ] - dim / 2 * np.log(2 * np.pi)

        np.testing.assert_allclose(log_likelihood, expected_log_likelihood, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
