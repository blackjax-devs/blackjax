import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import optax
from absl.testing import absltest

import blackjax
from blackjax.vi._gaussian_vi import _objective_value_from_log_ratio
from blackjax.vi.meanfield_vi import (
    KL,
    MFVIState,
    RenyiAlpha,
    generate_meanfield_logdensity,
    init,
    sample,
    step,
)
from tests.fixtures import BlackJAXTest, std_normal_logdensity


class MFVIUnitTest(BlackJAXTest):
    """Unit tests for the meanfield VI building blocks."""

    def setUp(self):
        super().setUp()
        self.optimizer = optax.adam(1e-2)

    def test_init_zeros_mean(self):
        """init sets mu to zero for all leaves."""
        position = jnp.ones(4)
        state = init(position, self.optimizer)
        np.testing.assert_array_equal(state.mu, jnp.zeros(4))

    def test_init_rho_minus_two(self):
        """init sets rho to -2 for all leaves."""
        position = jnp.ones(4)
        state = init(position, self.optimizer)
        np.testing.assert_array_equal(state.rho, -2.0 * jnp.ones(4))

    def test_init_pytree_position(self):
        """init works with PyTree positions."""
        position = {"a": jnp.zeros(3), "b": jnp.ones(2)}
        state = init(position, self.optimizer)
        chex.assert_trees_all_equal_shapes(state.mu, position)
        chex.assert_trees_all_equal_shapes(state.rho, position)

    def test_step_returns_state_and_info(self):
        """step returns (MFVIState, MFVIInfo) with finite ELBO."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = step(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        self.assertIsInstance(new_state, MFVIState)
        assert jnp.isfinite(info.elbo)

    def test_step_output_shapes_unchanged(self):
        """step output mu/rho have same shape as before."""
        position = {"x": jnp.zeros(3), "y": jnp.zeros(2)}
        state = init(position, self.optimizer)

        new_state, _ = step(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        chex.assert_trees_all_equal_shapes(state.mu, new_state.mu)
        chex.assert_trees_all_equal_shapes(state.rho, new_state.rho)

    def test_elbo_decreases_over_steps(self):
        """ELBO should improve (decrease KL) after several optimization steps."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        def logdensity_fn(x):
            return -0.5 * jnp.sum((x - 3.0) ** 2)

        initial_elbo = None
        for i in range(50):
            subkey = jax.random.fold_in(self.next_key(), i)
            state, info = jax.jit(step, static_argnums=(2, 3))(
                subkey, state, logdensity_fn, self.optimizer
            )
            if initial_elbo is None:
                initial_elbo = float(info.elbo)

        # KL divergence (= -ELBO here) should decrease; ELBO should increase
        assert float(info.elbo) < initial_elbo  # elbo = KL, lower is better

    def test_sample_shape(self):
        """sample returns (num_samples, ...) shaped output."""
        position = jnp.zeros(3)
        state = init(position, self.optimizer)
        samples = sample(self.next_key(), state, num_samples=20)
        self.assertEqual(samples.shape, (20, 3))

    def test_sample_pytree_shape(self):
        """sample works with PyTree positions."""
        position = {"a": jnp.zeros(2), "b": jnp.zeros(4)}
        state = init(position, self.optimizer)
        samples = sample(self.next_key(), state, num_samples=10)
        self.assertEqual(samples["a"].shape, (10, 2))
        self.assertEqual(samples["b"].shape, (10, 4))

    def test_generate_meanfield_logdensity(self):
        """generate_meanfield_logdensity returns a finite scalar."""
        position = jnp.zeros(3)
        state = init(position, self.optimizer)
        logdensity = generate_meanfield_logdensity(state.mu, state.rho)
        val = logdensity(jnp.ones(3))
        self.assertEqual(val.shape, ())
        assert jnp.isfinite(val)

    def test_jit_compatible(self):
        """step is JIT-compilable."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = jax.jit(step, static_argnums=(2, 3))(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        assert jnp.isfinite(info.elbo)

    def test_step_with_kl_objective(self):
        """MFVI step works explicitly with KL()."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = step(
            self.next_key(),
            state,
            std_normal_logdensity,
            self.optimizer,
            objective=KL(),
        )

        self.assertIsInstance(new_state, MFVIState)
        assert jnp.isfinite(info.elbo)

    def test_step_with_renyi_objective(self):
        """MFVI step works with RenyiAlpha(alpha=n) when STL is False."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = step(
            self.next_key(),
            state,
            std_normal_logdensity,
            self.optimizer,
            objective=RenyiAlpha(alpha=0.5),
            stl_estimator=False,
        )

        self.assertIsInstance(new_state, MFVIState)
        assert jnp.isfinite(info.elbo)

    def test_renyi_with_stl_raises(self):
        """MFVI should raise error STL for RenyiAlpha(alpha != 1)."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        with self.assertRaises(ValueError):
            step(
                self.next_key(),
                state,
                std_normal_logdensity,
                self.optimizer,
                objective=RenyiAlpha(alpha=0.5),
                stl_estimator=True,
            )

    def test_objective_value_renyi_alpha_one_matches_kl(self):
        """RenyiAlpha(alpha=1.0) should recover the KL objective."""
        log_ratio = jnp.array([1.0, 2.0, 3.0])

        value = _objective_value_from_log_ratio(log_ratio, RenyiAlpha(alpha=1.0))

        self.assertAlmostEqual(float(value), 2.0)

    def test_objective_value_invalid_objective_raises(self):
        """Unsupported objective types should raise TypeError."""
        log_ratio = jnp.array([1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            _objective_value_from_log_ratio(log_ratio, object())


class MFVITest(BlackJAXTest):
    def test_recover_posterior(self):
        ground_truth = [
            # loc, scale
            (2, 4),
            (3, 5),
        ]

        def logdensity_fn(x):
            logpdf = stats.norm.logpdf(x["x_1"], *ground_truth[0]) + stats.norm.logpdf(
                x["x_2"], *ground_truth[1]
            )
            return jnp.sum(logpdf)

        initial_position = {"x_1": 0.0, "x_2": 0.0}

        num_steps = 15_000
        num_samples = 100

        optimizer = optax.sgd(1e-2)
        mfvi = blackjax.meanfield_vi(logdensity_fn, optimizer, num_samples)
        state = mfvi.init(initial_position)

        rng_key = self.next_key()
        for i in range(num_steps):
            subkey = jax.random.fold_in(rng_key, i)
            state, _ = jax.jit(mfvi.step)(subkey, state)

        loc_1, loc_2 = state.mu["x_1"], state.mu["x_2"]
        scale = jax.tree.map(jnp.exp, state.rho)
        scale_1, scale_2 = scale["x_1"], scale["x_2"]
        self.assertAlmostEqual(loc_1, ground_truth[0][0], delta=0.01)
        self.assertAlmostEqual(scale_1, ground_truth[0][1], delta=0.01)
        self.assertAlmostEqual(loc_2, ground_truth[1][0], delta=0.01)
        self.assertAlmostEqual(scale_2, ground_truth[1][1], delta=0.01)

    def test_top_level_api_with_renyi(self):
        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        optimizer = optax.adam(1e-2)
        algo = blackjax.meanfield_vi(
            logdensity_fn,
            optimizer,
            20,
            objective=RenyiAlpha(alpha=0.5),
            stl_estimator=False,
        )

        state = algo.init(jnp.zeros(2))
        state, info = algo.step(self.next_key(), state)
        assert jnp.isfinite(info.elbo)


if __name__ == "__main__":
    absltest.main()
