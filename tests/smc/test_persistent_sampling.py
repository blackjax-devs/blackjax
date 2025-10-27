"""Test the Persistent Sampling steps and routine"""

from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized
from jax.scipy.special import logsumexp

import blackjax
import blackjax.smc.resampling as resampling
from blackjax import adaptive_persistent_sampling_smc, persistent_sampling_smc
from blackjax.smc import extend_params
from blackjax.smc.persistent_sampling import (
    PersistentSMCState,
    PersistentStateInfo,
    compute_log_persistent_weights,
    compute_log_Z,
    compute_persistent_ess,
    init,
    remove_padding,
    resample_from_persistent,
)
from blackjax.types import ArrayLikeTree, PRNGKey
from tests.smc import SMCLinearRegressionTestCase

########################################################################################
# Unit Tests
########################################################################################


class PersistentSamplingUnitTest(chex.TestCase):
    """Unit tests for core persistent sampling functions."""

    def setUp(self) -> None:
        """Set up test case with random key."""
        super().setUp()
        self.key = jax.random.key(10)

    @parameterized.named_parameters(
        ("1d_array", lambda key: jax.random.normal(key, (10,))),
        ("2d_array", lambda key: jax.random.normal(key, (10, 3))),
        ("3d_array", lambda key: jax.random.normal(key, (10, 3, 4))),
        (
            "dict_scalar",
            lambda key: {
                "a": jax.random.normal(key, (10,)),
                "b": jax.random.normal(jax.random.fold_in(key, 1), (10,)),
            },
        ),
        (
            "dict_1d",
            lambda key: {
                "x": jax.random.normal(key, (10, 2)),
                "y": jax.random.normal(jax.random.fold_in(key, 1), (10, 3)),
            },
        ),
        (
            "dict_2d",
            lambda key: {
                "pos": jax.random.normal(key, (10, 3, 3)),
                "vel": jax.random.normal(jax.random.fold_in(key, 1), (10, 3, 3)),
            },
        ),
        (
            "nested_dict",
            lambda key: {
                "outer": {
                    "inner1": jax.random.normal(key, (10, 2)),
                    "inner2": jax.random.normal(jax.random.fold_in(key, 1), (10,)),
                },
                "scalar": jax.random.normal(jax.random.fold_in(key, 2), (10,)),
            },
        ),
        (
            "mixed_dict",
            lambda key: {
                "scalar": jax.random.normal(key, (10,)),
                "vec": jax.random.normal(jax.random.fold_in(key, 1), (10, 4)),
                "mat": jax.random.normal(jax.random.fold_in(key, 2), (10, 2, 2)),
            },
        ),
    )
    def test_init(self, particle_generator: Callable) -> None:
        """Test that init properly sets up the PersistentSMCState with different pytree
        shapes."""
        num_particles = 10
        n_schedule = 5

        # Generate particles
        rng_key, init_key = jax.random.split(self.key, 2)
        particles = particle_generator(init_key)

        def loglikelihood_fn(x: ArrayLikeTree) -> jnp.ndarray:
            leaves = jax.tree.leaves(x)
            return jnp.array(sum(jnp.sum(leaf) for leaf in leaves))

        state = init(particles, loglikelihood_fn, n_schedule)

        # Check state properties
        assert isinstance(state, PersistentSMCState)
        assert state.iteration == 0
        assert state.tempering_schedule[0] == 0.0
        assert state.persistent_log_Z[0] == 0.0

        # Check shapes
        assert state.persistent_log_likelihoods.shape == (n_schedule + 1, num_particles)
        assert state.tempering_schedule.shape == (n_schedule + 1,)
        assert state.persistent_log_Z.shape == (n_schedule + 1,)

        # Check that log-likelihoods are computed
        expected_log_likelihoods = jax.vmap(loglikelihood_fn)(particles)
        np.testing.assert_allclose(
            state.persistent_log_likelihoods[0],
            expected_log_likelihoods,
            rtol=1e-5,
        )

        # Verify particles structure matches
        state_leaves = jax.tree.leaves(state.particles)
        input_leaves = jax.tree.leaves(particles)
        assert len(state_leaves) == len(input_leaves)

        for state_leaf, input_leaf in zip(state_leaves, input_leaves):
            np.testing.assert_allclose(state_leaf, input_leaf, rtol=1e-5)

    def test_remove_padding(self) -> None:
        """Test that remove_padding correctly removes zero-padded arrays."""
        num_particles = 10
        num_dim = 2
        n_schedule = 5

        key, init_key = jax.random.split(self.key, 2)
        particles = jax.random.normal(init_key, shape=(num_particles, num_dim))

        def loglikelihood_fn(x: jnp.ndarray) -> jnp.ndarray:
            return stats.norm.logpdf(x).sum()

        state = init(particles, loglikelihood_fn, n_schedule)

        # Simulate advancing to iteration 2
        state = state._replace(iteration=2)

        # Remove padding
        trimmed_state = remove_padding(state)

        # Check that the leaves of the persistent particles are trimmed correctly, i.e.
        # the first dimension matches iteration + 1 for persistent_particles
        trimmed_leaves = jax.tree.leaves(trimmed_state.persistent_particles)
        original_leaves = jax.tree.leaves(state.persistent_particles)
        for trimmed_leaf, original_leaf in zip(trimmed_leaves, original_leaves):
            assert trimmed_leaf.shape[0] == 3  # iteration + 1
            assert trimmed_leaf.shape[1:] == original_leaf.shape[1:]

        # Check that remaining arrays are trimmed to iteration + 1
        assert trimmed_state.persistent_log_likelihoods.shape[0] == 3
        assert trimmed_state.tempering_schedule.shape[0] == 3
        assert trimmed_state.persistent_log_Z.shape[0] == 3
        assert trimmed_state.iteration == 2

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_compute_log_Z(self) -> None:
        """Test log normalizing constant computation."""
        num_particles = 200
        iteration = 3

        # Create weights drawn from a uniform distribution
        key, sample_key = jax.random.split(self.key)
        samples = jax.random.uniform(
            sample_key,
            shape=(iteration, num_particles),
        )
        log_weights = stats.uniform.logpdf(samples)

        # Compute log_Z (with and without JIT)
        compute_log_Z_fn = self.variant(compute_log_Z)
        log_Z = compute_log_Z_fn(log_weights, iteration)

        # For uniform weights, log_Z should be approximately 0
        np.testing.assert_allclose(log_Z, 0.0, atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_compute_log_persistent_weights(self) -> None:
        """Test compute_log_persistent_weights."""
        num_particles = 20
        num_iterations = 3

        key, sample_key = jax.random.split(self.key)
        persistent_log_likelihoods = jax.random.normal(
            sample_key, shape=(num_iterations + 2, num_particles)
        )
        persistent_log_Z = jnp.zeros(num_iterations + 2)
        tempering_schedule = jnp.array([0.0, 0.3, 0.6, 1.0, 0.0])
        iteration = 2

        compute_log_persistent_weights_fn = self.variant(compute_log_persistent_weights)

        log_weights, log_Z = compute_log_persistent_weights_fn(
            persistent_log_likelihoods,
            persistent_log_Z,
            tempering_schedule,
            iteration,
        )

        # Check shapes
        assert log_weights.shape == (num_iterations + 2, num_particles)
        assert isinstance(log_Z, jnp.ndarray) or isinstance(log_Z, float)

        # Check that weights are finite where they should be
        assert jnp.all(jnp.isfinite(log_weights[: iteration + 1]))

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_persistent_ess_uniform_weights(self) -> None:
        """Test ESS calculation with uniform weights."""
        num_particles = 200
        num_iterations = 3

        # Uniform weights (in log space, normalized)
        log_weights = jnp.ones((num_iterations, num_particles))
        norm_log_weights = log_weights - logsumexp(log_weights)

        # Compute ESS with normalized weights and unnormalized weights, testing if
        # normalization inside the function works correctly (with and without JIT)
        compute_persistent_ess_fn = self.variant(compute_persistent_ess)
        compute_persistent_ess_with_norm_fn = self.variant(
            partial(compute_persistent_ess, normalize_weights=True)
        )
        ess = compute_persistent_ess_fn(norm_log_weights)
        ess_with_norm = compute_persistent_ess_with_norm_fn(log_weights)

        # For uniform weights, ESS should be close to the total number of particles
        expected_ess = num_iterations * num_particles
        np.testing.assert_allclose(ess, expected_ess, rtol=1e-5)
        np.testing.assert_allclose(ess_with_norm, expected_ess, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_persistent_ess_single_weight(self) -> None:
        """Test ESS calculation when one weight is 1 and rest are 0."""
        num_particles = 10
        num_iterations = 3

        # Create weights where one is 1 and rest are 0
        log_weights = jnp.full((num_iterations, num_particles), -jnp.inf)
        log_weights = log_weights.at[0, 0].set(0.0)  # log(1) = 0

        # Compute ESS with and without normalization, although normalization param
        # should not matter in this case, since only one weight is non-zero
        compute_persistent_ess_fn = self.variant(compute_persistent_ess)
        compute_persistent_ess_with_norm_fn = self.variant(
            partial(compute_persistent_ess, normalize_weights=True)
        )
        ess = compute_persistent_ess_fn(log_weights)
        ess_with_norm = compute_persistent_ess_with_norm_fn(log_weights)

        # ESS should be close to 1
        np.testing.assert_allclose(ess, 1.0, rtol=1e-5)
        np.testing.assert_allclose(ess_with_norm, 1.0, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_resampling_from_persistent(self) -> None:
        """Test resampling N from multiple iterations of particles."""
        num_particles = 10
        num_dim = 2
        num_iterations = 3

        rng_key, init_key = jax.random.split(self.key, 2)

        # Create persistent particles from multiple iterations
        persistent_particles = jax.random.normal(
            init_key,
            shape=(num_iterations, num_particles, num_dim),
        )

        # Create uniform weights
        persistent_weights = (
            jnp.ones((num_iterations, num_particles)) / num_iterations / num_particles
        )

        # Resample (with and without JIT)
        resample_from_persistent_fn = self.variant(
            partial(resample_from_persistent, resample_fn=resampling.systematic)
        )
        resampled_particles, resample_idx = resample_from_persistent_fn(
            rng_key,
            persistent_particles,
            persistent_weights,
        )

        # Check output shape
        assert resampled_particles.shape == (num_particles, num_dim)
        assert resample_idx.shape == (num_particles,)

        # Check that indices are valid
        assert jnp.all(resample_idx >= 0)
        assert jnp.all(resample_idx < num_iterations * num_particles)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    @parameterized.named_parameters(
        (
            "dict_with_1d_2d",
            lambda key: {
                "pos": jax.random.normal(key, (8, 3)),
                "vel": jax.random.normal(jax.random.fold_in(key, 1), (8, 3, 2)),
            },
        ),
        (
            "deeply_nested",
            lambda key: {
                "level1": {
                    "level2": {
                        "level3": jax.random.normal(key, (8, 2)),
                    },
                    "level2_flat": jax.random.normal(jax.random.fold_in(key, 1), (8,)),
                },
            },
        ),
    )
    def test_resample_from_persistent_with_pytrees(
        self,
        particle_generator: Callable,
    ) -> None:
        """Test resample_from_persistent with complex pytree structures."""
        num_particles = 8
        num_iterations = 3

        rng_key, init_key = jax.random.split(self.key, 2)

        # Generate particles for one iteration, then expand to multiple iterations
        single_iter_particles = particle_generator(init_key)

        # Create persistent particles by stacking multiple iterations
        def expand_to_iterations(leaf: jnp.ndarray) -> jnp.ndarray:
            # Expand first particle dimension to (num_iterations, num_particles, ...)
            return jnp.tile(
                leaf[None, ...], (num_iterations, 1) + (1,) * (leaf.ndim - 1)
            )

        persistent_particles = jax.tree.map(expand_to_iterations, single_iter_particles)

        persistent_weights = (
            jnp.ones((num_iterations, num_particles)) / num_iterations / num_particles
        )

        resample_from_persistent_fn = self.variant(
            partial(resample_from_persistent, resample_fn=resampling.systematic)
        )

        rng_key, resample_key = jax.random.split(rng_key)
        resampled_particles, resample_idx = resample_from_persistent_fn(
            resample_key,
            persistent_particles,
            persistent_weights,
        )

        # Check that structure is preserved
        orig_leaves = jax.tree.leaves(single_iter_particles)
        resampled_leaves = jax.tree.leaves(resampled_particles)
        assert len(orig_leaves) == len(resampled_leaves)

        # Check shapes (should match original except batch dimension)
        for orig_leaf, resampled_leaf in zip(orig_leaves, resampled_leaves):
            assert resampled_leaf.shape == orig_leaf.shape


########################################################################################
# State Update Tests
########################################################################################


class PersistentSamplingStateUpdateTest(chex.TestCase):
    """Tests that verify state updates properly at each iteration."""

    def setUp(self) -> None:
        super().setUp()
        self.key = jax.random.key(4242)

    def test_state_updates_each_iteration(self) -> None:
        """Test inference loop that validates state updates."""
        num_particles = 50
        num_dim = 2
        n_schedule = 10

        rng_key, init_key = jax.random.split(self.key, 2)
        particles = jax.random.normal(init_key, shape=(num_particles, num_dim))

        def logprior_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                stats.multivariate_normal.logpdf(
                    x, jnp.zeros((num_dim,)), jnp.eye(num_dim)
                )
            )

        def loglikelihood_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                stats.multivariate_normal.logpdf(
                    x, jnp.zeros((num_dim,)), 0.5 * jnp.eye(num_dim)
                )
            )

        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(num_dim),
                "num_integration_steps": 30,
            },
        )

        ps = persistent_sampling_smc(
            logprior_fn,
            loglikelihood_fn,
            n_schedule=n_schedule,
            mcmc_step_fn=hmc_kernel,
            mcmc_init_fn=hmc_init,
            mcmc_parameters=hmc_parameters,
            resampling_fn=resampling.systematic,
            num_mcmc_steps=5,
        )

        state = ps.init(particles)  # type: ignore

        # Verify initial state
        assert state.iteration == 0, "Initial iteration should be 0"
        assert state.tempering_schedule[0] == 0.0, "Initial lambda should be 0.0"
        assert state.persistent_log_Z[0] == 0.0, "Initial log_Z should be 0.0"
        initial_log_liks = state.persistent_log_likelihoods[0]
        assert jnp.all(
            jnp.isfinite(initial_log_liks)
        ), "Initial log-likelihoods should be finite"

        # Run multiple steps with different lambda values
        lambda_schedule = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        prev_log_Z = state.log_Z

        for step_idx, lmbda in enumerate(lambda_schedule):
            rng_key, subkey = jax.random.split(rng_key)
            prev_state = state
            state, info = ps.step(subkey, state, lmbda)  # type: ignore

            expected_iteration = step_idx + 1

            # 1. Check that iteration is incremented
            assert (
                state.iteration == expected_iteration
            ), f"Iteration should be {expected_iteration}, got {state.iteration}"

            # 2. Check that the ensemble grows - verify persistent particles are
            # populated at the new iteration index
            state_particles = jax.tree.leaves(state.persistent_particles)[0]
            assert (
                jnp.count_nonzero(state_particles[expected_iteration]) > 0
            ), f"Particles at iteration {expected_iteration} should be non-zero"

            # Verify particles from previous iterations are preserved
            for prev_iter in range(expected_iteration):
                prev_particles = state_particles[prev_iter]
                assert (
                    jnp.count_nonzero(prev_particles) > 0
                ), f"Particles from iteration {prev_iter} should still be present"

            # 3. Check that lambda is set correctly in the schedule
            np.testing.assert_allclose(
                state.tempering_param,
                lmbda,
                rtol=1e-5,
                err_msg=f"Lambda at iteration {expected_iteration} should be {lmbda}",
            )
            np.testing.assert_allclose(
                state.tempering_schedule[expected_iteration],
                lmbda,
                rtol=1e-5,
                err_msg=f"Tempering schedule at index {expected_iteration} "
                f"should be {lmbda}",
            )

            # 4. Check that weights are calculated appropriately
            weights = state.persistent_weights

            # Verify weights shape: should have entries up to current iteration
            assert weights.shape == (
                n_schedule + 1,
                num_particles,
            ), f"Weights shape should be ({n_schedule + 1}, {num_particles})"

            # Verify weights are finite for all iterations up to current
            active_weights = weights[: expected_iteration + 1]
            assert jnp.all(
                jnp.isfinite(active_weights)
            ), f"Weights up to iteration {expected_iteration} should be finite"

            # Verify weights are non-negative
            assert jnp.all(active_weights >= 0), "Weights should be non-negative"

            # Verify weights sum to iteration * num_particles (as per paper)
            expected_sum = (expected_iteration + 1) * num_particles
            actual_sum = jnp.sum(active_weights)
            np.testing.assert_allclose(
                actual_sum,
                expected_sum,
                rtol=1e-4,
                err_msg=f"Weights should sum to {expected_sum} at "
                f"iteration {expected_iteration}",
            )

            # 5. Check that ESS is calculated sensibly
            log_weights = jnp.log(weights[: expected_iteration + 1])
            ess = compute_persistent_ess(log_weights, normalize_weights=True)

            # ESS should be positive and finite
            assert jnp.isfinite(
                ess
            ), f"ESS should be finite at iteration {expected_iteration}"
            assert ess > 0, f"ESS should be positive at iteration {expected_iteration}"

            # ESS should not exceed the total number of persistent particles
            max_ess = (expected_iteration + 1) * num_particles
            assert ess <= max_ess + 1e-5, (
                f"ESS {ess} should not exceed {max_ess} at "
                f"iteration {expected_iteration}"
            )

            # 6. Check that log-likelihoods are updated correctly
            current_log_liks = state.persistent_log_likelihoods[expected_iteration]

            # Log-likelihoods should be finite
            assert jnp.all(
                jnp.isfinite(current_log_liks)
            ), f"Log-likelihoods at iteration {expected_iteration} should be finite"

            # Verify log-likelihoods are computed for current particles
            current_particles = state.particles
            expected_log_liks = jax.vmap(loglikelihood_fn)(current_particles)
            np.testing.assert_allclose(
                current_log_liks,
                expected_log_liks,
                rtol=1e-5,
                err_msg=f"Log-likelihoods should match computed values at "
                f"iteration {expected_iteration}",
            )

            # Verify log-likelihoods from previous iterations are preserved
            for prev_iter in range(expected_iteration):
                prev_log_liks = state.persistent_log_likelihoods[prev_iter]
                prev_state_log_liks = prev_state.persistent_log_likelihoods[prev_iter]
                np.testing.assert_allclose(
                    prev_log_liks,
                    prev_state_log_liks,
                    rtol=1e-5,
                    err_msg=f"Log-likelihoods from iteration {prev_iter} should be "
                    "preserved",
                )

            # 7. Check that log_Z is updated and makes sense
            current_log_Z = state.log_Z

            # log_Z should be finite
            assert jnp.isfinite(
                current_log_Z
            ), f"log_Z should be finite at iteration {expected_iteration}"

            # For this problem with Gaussian prior/likelihood, log_Z should generally
            # increase (or stay similar) as we incorporate more data
            # NOTE: This is not a strict requirement, but helps catch obvious bugs
            if lmbda > 0:
                # Just check it's not wildly different (within a reasonable range)
                assert (
                    jnp.abs(current_log_Z - prev_log_Z) < 100
                ), f"log_Z change seems unreasonable: {prev_log_Z} -> {current_log_Z}"

            prev_log_Z = current_log_Z

            # Verify log_Z history is updated correctly
            np.testing.assert_allclose(
                state.persistent_log_Z[expected_iteration],
                current_log_Z,
                rtol=1e-5,
                err_msg=f"persistent_log_Z at index {expected_iteration} should match "
                "log_Z",
            )

            # 8. Additional checks: verify padding is still in place for future
            # iterations
            if expected_iteration < n_schedule:
                future_particles = state_particles[expected_iteration + 1 :]
                # Future particles should be zeros (padding)
                assert jnp.allclose(
                    future_particles, 0.0
                ), "Future iterations should still be zero-padded"

                future_log_liks = state.persistent_log_likelihoods[
                    expected_iteration + 1 :
                ]
                assert jnp.allclose(
                    future_log_liks, 0.0
                ), "Future log-likelihoods should still be zero-padded"


def inference_loop_adaptive(
    rng_key: PRNGKey,
    kernel: Callable,
    initial_state: PersistentSMCState,
    target_ess: float,
    max_iterations: int,
) -> PersistentSMCState:
    """Run adaptive SMC until condition is met."""

    def cond(carry: tuple[PersistentSMCState, PRNGKey]) -> jnp.ndarray:
        """Returns True while lambda < 1.0 or ESS < target_ess and
        iteration < max_iterations."""
        state, _ = carry
        ess = blackjax.persistent_sampling.compute_persistent_ess(
            jnp.log(state.persistent_weights),
            normalize_weights=True,
        )
        return jnp.logical_and(
            jnp.logical_or(
                state.tempering_param < 1.0, ess < target_ess * state.num_particles
            ),
            state.iteration < max_iterations,
        )

    def one_step(
        carry: tuple[PersistentSMCState, PRNGKey],
    ) -> tuple[PersistentSMCState, PRNGKey]:
        state, key = carry
        key, subkey = jax.random.split(key)
        state, _ = kernel(subkey, state)
        return state, key

    final_state, _ = jax.lax.while_loop(
        cond,
        one_step,
        (initial_state, rng_key),
    )

    return final_state


########################################################################################
# Posterior Estimation Tests
########################################################################################


def inference_loop_fixed(
    rng_key: PRNGKey,
    kernel: Callable,
    initial_state: PersistentSMCState,
    tempering_schedule: jnp.ndarray,
) -> PersistentSMCState:
    """Inference loop for fixed schedule persistent sampling."""

    def body_fn(
        carry: tuple[int, PersistentSMCState],
        lmbda: float,
    ) -> tuple[
        tuple[int, PersistentSMCState], tuple[PersistentSMCState, PersistentStateInfo]
    ]:
        i, state = carry
        subkey = jax.random.fold_in(rng_key, i)
        new_state, info = kernel(subkey, state, lmbda)
        return (i + 1, new_state), (new_state, info)

    (_, result), _ = jax.lax.scan(
        body_fn,  # type: ignore
        (0, initial_state),
        tempering_schedule,  # type: ignore
    )
    return result


class PersistentSamplingPosteriorTest(SMCLinearRegressionTestCase):
    """Integration tests for persistent sampling on regression problems,
    adapted from blackjax.tests.smc.tempered.TemperedSMCLinearRegressionTestCase."""

    def setUp(self) -> None:
        super().setUp()
        self.key = jax.random.key(10)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_fixed_schedule_persistent_sampling(self) -> None:
        """Test persistent sampling with fixed tempering schedule."""
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        num_tempering_steps = 10

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)
        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            },
        )

        ps = persistent_sampling_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_schedule=num_tempering_steps,
            mcmc_step_fn=hmc_kernel,
            mcmc_init_fn=hmc_init,
            mcmc_parameters=hmc_parameters,
            resampling_fn=resampling.systematic,
            num_mcmc_steps=10,
        )
        init_state = ps.init(init_particles)  # type: ignore

        key, sample_key = jax.random.split(self.key)
        result = self.variant(
            partial(
                inference_loop_fixed,
                kernel=ps.step,
            )
        )(
            rng_key=sample_key,
            initial_state=init_state,
            tempering_schedule=lambda_schedule,
        )

        # Check that we get reasonable posterior estimates
        self.assert_linear_regression_test_case(result)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_adaptive_persistent_sampling(self) -> None:
        """Test persistent sampling with adaptive schedule."""
        num_particles = 100
        max_iterations = 100

        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        iterates = []
        results = []

        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_init = blackjax.hmc.init

        base_params = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            }
        )

        # verify results are equivalent with all shared, all unshared, and mixed params
        hmc_parameters_list = [
            base_params,
            jax.tree.map(lambda x: jnp.repeat(x, num_particles, axis=0), base_params),
            jax.tree_util.tree_map_with_path(
                lambda path, x: (
                    jnp.repeat(x, num_particles, axis=0)
                    if path[0].key == "step_size"
                    else x
                ),
                base_params,
            ),
        ]

        for target_ess, hmc_parameters in zip([1, 3, 5], hmc_parameters_list):
            ps = adaptive_persistent_sampling_smc(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                max_iterations=max_iterations,
                mcmc_step_fn=hmc_kernel,
                mcmc_init_fn=hmc_init,
                mcmc_parameters=hmc_parameters,
                resampling_fn=resampling.systematic,
                target_ess=target_ess,
                num_mcmc_steps=5,
            )
            init_state = ps.init(init_particles)  # type: ignore

            loop_fn = self.variant(
                partial(
                    inference_loop_adaptive,
                    kernel=ps.step,
                    target_ess=target_ess,
                    max_iterations=max_iterations,
                )
            )

            key, sample_key = jax.random.split(self.key)
            result = self.variant(loop_fn)(rng_key=sample_key, initial_state=init_state)
            iterates.append(result.iteration)
            results.append(result)

            # Check that iterations do not exceed max allowed
            # NOTE: This is not enforced in the algorithm, and needs to be
            # checked post-hoc, e.g. in the inference loop. Its checked here
            # to make sure that the final distribution gets sampled.
            assert result.iteration < max_iterations

            # Check that we get reasonable posterior estimates
            self.assert_linear_regression_test_case(result)

        # Check that higher ESS leads to more iterations
        assert iterates[1] >= iterates[0]
        assert iterates[2] >= iterates[1]


########################################################################################
# Normalizing Constant / Marginal Likelihood Tests
########################################################################################


def multivariate_normal_log_pdf(
    x: jnp.ndarray,
    chol_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log density of multivariate normal with zero mean and covariance
    defined by its Cholesky factor."""
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        np.sum(np.log(np.abs(np.diag(chol_cov)))) + dim * np.log(2 * np.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -(0.5 * norm_y + normalizing_constant)


class NormalizingConstantTest(chex.TestCase):
    """Test normalizing constant estimate for persistent sampling,
    adapted from blackjax.tests.smc.tempered.NormalizingConstantTest."""

    def setUp(self) -> None:
        super().setUp()
        self.key = jax.random.key(2356)

    def _setup_test_problem(
        self, num_dim: int
    ) -> tuple[PRNGKey, jnp.ndarray, Callable, Callable,]:
        """Setup common test problem: random covariance and log functions."""
        rng_key, cov_key = jax.random.split(self.key, 2)
        chol_cov = jax.random.uniform(cov_key, shape=(num_dim, num_dim))
        iu = np.triu_indices(num_dim, 1)
        chol_cov = chol_cov.at[iu].set(0.0)
        cov = chol_cov @ chol_cov.T

        def logprior_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                stats.multivariate_normal.logpdf(
                    x, jnp.zeros((num_dim,)), jnp.eye(num_dim)
                )
            )

        loglikelihood_fn = partial(multivariate_normal_log_pdf, chol_cov=chol_cov)

        return rng_key, cov, logprior_fn, loglikelihood_fn

    def _get_hmc_config(self, num_dim: int) -> dict:
        """Get HMC kernel configuration."""
        return {
            "init": blackjax.hmc.init,
            "kernel": blackjax.hmc.build_kernel(),
            "parameters": extend_params(
                {
                    "step_size": 10e-2,
                    "inverse_mass_matrix": jnp.eye(num_dim),
                    "num_integration_steps": 50,
                }
            ),
        }

    def _compute_expected_log_likelihood(self, cov: jnp.ndarray, num_dim: int) -> float:
        """Compute expected log marginal likelihood for prior :math:`N(0, I)` and
        likelihood :math:`N(0, cov)`."""
        return -0.5 * np.linalg.slogdet(np.eye(num_dim) + cov)[
            1
        ] - num_dim / 2 * np.log(2 * np.pi)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_normalizing_constant_fixed_schedule(self) -> None:
        """Test that persistent sampling with fixed schedule accurately
        estimates the normalizing constant."""
        num_particles = 500
        num_dim = 2
        num_tempering_steps = 20

        rng_key, cov, logprior_fn, loglikelihood_fn = self._setup_test_problem(num_dim)
        hmc_config = self._get_hmc_config(num_dim)

        rng_key, init_key = jax.random.split(rng_key, 2)
        x_init = jax.random.normal(init_key, shape=(num_particles, num_dim))

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        ps = persistent_sampling_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_schedule=num_tempering_steps,
            mcmc_step_fn=hmc_config["kernel"],
            mcmc_init_fn=hmc_config["init"],
            mcmc_parameters=hmc_config["parameters"],
            resampling_fn=resampling.systematic,
            num_mcmc_steps=10,
        )

        init_state = ps.init(x_init)  # type: ignore
        rng_key, sample_key = jax.random.split(rng_key)

        result = self.variant(partial(inference_loop_fixed, kernel=ps.step))(
            rng_key=sample_key,
            initial_state=init_state,
            tempering_schedule=lambda_schedule,
        )

        # Calculate analytical expected marginal likelihood based
        # on a standard normal prior (:math:`N(0, I)`) and likelihood
        # :math:`N(0, cov)`.
        expected = self._compute_expected_log_likelihood(cov, num_dim)

        # Check that the estimated log marginal likelihood is close to the expected
        # value
        np.testing.assert_allclose(result.log_Z, expected, rtol=1e-1)

    @chex.variants(with_jit=True, without_jit=True)  # type: ignore
    def test_normalizing_constant_adaptive_schedule(self) -> None:
        """Test that persistent sampling with adaptive schedule accurately
        estimates the normalizing constant."""
        num_particles = 500
        num_dim = 2
        max_iterations = 30
        target_ess = 3

        rng_key, cov, logprior_fn, loglikelihood_fn = self._setup_test_problem(num_dim)
        hmc_config = self._get_hmc_config(num_dim)

        rng_key, init_key = jax.random.split(rng_key, 2)
        x_init = jax.random.normal(init_key, shape=(num_particles, num_dim))

        ps = adaptive_persistent_sampling_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=hmc_config["kernel"],
            mcmc_init_fn=hmc_config["init"],
            mcmc_parameters=hmc_config["parameters"],
            resampling_fn=resampling.systematic,
            target_ess=target_ess,
            num_mcmc_steps=10,
        )

        init_state = ps.init(x_init)  # type: ignore
        rng_key, sample_key = jax.random.split(rng_key)

        result = self.variant(
            partial(
                inference_loop_adaptive,
                kernel=ps.step,
                target_ess=target_ess,
                max_iterations=max_iterations,
            )
        )(
            rng_key=sample_key,
            initial_state=init_state,
        )

        # Calculate analytical expected marginal likelihood based
        # on a standard normal prior (:math:`N(0, I)`) and likelihood
        # :math:`N(0, cov)`.
        expected = self._compute_expected_log_likelihood(cov, num_dim)

        # check if we did not exceed max iterations, so that correct target
        # distribution was sampled
        # NOTE: This is not enforced in the algorithm, and needs to be
        # checked post-hoc, e.g. in the inference loop. Its checked here
        # to make sure that the final distribution gets sampled.
        assert result.iteration < max_iterations

        # Check that the estimated log marginal likelihood is close to the expected
        # value
        np.testing.assert_allclose(result.log_Z, expected, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
