"""Unit tests for the full K-fold MEADS adaptation (issue #781)."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from blackjax.adaptation.meads_adaptation import MEADSAdaptationState, base
from tests.fixtures import BlackJAXTest


def make_logdensity(dim=2):
    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    return logdensity


class TestMEADSBase(BlackJAXTest):
    """Tests for the base() init/update functions."""

    def test_init_shape(self):
        """Fold parameters should have leading axis of size num_folds."""
        num_chains, num_folds, dim = 8, 4, 3
        init, _ = base(num_folds=num_folds)

        positions = jnp.ones((num_chains, dim))
        grads = jnp.ones((num_chains, dim))
        state = init(positions, grads)

        self.assertIsInstance(state, MEADSAdaptationState)
        self.assertEqual(state.step_size.shape, (num_folds,))
        self.assertEqual(state.alpha.shape, (num_folds,))
        self.assertEqual(state.delta.shape, (num_folds,))
        self.assertEqual(state.position_sigma.shape, (num_folds, dim))

    def test_init_same_params_across_folds(self):
        """All folds should start with identical parameters."""
        num_chains, num_folds, dim = 8, 4, 3
        init, _ = base(num_folds=num_folds)

        positions = jax.random.normal(self.next_key(), (num_chains, dim))
        grads = jax.random.normal(self.next_key(), (num_chains, dim))
        state = init(positions, grads)

        # All fold step sizes should be equal at init
        np.testing.assert_array_equal(state.step_size, state.step_size[0])
        np.testing.assert_array_equal(state.alpha, state.alpha[0])
        np.testing.assert_array_equal(state.delta, state.delta[0])

    def test_update_only_changes_target_fold(self):
        """update() should only modify the target fold's parameters."""
        num_chains, num_folds, dim = 8, 4, 3
        init, update = base(num_folds=num_folds)
        n_per_fold = num_chains // num_folds

        positions = jnp.ones((num_chains, dim))
        grads = jnp.ones((num_chains, dim))
        state = init(positions, grads)

        source_fold = 0
        target_fold = 1
        source_positions = positions[:n_per_fold]
        source_grads = grads[:n_per_fold] * 2.0  # different from init

        new_state = update(state, source_positions, source_grads, source_fold)

        # Only target fold changed
        self.assertEqual(new_state.current_iteration, 1)
        np.testing.assert_array_equal(new_state.step_size[0], state.step_size[0])
        np.testing.assert_array_equal(new_state.step_size[2], state.step_size[2])
        np.testing.assert_array_equal(new_state.step_size[3], state.step_size[3])
        # Target fold should differ (grads * 2 → different epsilon)
        self.assertFalse(
            jnp.allclose(new_state.step_size[target_fold], state.step_size[target_fold])
        )

    def test_step_size_multiplier_effect(self):
        """step_size_multiplier should scale the step size."""
        num_chains, dim = 8, 3
        positions = jnp.ones((num_chains, dim))
        grads = jnp.ones((num_chains, dim))

        init1, _ = base(num_folds=4, step_size_multiplier=0.5)
        init2, _ = base(num_folds=4, step_size_multiplier=1.0)

        state1 = init1(positions, grads)
        state2 = init2(positions, grads)

        np.testing.assert_allclose(state2.step_size, state1.step_size * 2.0, rtol=1e-5)

    def test_damping_slowdown_effect(self):
        """Higher damping_slowdown should produce stronger damping (higher alpha)."""
        num_chains, dim = 8, 3
        positions = jax.random.normal(self.next_key(), (num_chains, dim))
        grads = jax.random.normal(self.next_key(), (num_chains, dim))

        init1, _ = base(num_folds=4, damping_slowdown=1.0)
        init2, _ = base(num_folds=4, damping_slowdown=10.0)

        state1 = init1(positions, grads)
        state2 = init2(positions, grads)

        # Higher damping_slowdown raises the floor on gamma, so alpha is larger.
        # The floor term is damping_slowdown/(t*eps); with the eigenvalue-based
        # term fixed, a 10x larger floor must produce alpha >= the 1x case.
        self.assertTrue(
            jnp.all(state2.alpha >= state1.alpha),
            "Higher damping_slowdown should produce alpha >= lower damping_slowdown",
        )

    def test_invalid_num_folds_base(self):
        """base() should raise for num_folds < 1."""
        with pytest.raises(ValueError, match="num_folds"):
            base(num_folds=0)


class TestMEADSAdaptation(BlackJAXTest):
    """Tests for the full meads_adaptation run loop."""

    def _make_problem(self, num_chains=16, dim=2):
        logdensity = make_logdensity(dim)
        positions = jax.random.normal(self.next_key(), (num_chains, dim))
        return logdensity, positions

    def test_invalid_num_chains(self):
        """Should raise if num_chains not divisible by num_folds."""
        with pytest.raises(ValueError, match="divisible"):
            blackjax.meads_adaptation(make_logdensity(), num_chains=10, num_folds=4)

    def test_invalid_num_folds(self):
        """Should raise if num_folds < 1."""
        with pytest.raises(ValueError, match="num_folds"):
            blackjax.meads_adaptation(make_logdensity(), num_chains=8, num_folds=0)
        with pytest.raises(ValueError, match="num_folds"):
            blackjax.meads_adaptation(make_logdensity(), num_chains=8, num_folds=-1)

    def test_output_shapes(self):
        """Returned states and parameters should have correct shapes."""
        num_chains, dim = 16, 2
        logdensity, positions = self._make_problem(num_chains, dim)

        warmup = blackjax.meads_adaptation(
            logdensity, num_chains=num_chains, num_folds=4
        )
        (last_states, params), _ = warmup.run(self.next_key(), positions, num_steps=10)

        self.assertEqual(last_states.position.shape, (num_chains, dim))
        # Parameters should be scalar (mean across folds)
        self.assertEqual(params["step_size"].shape, ())
        self.assertEqual(params["alpha"].shape, ())
        self.assertEqual(params["delta"].shape, ())
        self.assertEqual(params["momentum_inverse_scale"].shape, (dim,))

    def test_folds_develop_different_params(self):
        """After several steps the folds should have different parameters."""
        num_chains, dim = 16, 3
        logdensity, positions = self._make_problem(num_chains, dim)

        # Run enough steps that folds diverge (at least num_folds cycles)
        num_folds = 4

        # We can't directly inspect per-fold state from the public API,
        # so use base() directly.
        n_per_fold = num_chains // num_folds
        init, update = base(num_folds=num_folds)

        positions_varied = jax.random.normal(self.next_key(), (num_chains, dim))
        grads_varied = jax.random.normal(self.next_key(), (num_chains, dim))

        state = init(positions_varied, grads_varied)

        # Run num_folds update cycles with different stats per fold
        for t in range(num_folds * 2):
            source = t % num_folds
            src_pos = positions_varied[source * n_per_fold : (source + 1) * n_per_fold]
            src_grad = grads_varied[source * n_per_fold : (source + 1) * n_per_fold] * (
                source + 1
            )  # vary by fold
            state = update(state, src_pos, src_grad, source)

        # Folds should have different step sizes
        self.assertFalse(jnp.allclose(state.step_size[0], state.step_size[1]))

    def test_num_folds_1_chains_advance(self):
        """With num_folds=1 all chains must advance every step (no fold frozen)."""
        num_chains, dim = 8, 2
        logdensity, positions = self._make_problem(num_chains, dim)

        warmup = blackjax.meads_adaptation(
            logdensity, num_chains=num_chains, num_folds=1
        )
        (last_states, params), warmup_info = warmup.run(
            self.next_key(), positions, num_steps=5
        )

        # With num_folds=1 the single fold is never frozen, so every chain
        # should have moved from its starting position by at least one step.
        init_pos = np.array(positions)
        final_pos = np.array(last_states.position)
        self.assertFalse(
            np.allclose(init_pos, final_pos),
            "All chains stayed at init with num_folds=1 — fold was incorrectly frozen",
        )

        # The per-step trajectory: at no step should all chains be unchanged
        # from the previous step (which would indicate a full freeze).
        pos_trace = np.array(warmup_info.state.position)  # [5, num_chains, dim]
        for t in range(1, 5):
            self.assertFalse(
                np.allclose(pos_trace[t], pos_trace[t - 1]),
                f"All chains frozen at step {t} with num_folds=1",
            )

        # Output shapes should still be correct with a single fold
        self.assertEqual(params["step_size"].shape, ())

    @chex.assert_max_traces(n=2)
    def test_no_recompilation(self):
        """The scan body should not retrace on repeated calls."""
        num_chains, dim = 8, 2
        logdensity, positions = self._make_problem(num_chains, dim)

        warmup = blackjax.meads_adaptation(
            logdensity, num_chains=num_chains, num_folds=4
        )
        warmup.run(self.next_key(), positions, num_steps=5)
        warmup.run(self.next_key(), positions, num_steps=5)

    def test_run_produces_valid_samples(self):
        """After adaptation, sampling with the returned parameters should work."""
        num_chains, dim = 16, 2
        logdensity, positions = self._make_problem(num_chains, dim)

        warmup = blackjax.meads_adaptation(
            logdensity, num_chains=num_chains, num_folds=4
        )
        (last_states, params), _ = warmup.run(self.next_key(), positions, num_steps=50)

        ghmc = blackjax.ghmc(logdensity, **params)
        step = jax.jit(jax.vmap(ghmc.step))
        keys = jax.random.split(self.next_key(), num_chains)
        new_states, info = step(keys, last_states)

        self.assertEqual(new_states.position.shape, (num_chains, dim))
        # Acceptance rates should be finite
        self.assertTrue(jnp.all(jnp.isfinite(new_states.logdensity)))
