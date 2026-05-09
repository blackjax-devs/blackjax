"""Test MCMC diagnostics."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.diagnostics as diagnostics

test_cases = [
    {
        "chain_axis": 0,
        "sample_axis": 1,
    },
    {
        "chain_axis": 1,
        "sample_axis": 0,
    },
    {
        "chain_axis": 0,
        "sample_axis": -1,
    },
    {
        "chain_axis": -1,
        "sample_axis": 0,
    },
]


def insert_list(input_list, loc, elem):
    if loc == -1:
        input_list.append(elem)
    else:
        input_list.insert(loc, elem)
    return input_list


class DiagnosticsTest(chex.TestCase):
    """Tests for MCMC diagnostics."""

    def setUp(self):
        super().setUp()
        self.num_samples = 5000
        self.test_seed = 32

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        itertools.product(test_cases, [1, 2, 10], [(), (3,), (5, 7)])
    )
    def test_rhat_ess(self, case, num_chains, event_shape):
        rng_key = jax.random.key(self.test_seed)
        sample_shape = list(event_shape)
        if case["chain_axis"] < case["sample_axis"]:
            sample_shape = insert_list(sample_shape, case["chain_axis"], num_chains)
            sample_shape = insert_list(
                sample_shape, case["sample_axis"], self.num_samples
            )
        else:
            sample_shape = insert_list(
                sample_shape, case["sample_axis"], self.num_samples
            )
            sample_shape = insert_list(sample_shape, case["chain_axis"], num_chains)
        mc_samples = jax.random.normal(rng_key, shape=sample_shape)

        potential_scale_reduction = self.variant(
            functools.partial(diagnostics.potential_scale_reduction, **case)
        )
        if num_chains > 1:
            rhat_val = potential_scale_reduction(mc_samples)
            np.testing.assert_array_equal(rhat_val.shape, event_shape)
            np.testing.assert_allclose(rhat_val, 1.0, rtol=1e-03)
        else:
            np.testing.assert_raises(
                AssertionError, potential_scale_reduction, mc_samples
            )

        # With iid samples we should get ess close to number of samples.
        effective_sample_size = self.variant(
            functools.partial(diagnostics.effective_sample_size, **case)
        )
        ess_val = effective_sample_size(mc_samples)
        np.testing.assert_array_equal(ess_val.shape, event_shape)
        np.testing.assert_allclose(ess_val, num_chains * self.num_samples, rtol=10)


    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        # (num_superchains, num_chains, num_samples, event_shape)
        (4, 8, 100, ()),
        (4, 8, 100, (3,)),
        (8, 4, 50, (5,)),
        (4, 1, 200, ()),  # M=1: reduces to classical Rhat
    )
    def test_nested_rhat_converged(
        self, num_superchains, num_chains, num_samples, event_shape
    ):
        """Nested Rhat should be close to 1 for IID samples from the target."""
        rng_key = jax.random.key(self.test_seed)
        sample_shape = (num_superchains, num_chains, num_samples) + event_shape
        mc_samples = jax.random.normal(rng_key, shape=sample_shape)

        nested_rhat_fn = self.variant(diagnostics.nested_rhat)
        nrhat_val = nested_rhat_fn(mc_samples)
        np.testing.assert_array_equal(nrhat_val.shape, event_shape)
        np.testing.assert_allclose(nrhat_val, 1.0, rtol=0.05)

    @chex.all_variants(with_pmap=False)
    def test_nested_rhat_detects_nonconvergence(self):
        """Nested Rhat >> 1 when superchains have different means (bias)."""
        rng_key = jax.random.key(self.test_seed)
        K, M, N = 4, 8, 100
        # Each superchain draws from a distribution with a different mean,
        # simulating chains that haven't converged from their initial points.
        means = jnp.array([0.0, 5.0, -5.0, 10.0]).reshape(K, 1, 1)
        samples = means + jax.random.normal(rng_key, shape=(K, M, N))

        nested_rhat_fn = self.variant(diagnostics.nested_rhat)
        nrhat_val = nested_rhat_fn(samples)
        # Should be well above 1 since superchains haven't converged
        self.assertGreater(float(nrhat_val), 2.0)

    @chex.all_variants(with_pmap=False)
    def test_nested_rhat_m1_matches_classical(self):
        """When M=1, nested Rhat should approximate classical Rhat.

        Per Remark 2.3 in Margossian et al. (2024), nested Rhat reduces to the
        classical Rhat when each superchain contains a single chain (M=1).
        """
        rng_key = jax.random.key(self.test_seed)
        K, N = 10, 500
        # IID samples: both should be close to 1
        samples_nested = jax.random.normal(rng_key, shape=(K, 1, N))
        samples_classical = samples_nested.squeeze(axis=1)  # (K, N)

        nested_rhat_fn = self.variant(diagnostics.nested_rhat)
        classical_rhat_fn = self.variant(diagnostics.potential_scale_reduction)

        nrhat_val = nested_rhat_fn(samples_nested)
        rhat_val = classical_rhat_fn(samples_classical)
        np.testing.assert_allclose(nrhat_val, rhat_val, rtol=0.01)

    @chex.all_variants(with_pmap=False)
    def test_nested_rhat_requires_multiple_superchains(self):
        """Nested Rhat should raise when K < 2."""
        rng_key = jax.random.key(self.test_seed)
        samples = jax.random.normal(rng_key, shape=(1, 4, 100))
        nested_rhat_fn = self.variant(diagnostics.nested_rhat)
        with self.assertRaises(AssertionError):
            nested_rhat_fn(samples)


if __name__ == "__main__":
    absltest.main()
