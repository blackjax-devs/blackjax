"""Test MCMC diagnostics."""
import functools
import itertools

import chex
import jax
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


if __name__ == "__main__":
    absltest.main()
