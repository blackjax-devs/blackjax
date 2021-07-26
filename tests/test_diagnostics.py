"""Test MCMC diagnoistics."""
import jax
import numpy as np
import pytest

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


@pytest.mark.parametrize("case", test_cases)
@pytest.mark.parametrize("nchain", [2, 10])
@pytest.mark.parametrize("event_shape", [(), (3,), (5, 7)])
def test_rhat_ess(case, nchain, event_shape):
    rng_key = jax.random.PRNGKey(32)
    n_samples = 5000
    sample_shape = list(event_shape)
    if case["chain_axis"] < case["sample_axis"]:
        sample_shape = insert_list(sample_shape, case["chain_axis"], nchain)
        sample_shape = insert_list(sample_shape, case["sample_axis"], n_samples)
    else:
        sample_shape = insert_list(sample_shape, case["sample_axis"], n_samples)
        sample_shape = insert_list(sample_shape, case["chain_axis"], nchain)
    mc_samples = jax.random.normal(rng_key, shape=sample_shape)

    rhat_val = diagnostics.potential_scale_reduction(mc_samples, **case)
    np.testing.assert_array_equal(rhat_val.shape, event_shape)
    np.testing.assert_allclose(rhat_val, 1.0, rtol=1e-03)

    # With iid samples we should get ess close to number of samples.
    ess_val = diagnostics.effective_sample_size(mc_samples, **case)
    np.testing.assert_array_equal(ess_val.shape, event_shape)
    np.testing.assert_allclose(ess_val, nchain * n_samples, rtol=10)
