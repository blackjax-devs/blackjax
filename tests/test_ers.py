"""Test for Ensemble rejection sampling"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
import scipy.stats as stats

import blackjax.ers as ers


def inference_loop(rng_key, kernel, num_samples):
    def one_step(_, rng_key):
        sample, accept = kernel(rng_key)
        return None, sample, accept

    keys = jax.random.split(rng_key, num_samples)
    _, (states, accepts) = jax.lax.scan(one_step, None, keys)

    return states, accepts


@pytest.mark.parametrize("b", [1., 2., 3.])
def test_truncated_normal(b):
    """Test the ERS kernel on a truncated Gaussian example."""
    key = jax.random.PRNGKey(42)
    n_samples = 10_000
    sampling_keys = jax.random.split(key, n_samples)
    potential_fn = lambda x: jnp.where(x > b, x ** 2 / 2, jnp.inf)
    proposal_sampler = lambda k, n: b + jax.random.normal(k, (n,))
    proposal_potential_fn = lambda x: (x - b) ** 2 / 2

    potentials_lower_bound = jnp.log(1 - jsp.stats.norm.cdf(b)) + b ** 2 / 2
    worst_proba_accept = np.exp(potentials_lower_bound)

    for m in [1, 50, 150]:
        kernel = ers.kernel(potential_fn, (proposal_sampler, proposal_potential_fn), potentials_lower_bound, m)
        states, accepts = jax.vmap(kernel)(sampling_keys)
        n_accepted = np.sum(accepts)
        lower_bound_proba = m * worst_proba_accept / (1 + (m - 1) * worst_proba_accept)

        # assert n_accepted / n_samples >= lower_bound_proba

        accepted_states = states[accepts]
        print()
        print(m)
        print(lower_bound_proba)
        print(n_accepted.item() / n_samples)
        print(n_accepted.item())
        print(stats.kstest(accepted_states, stats.truncnorm(a=b, b=np.inf).cdf))
