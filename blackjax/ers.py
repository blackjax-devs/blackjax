"""Public API for (Ensemble) rejection sampling kernels."""
import math
from typing import Callable, Tuple

__all__ = ["kernel"]

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print

import blackjax.inference.smc.base as base
import blackjax.inference.smc.resampling as resampling
from blackjax.types import PRNGKey, PyTree


def kernel(potential_fn: Callable,
           proposal: Tuple[Callable[[PRNGKey, int], PyTree], Callable[[PyTree], float]],
           potential_lower_bound: float,
           n: int,
           ) -> Callable:
    """
    Random Walk Metropolis Hastings algorithm with normal proposals.

    Parameters
    ----------
    potential_fn: Callable
        The (unnormalized) negative log-density to sample from.
    proposal: Tuple[Callable, Callable]
        Tuple containing a proposal sampler as a first argument, and the corresponding (unnormalized) potential function
        as a second argument. The sampler takes a PRNGKey and the number of samples and returns a PyTree of the samples
        batched along the first dimension.
    potential_lower_bound: float
        The quantity `exp(-potential_fn(x)) / exp(-proposal_potential_fn(x))` needs to be bounded above for
        the algorithm to be correct. This corresponds to a lower bound on the difference
        `proposal_potential_fn(x) - potential_fn(x)`
    n: int, optional
        Number of samples to use in the intermediate importance sample. Note that `n=1` recovers
        standard rejection sampling.

    References
    ----------
    [1] George Deligiannidis, Arnaud Doucet, Sylvain Rubenthaler. Ensemble Rejection Sampling.
        https://arxiv.org/abs/2001.09188

    """
    proposal_sampler, proposal_potential_fn = proposal
    log_n = math.log(n)

    def one_step(rng_key: PRNGKey) -> PyTree:
        """

        Parameters
        ----------
        rng_key: PRNGKey
            The random key to use to sample

        Returns
        -------
        sample: PyTree
            A sample corresponding to the target potential.
        accept: bool
            Is the sample accepted or not
        """
        sampling_key, resampling_key, rejection_key = jax.random.split(rng_key, 3)
        proposals = proposal_sampler(sampling_key, n)
        log_weights = jax.vmap(lambda z: proposal_potential_fn(z) - potential_fn(z))(proposals)
        log_weights, log_normalizing_constant = base.normalize(log_weights, return_log=True)
        idx = resampling.multinomial(jnp.exp(log_weights), resampling_key, 1)[0]
        proposed_sample = proposals[idx]

        proposed_log_weight = log_weights[idx]
        log_upper_bound = jnp.logaddexp(log_normalizing_constant,
                                        -log_n + jnp.logaddexp(-potential_lower_bound, -proposed_log_weight))

        log_alpha = log_normalizing_constant - log_upper_bound
        p_accept = jnp.clip(jnp.exp(log_alpha), a_max=1)
        do_accept = jax.random.bernoulli(rejection_key, p_accept)
        return proposed_sample, do_accept

    return one_step
