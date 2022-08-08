"""
strategies to tune the parameters of mcmc kernels
used within smc, based on particles
"""

import jax
import jax.numpy as jnp


def no_tunning(mcmc_algorithm, mcmc_parameters):
    """
    default implementation that will not take particles
    into consideration to build a mcmc algorithm.
    """

    def kernel_factory(logprob_fn, particles):
        return mcmc_algorithm(logprob_fn, **mcmc_parameters).step

    return kernel_factory


def proposal_distribution_tunning(mcmc_algorithm, mcmc_parameters):
    """
    tunes the proposal distribution based on particles, in order
    to be used, an mcmc_parameter called "proposal_distribution_factory"
    needs to be in place.
    """

    def kernel_factory(logprob_fn, particles):
        if "proposal_distribution_factory" not in mcmc_parameters:
            raise ValueError(
                "in order to use proposal distribution tunning, you "
                "need to include a 'proposal_distribution_factory' parameter "
                "when specifying the kernels' mcmc_parameters"
            )

        proposal_distribution_factory = mcmc_parameters["proposal_distribution_factory"]
        return mcmc_algorithm(
            logprob_fn,
            proposal_distribution=proposal_distribution_factory(logprob_fn, particles),
        ).step

    return kernel_factory


def normal_proposal_from_particles(logprob_fn, particles):
    """
    builds a new normal proposal distribution based on
    particles mean and std. since particles
    are represented as lists with one element
    per posterior variable, we need to calculate
    mean/std per variable.
    """
    _, unravel_fn = jax.flatten_util.ravel_pytree(particles)

    particles_means = jax.tree_util.tree_map(
        lambda x: jax.numpy.mean(x, axis=0), particles
    )
    particles_stds = jax.tree_util.tree_map(
        lambda x: jax.numpy.std(x, axis=0), particles
    )

    def proposal_distribution(rng_key):
        to_return = jax.tree_map(
            lambda x: jax.random.normal(rng_key, shape=x.shape), particles_means
        )
        to_return = jax.tree_util.tree_map(jnp.multiply, particles_stds, to_return)
        to_return = jax.tree_util.tree_map(jnp.add, particles_means, to_return)
        return to_return

    return proposal_distribution
