from typing import Tuple, Callable, List

import jax
from jax._src.flatten_util import ravel_pytree

from blackjax import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last, SMCInfo
import jax.random
import jax.numpy as jnp

from blackjax.smc.from_mcmc import step_from_mcmc_parameters
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride, init, InnerKernelTuningStrategy
from blackjax.smc.resampling import stratified
from blackjax.types import PRNGKey, ArrayLikeTree
from blackjax.util import generate_gaussian_noise


def esjd(m):
    """Implements ESJD (expected squared jumping distance). Inner Mahalanobis distance
    is computed using the Cholesky decomposition of M=LLt, and then inverting L.
    Whenever M is symmetrical definite positive then it must exist a Cholesky Decomposition. For example,
     if M is the Covariance Matrix of Metropolis-Hastings or the Inverse Mass Matrix of Hamiltonian Monte
    Carlo.
    """
    L = jnp.linalg.cholesky(m)
    print(L)

    def measure(previous_position, next_position, acceptance_probability):
        return acceptance_probability * jnp.power(
            jnp.linalg.norm(jnp.matmul(L, (ravel_pytree(previous_position)[0] - ravel_pytree(next_position)[0])), 2), 2)

    return jax.vmap(measure)


def update_parameter_distribution(
        key,
        previous_param_samples: ArrayLikeTree,
        previous_particles,
        latest_particles,
        measure_of_chain_mixing,
        alpha,
        sigma_parameters: ArrayLikeTree,
        acceptance_probability
):
    """Given an existing parameter distribution that were used to mutate previous_particles
    into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
    noise to those samples. The weights used are a linear function of the measure of chain mixing.
    Only works with float parameters, not integers.
    See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf

    Parameters
    ----------
    previous_param_samples:
        samples of the parameters of SMC inner MCMC chains. To be updated.
    previous_particles:
        particles from which the kernel step started
    latest_particles:
        particles after the step was performed
    measure_of_chain_mixing: Callable
        a callable that can compute a performance measure per chain
    alpha:
        a scalar to add to the weighting. See paper for details
    sigma_parameters:
        must have the same shape of previous_param_samples.
    acceptance_probability:
        the energy difference for each of the chains when taking a step from previous_particles
        into latest_particles.
    """
    noise_key, resampling_key = jax.random.split(key, 2)
    print(previous_param_samples)
    noises = jax.tree.map(lambda x, s: generate_gaussian_noise(noise_key, x.astype("float32"), sigma=s),
                          previous_param_samples, sigma_parameters)
    new_samples = jax.tree.map(lambda x, y: x + y, noises, previous_param_samples)

    # TODO SHOULD WE ADD SOME CHECK HERE TO AVOID AN INSANE AMMOUNT OF NOISEx
    chain_mixing_measurement = measure_of_chain_mixing(previous_particles, latest_particles, acceptance_probability)
    weights = alpha + chain_mixing_measurement
    weights = weights / jnp.sum(weights)
    resampling_idx = stratified(resampling_key, weights, len(chain_mixing_measurement))
    return jax.tree.map(lambda x: x[resampling_idx], new_samples), chain_mixing_measurement


def build_pretune(mcmc_init_fn,
                  mcmc_step_fn,
                  alpha,
                  sigma_parameters,
                  parameters_to_pretune: List[str],
                  performance_of_chain_measure_factory: Callable = lambda state: esjd(
                      state.parameter_override["inverse_mass_matrix"]),
                  round_to_integer: List[str] = None
                  ):
    """
    Implements Buchholz et al https://arxiv.org/pdf/1808.07730 pretuning procedure. The goal is to maintain
    a probability distribution of parameters, in order to assign different values to each inner MCMC chain.
    To have performant parameters for the distribution at step t, it takes a single step, measures
    the chain mixing, and reweights the probability distribution of parameters accordingly.
    Note that although similar, this strategy is different than inner_kernel_tuning. The latter updates
    the parameters based on the particles and transition information after the SMC step is executed. This
    implementation runs a single MCMC step which gets discarded, to then proceed with the SMC step execution.
    """
    if round_to_integer is None:
        round_to_integer_fn = lambda x: x
    else:
        def round_to_integer_fn(x):
            for k in round_to_integer:
                x[k] = jax.tree.map(lambda a: jnp.round(a).astype(int), x[k])
            return x

    def pretune(key, state, logposterior):
        unshared_mcmc_parameters, shared_mcmc_step_fn = step_from_mcmc_parameters(state.parameter_override,
                                                                                  mcmc_step_fn)
        one_step_fn, _ = update_and_take_last(mcmc_init_fn,
                                              logposterior,
                                              shared_mcmc_step_fn,
                                              1,
                                              100)

        print(jax.tree.map(lambda x: x.shape, state.sampler_state.particles))
        print(jax.tree.map(lambda x: x.shape, unshared_mcmc_parameters))
        new_state, info = one_step_fn(jax.random.split(key, 100),
                                      state.sampler_state.particles,
                                      unshared_mcmc_parameters)

        performance_of_chain_measure = performance_of_chain_measure_factory(state)
        print(state.parameter_override)
        print(state.parameter_override.fromkeys(parameters_to_pretune))
        new_parameter_distribution, chain_mixing_measurement = update_parameter_distribution(key,
                                                                                             previous_param_samples={
                                                                                                 key:
                                                                                                     state.parameter_override[
                                                                                                         key] for key in
                                                                                                 parameters_to_pretune},
                                                                                             previous_particles=state.sampler_state.particles,
                                                                                             latest_particles=new_state,
                                                                                             measure_of_chain_mixing=performance_of_chain_measure,
                                                                                             alpha=alpha,
                                                                                             sigma_parameters=sigma_parameters,
                                                                                             acceptance_probability=info.acceptance_rate)

        return round_to_integer_fn(new_parameter_distribution), chain_mixing_measurement

    def pretune_and_update(key, state: StateWithParameterOverride, logposterior):
        """
        Updates the parameters that need to be pretuned and returns the rest.
        """
        new_parameter_distribution, chain_mixing_measurement = pretune(key, state, logposterior)
        old_parameter_distribution = state.parameter_override
        updated_parameter_distribution = old_parameter_distribution
        for k in new_parameter_distribution:
            updated_parameter_distribution[k] = new_parameter_distribution[k]

        return updated_parameter_distribution

    return pretune_and_update
