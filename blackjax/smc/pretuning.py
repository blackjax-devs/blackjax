from typing import Tuple, Callable

import jax

from blackjax import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last, SMCInfo
import jax.random
import jax.numpy as jnp

from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride, init, InnerKernelTuningStrategy
from blackjax.types import PRNGKey
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
            jnp.linalg.norm(jnp.matmul(L, (previous_position - next_position)), 2), 2)

    return jax.vmap(measure)


def update_parameter_distribution(
        key,
        previous_param_samples,
        previous_particles,
        latest_particles,
        measure_of_chain_mixing,
        alpha,
        sigma_parameters,
        acceptance_probability
):
    """Given an existing parameter distribution that were used to mutate previous_particles
    into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
    noise to those samples. The weights used are a linear function of the measure of chain mixing.
    Only works with float parameters, not integers.
    See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf
    """
    noise_key, resampling_key = jax.random.split(key, 2)
    new_samples = generate_gaussian_noise(noise_key, previous_param_samples, mu=previous_param_samples,
                                          sigma=sigma_parameters)
    # TODO SHOULD WE ADD SOME CHECK HERE TO AVOID AN INSANE AMMOUNT OF NOISE
    chain_mixing_measurement = measure_of_chain_mixing(previous_particles, latest_particles, acceptance_probability)
    weights = alpha + chain_mixing_measurement
    weights = weights / jnp.sum(weights)
    return jax.random.choice(
        resampling_key,
        new_samples,
        shape=(len(previous_param_samples),),
        replace=True,
        p=weights,
    ), chain_mixing_measurement


def build_kernel(mcmc_init_fn,
                 mcmc_step_fn,
                 logposterior,
                 alpha,
                 sigma_parameters,

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

    def pretune(key, state):
        one_step_fn, _ = update_and_take_last(mcmc_init_fn,
                                              logposterior,
                                              mcmc_step_fn,
                                              1,
                                              100)
        new_state, info = one_step_fn(key, state.sampler_state.particles, state.parameter_override)

        new_parameter_distribution, chain_mixing_measurement = update_parameter_distribution(key,
                                                                                             state.parameter_override,
                                                                                             state.sampler_state.particles,
                                                                                             new_state.particles,
                                                                                             esjd(
                                                                                                 state.inverse_mass_matrix),
                                                                                             alpha,
                                                                                             sigma_parameters,
                                                                                             info.acceptance_probability)
        return new_parameter_distribution, chain_mixing_measurement

    def step(key, state: StateWithParameterOverride):
        """
        Keeps the non-updated parameters
        """
        new_parameter_distribution, chain_mixing_measurement = pretune(key, state)
        old_parameter_distribuold_parameter_distributiontion = state.parameter_override

        return StateWithParameterOverride(state, state.parameter_override.update(new_parameter_distribution))

    return step


def as_top_level_api(
        smc_algorithm,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        resampling_fn: Callable,
        mcmc_parameter_update_fn: InnerKernelTuningStrategy,
        initial_parameter_value,
        num_mcmc_steps: int = 10,
        **extra_parameters,
) -> SamplingAlgorithm:
    """In the context of an SMC sampler (whose step_fn returning state
    has a .particles attribute), there's an inner MCMC that is used
    to perturbate/update each of the particles. This adaptation tunes some
    parameter of that MCMC, based on particles.
    The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc, blackjax.tempered_smc or blackjax.inner_kernel_tuning (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair). See blackjax.smc_family
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_step_fn
        The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the
        inner kernel in i+1 iteration.
    initial_parameter_value
        Parameter to be used by the mcmc_factory before the first iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel(
        smc_algorithm,
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
        **extra_parameters,
    )

    def init_fn(position, rng_key=None):
        del rng_key
        return init(smc_algorithm.init, position, initial_parameter_value)

    def step_fn(
            rng_key: PRNGKey, state, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        return kernel(rng_key, state, **extra_step_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
