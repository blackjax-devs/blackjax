from typing import Callable, Dict, NamedTuple, Tuple, Union

from blackjax.base import SamplingAlgorithm
from blackjax.smc.adaptive_tempered import adaptive_tempered_smc
from blackjax.smc.base import SMCInfo, SMCState
from blackjax.smc.tempered import tempered_smc
from blackjax.types import ArrayTree, PRNGKey


class StateWithParameterOverride(NamedTuple):
    sampler_state: ArrayTree
    parameter_override: ArrayTree


def init(alg_init_fn, position, initial_parameter_value):
    return StateWithParameterOverride(alg_init_fn(position), initial_parameter_value)


def build_kernel(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_factory: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: Dict,
    resampling_fn: Callable,
    mcmc_parameter_update_fn: Callable[[SMCState, SMCInfo], ArrayTree],
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> Callable:
    """In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
    MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
    based on particles. The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair).
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_factory
        A callable that can construct an inner kernel out of the newly-computed parameter
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameters
        Other (fixed across SMC iterations) parameters for the inner kernel
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the inner kernel in i+1 iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.
    """

    def kernel(
        rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_factory(state.parameter_override),
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters,
        ).step
        new_state, info = step_fn(rng_key, state.sampler_state, **extra_step_parameters)
        new_parameter_override = mcmc_parameter_update_fn(new_state, info)
        return StateWithParameterOverride(new_state, new_parameter_override), info

    return kernel


class inner_kernel_tuning:
    """In the context of an SMC sampler (whose step_fn returning state
    has a .particles attribute), there's an inner MCMC that is used
    to perturbate/update each of the particles. This adaptation tunes some
    parameter of that MCMC, based on particles.
    The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair).
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_factory
        A callable that can construct an inner kernel out of the newly-computed parameter
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameters
        Other (fixed across SMC iterations) parameters for the inner kernel step
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the
        inner kernel in i+1 iteration.
    initial_parameter_value
        Paramter to be used by the mcmc_factory before the first iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        smc_algorithm: Union[adaptive_tempered_smc, tempered_smc],
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_factory: Callable,
        mcmc_init_fn: Callable,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        mcmc_parameter_update_fn: Callable[[SMCState, SMCInfo], ArrayTree],
        initial_parameter_value,
        num_mcmc_steps: int = 10,
        **extra_parameters,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            smc_algorithm,
            logprior_fn,
            loglikelihood_fn,
            mcmc_factory,
            mcmc_init_fn,
            mcmc_parameters,
            resampling_fn,
            mcmc_parameter_update_fn,
            num_mcmc_steps,
            **extra_parameters,
        )

        def init_fn(position):
            return cls.init(smc_algorithm.init, position, initial_parameter_value)

        def step_fn(
            rng_key: PRNGKey, state, **extra_step_parameters
        ) -> Tuple[StateWithParameterOverride, SMCInfo]:
            return kernel(rng_key, state, **extra_step_parameters)

        return SamplingAlgorithm(init_fn, step_fn)
