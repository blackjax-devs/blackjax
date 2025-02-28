from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_ns
from blackjax.ns.base import delete_fn
from blackjax.ns.base import init as init_base
from blackjax.ns.hr_slice import build_slice_kernel as slice_kernel
from blackjax.ns.hr_slice import init as slice_init
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "as_top_level_api", "build_kernel", "nss"]


def init(position, loglikelihood_fn, parameter_update_function):
    state = init_base(position, loglikelihood_fn)
    initial_parameter_value = parameter_update_function(
        state, NSInfo(state, state, state, None)
    )
    return StateWithParameterOverride(state, initial_parameter_value)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable,
    num_mcmc_steps: int,
) -> Callable:
    r"""Build an adaptive Nested Sampling kernel. Tunes the inner kernel parameters
    at each iteration.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    delete_fn : Callable
        Function that takes an array of log likelihoods and marks particles for deletion and updates.
    mcmc_step_fn:
        The initialisation of the transition kernel, should take as parameters.
        kernel = mcmc_step_fn(logprior, loglikelihood, logL0 (likelihood threshold), **mcmc_parameter_update_fn())
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        Function that updates the parameters of the inner kernel.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.

    Returns
    -------
    Callable
        A function that takes a rng_key and a NSState that contains the current state
        of the chain and returns a new state of the chain along with
        information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: StateWithParameterOverride,
    ) -> tuple[StateWithParameterOverride, NSInfo]:
        step_fn = base_ns(
            logprior_fn,
            loglikelihood_fn,
            delete_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            num_mcmc_steps,
        )
        new_state, info = step_fn(
            rng_key, state.sampler_state, state.parameter_override
        )
        new_parameter_override = mcmc_parameter_update_fn(new_state, info)
        return (
            StateWithParameterOverride(new_state, new_parameter_override),
            info,
        )

    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    mcmc_step_fn:
        The initialisation of the transition kernel, should take as parameters.
        kernel = mcmc_step_fn(logprior, loglikelihood, logL0 (likelihood threshold), **mcmc_parameter_update_fn())
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        A function that updates the parameters given the current state and info.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete : int, optional
        Number of particles to delete in each iteration. Default is 1.

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_step_fn,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, mcmc_parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)

def default_stepper(x, n):
    return jax.tree_map(lambda x, n: x + n, x, n)

def default_predict_fn(key, **kwargs):
    cov = kwargs["cov"]
    n = jax.random.multivariate_normal(
        key, mean=jnp.zeros(cov.shape[0]), cov=cov
    )
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", n, invcov, n))
    n = n / norm[..., None]
    return n

def default_train_fn(particles):
    cov = particles_covariance_matrix(particles)
    return {"cov": jnp.atleast_2d(cov)}

def nss(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_mcmc_steps: int,
    n_delete: int = 1,
    ravel_fn: Callable = lambda x: x,
    stepper: Callable = default_stepper,
    train_fn: Callable = default_train_fn,
    predict_fn: Callable = default_predict_fn,
) -> SamplingAlgorithm:
    """Implements the a baseline Nested Slice Sampling kernel.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log prior probability.
    loglikelihood_fn: Callable
        A function that computes the log likelihood.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete: int, optional
        Number of particles to delete in each iteration. Default is 1.
    ravel_fn: Callable, optional
        Optional function to ravel the proposal to the same shape as the state space, if needed
    stepper: Callable, optional
        Optional function to add the proposal to the current state, if needed

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)
    mcmc_init_fn = slice_init

    def parameter_update_fn(state, _):
        return train_fn(state.particles)

    def step(logprior, loglikelihood, logL0, **kwargs):
        def proposal_distribution(key):
            return ravel_fn(predict_fn(key, **kwargs))
        return slice_kernel(logprior, loglikelihood, logL0, proposal_distribution, stepper)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        step,
        mcmc_init_fn,
        parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
