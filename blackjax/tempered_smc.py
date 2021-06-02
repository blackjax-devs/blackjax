"""Public API for the Tempered SMC Kernel"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.inference.smc.ess import ess_solver
from blackjax.inference.smc.smc import SMCInfo, SMCState
from blackjax.inference.smc.smc import kernel as smc_kernel
from blackjax.inference.smc.solver import dichotomy_solver

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]

__all__ = ["adaptive_tempered_smc", "fixed_schedule_tempered_smc", "TemperedSMCState"]


class TemperedSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    n_iter: int
        Current tempered SMC step.
    smc_state: SMCState
        The particles' state.
    lmbda: float
        Current value of the tempering parameter.
    """

    n_iter: int
    smc_state: SMCState
    lmbda: float


class TemperedSMCInfo(NamedTuple):
    """Additional information on the tempered SMC step.

    total_steps: int
        Number of steps in the MCMC pass.
    smc_info: SMCInfo
        Information about the inner SMC sampler.
    """

    smc_info: SMCInfo


def adaptive_tempered_smc(
    prior_logrob_fn: Callable,
    potential_fn: Callable,
    mcmc_kernel_factory: Callable,
    new_mcmc_state: Callable,
    resampling_method: Callable,
    target_ess: float,
    root_solver: Callable = dichotomy_solver,
    use_log_ess: bool = True,
    mcmc_iter: int = 10,
):
    r"""Build a Tempered SMC step using adaptive schedule.

    Tempered SMC uses tempering to sample from a distribution given by

    :math..
        p(x) \propto p_0(x) \exp(-V(x)) \mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from and for which the density
    is easy to compute, and :math:`\exp(-V(x))` is an unnormalized likelihood term for which :math:`V(x)` is easy
    to compute pointwise.

    Parameters
    ----------
    prior_logrob_fn: Callable
        A function that computes the log density of the prior distribution
    potential_fn: Callable
        A function that returns the potential energy of a chain at a given position.
    mcmc_kernel_factory: Callable
        A callable function that creates a mcmc kernel from a potential function.
    new_mcmc_state: Callable
        How to create a new MCMC state from the SMC particles
    resampling_method: Callable
        A random function that resamples generated particles based of weights
    target_ess: float
        The target ESS for the adaptive MCMC tempering
    root_solver: Callable, optional
        A solver utility to find delta matching the target ESS. Signature is
        `root_solver(fun, delta_0, min_delta, max_delta)`, default is dichotomy_solver
    use_log_ess: bool, optional
        Use ESS in log space to solve for delta, default is `True`.
        This is usually more stable when using gradient based solvers.
    mcmc_iter: int
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def choose_lambda(state):
        lmbda = state.lmbda
        max_delta = 1 - lmbda
        delta = ess_solver(
            jax.vmap(potential_fn),
            state.smc_state,
            target_ess,
            max_delta,
            root_solver,
            use_log_ess,
        )
        delta = jnp.clip(delta, 0.0, max_delta)

        return delta

    return tempered_smc(
        prior_logrob_fn,
        potential_fn,
        mcmc_kernel_factory,
        new_mcmc_state,
        resampling_method,
        choose_lambda,
        mcmc_iter,
    )


def fixed_schedule_tempered_smc(
    prior_logrob_fn: Callable,
    potential_fn: Callable,
    mcmc_kernel_factory: Callable,
    new_mcmc_state: Callable,
    resampling_method: Callable,
    lambda_schedule: Union[np.ndarray, jnp.ndarray],
    mcmc_iter: int = 10,
):
    r"""Build a Tempered SMC step.

    Parameters
    ----------
    prior_logrob_fn: Callable
        A function that computes the log density of the prior distribution
    potential_fn: Callable
        A function that returns the potential energy of a chain at a given position.
    mcmc_kernel_factory: Callable
        A callable function that creates a mcmc kernel from a potential function.
    new_mcmc_state: Callable
        How to create a new MCMC state from the SMC particles
    resampling_method: Callable
        A random function that resamples generated particles based of weights
    lambda_schedule: array_like
        The fixed ahead schedule of tempering parameters. It is assumed to be increasing.
        If not the algorithm will return garbage. Not check is done.
    mcmc_iter: int
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    lambda_schedule = jnp.clip(lambda_schedule, 0.0, 1.0)

    def choose_lambda(state):
        iteration = state.n_iter
        next_lambda = lambda_schedule[iteration]
        delta = next_lambda - state.lmbda
        return delta

    return tempered_smc(
        prior_logrob_fn,
        potential_fn,
        mcmc_kernel_factory,
        new_mcmc_state,
        resampling_method,
        choose_lambda,
        mcmc_iter,
    )


def tempered_smc(
    prior_logrob_fn: Callable,
    potential_fn: Callable,
    mcmc_kernel_factory: Callable,
    new_mcmc_state: Callable,
    resampling_method: Callable,
    choose_lambda: Callable,
    mcmc_iter: int,
):
    """Build the base Tempered SMC kernel.

    Tempered SMC uses tempering to sample from a distribution given by

    :math..
        p(x) \propto p_0(x) \exp(-V(x)) \mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from and for which the density
    is easy to compute, and :math:`\exp(-V(x))` is an unnormalized likelihood term for which :math:`V(x)` is easy
    to compute pointwise.

    Parameters
    ----------
    prior_logrob_fn
        A function that computes the log density of the prior distribution
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    mcmc_kernel_factory
        A function that creates a mcmc kernel from a potential function.
    new_mcmc_state
        How to create a new MCMC state from the SMC particles
    resampling_method
        A random function that resamples generated particles based of weights
    choose_lambda
        A function that generates a new value of lambda given the current state of the chain.
    mcmc_iter: int
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    smc_step = smc_kernel(
        mcmc_kernel_factory, new_mcmc_state, resampling_method, mcmc_iter
    )

    def one_step(
        rng_key: jnp.ndarray, state: TemperedSMCState
    ) -> Tuple[TemperedSMCState, TemperedSMCInfo]:
        """Move the particles one step using the Tempered SMC algorithm.

        Parameters
        ----------
        rng_key
            JAX PRNGKey for randomness
        state
            Current state of the tempered SMC algorithm

        Returns
        -------
        state
            The new state of the tempered SMC algorithm
         info
            Additional information on the SMC step
        """
        delta = choose_lambda(state)

        log_weights_fn = lambda pytree: -delta * potential_fn(pytree)
        lambda_potential_fn = lambda pytree: -prior_logrob_fn(
            pytree
        ) + state.lmbda * potential_fn(pytree)

        smc_state, smc_info = smc_step(
            rng_key, state.smc_state, lambda_potential_fn, log_weights_fn
        )
        state = TemperedSMCState(state.n_iter + 1, smc_state, state.lmbda + delta)
        info = TemperedSMCInfo(smc_info)

        return state, info

    return one_step
