"""Public API for the Tempered SMC Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

import blackjax.inference.smc.solver as solver
from blackjax.inference.smc.base import SMCInfo, smc
from blackjax.inference.smc.ess import ess_solver
from blackjax.types import PRNGKey, PyTree

__all__ = ["adaptive_tempered_smc", "tempered_smc"]


class TemperedSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.
    """

    particles: PyTree
    lmbda: float


def adaptive_tempered_smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_kernel_factory: Callable,
    make_mcmc_state: Callable,
    resampling_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    use_log_ess: bool = True,
    mcmc_iter: int = 10,
) -> Callable:
    r"""Build a Tempered SMC step using an adaptive schedule.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log-prior density.
    loglikelihood_fn: Callable
        A function that returns the log-likelihood density.
    mcmc_kernel_factory: Callable
        A callable function that creates a mcmc kernel from a log-probability
        density function.
    make_mcmc_state: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        A random function that resamples generated particles based of weights
    target_ess: float
        The target ESS for the adaptive MCMC tempering
    root_solver: Callable, optional
        A solver utility to find delta matching the target ESS. Signature is
        `root_solver(fun, delta_0, min_delta, max_delta)`, default is a dichotomy solver
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

    def compute_delta(state: TemperedSMCState) -> float:
        lmbda = state.lmbda
        max_delta = 1 - lmbda
        delta = ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            target_ess,
            max_delta,
            root_solver,
            use_log_ess,
        )
        delta = jnp.clip(delta, 0.0, max_delta)

        return delta

    kernel = tempered_smc(
        logprior_fn,
        loglikelihood_fn,
        mcmc_kernel_factory,
        make_mcmc_state,
        resampling_fn,
        mcmc_iter,
    )

    def one_step(
        rng_key: PRNGKey, state: TemperedSMCState
    ) -> Tuple[TemperedSMCState, SMCInfo]:
        delta = compute_delta(state)
        lmbda = delta + state.lmbda
        return kernel(rng_key, state, lmbda)

    return one_step


def tempered_smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_kernel_factory: Callable,
    make_mcmc_state: Callable,
    resampling_fn: Callable,
    num_mcmc_iterations: int,
) -> Callable:
    """Build the base Tempered SMC kernel.

    Tempered SMC uses tempering to sample from a distribution given by

    :math..
        p(x) \\propto p_0(x) \\exp(-V(x)) \\mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from and for which the density
    is easy to compute, and :math:`\\exp(-V(x))` is an unnormalized likelihood term for which :math:`V(x)` is easy
    to compute pointwise.

    Parameters
    ----------
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given
        position.
    mcmc_kernel_factory
        A function that creates a mcmc kernel from a log-probability density function.
    make_mcmc_state: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn
        A random function that resamples generated particles based of weights
    num_mcmc_iterations
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    kernel = smc(
        mcmc_kernel_factory, make_mcmc_state, resampling_fn, num_mcmc_iterations
    )

    def one_step(
        rng_key: PRNGKey, state: TemperedSMCState, lmbda: float
    ) -> Tuple[TemperedSMCState, SMCInfo]:
        """Move the particles one step using the Tempered SMC algorithm.

        Parameters
        ----------
        rng_key
            JAX PRNGKey for randomness
        state
            Current state of the tempered SMC algorithm
        lmbda
            Current value of the tempering parameter

        Returns
        -------
        state
            The new state of the tempered SMC algorithm
        info
            Additional information on the SMC step
        """
        delta = lmbda - state.lmbda

        def log_weights_fn(position: PyTree) -> float:
            return delta * loglikelihood_fn(position)

        def tempered_logposterior_fn(position: PyTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = state.lmbda * loglikelihood_fn(position)
            return logprior + tempered_loglikelihood

        smc_state, info = kernel(
            rng_key, state.particles, tempered_logposterior_fn, log_weights_fn
        )
        state = TemperedSMCState(smc_state, state.lmbda + delta)

        return state, info

    return one_step
