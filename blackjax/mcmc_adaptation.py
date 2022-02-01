from typing import Callable, Union

import jax

from blackjax import hmc, nuts
from blackjax.adaptation.window_adaptation import (
    window_adaptation_base,
    window_adaptation_schedule,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["window_adaptation"]


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logprob_fn: Callable,
    num_steps: int = 1000,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
    **parameters,
) -> AdaptationAlgorithm:
    """Adapt the parameters of algorithms in the HMC family.

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logprob_fn
        The log density probability density function from which we wish to sample.
    num_steps
        The number of adaptation steps.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    **parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the
    tuned parameter values from an initial state.

    """

    kernel = algorithm.new_kernel()

    def kernel_factory(step_size: float, inverse_mass_matrix: Array):
        def kernel_fn(rng_key, state):
            return kernel(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                **parameters,
            )

        return kernel_fn

    schedule_fn = window_adaptation_schedule(num_steps)
    init, update, final = window_adaptation_base(
        kernel_factory,
        schedule_fn,
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    @jax.jit
    def one_step(carry, rng_key):
        state, warmup_state = carry
        state, warmup_state, info = update(rng_key, state, warmup_state)
        return ((state, warmup_state), (state, warmup_state, info))

    def run(rng_key: PRNGKey, position: PyTree):
        init_state = algorithm.init(position, logprob_fn)
        init_warmup_state = init(init_state, initial_step_size)

        keys = jax.random.split(rng_key, num_steps)
        last_state, warmup_chain = jax.lax.scan(
            one_step,
            (init_state, init_warmup_state),
            keys,
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = final(last_warmup_state)
        kernel = kernel_factory(step_size, inverse_mass_matrix)

        return last_chain_state, kernel, warmup_chain

    return AdaptationAlgorithm(run)
