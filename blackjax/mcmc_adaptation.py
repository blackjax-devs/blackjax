from typing import Callable, Union

import jax

from blackjax import hmc, nuts
from blackjax.base import AdaptationAlgorithm
from blackjax.stan_warmup import window_adaptation_base, window_adaptation_schedule
from blackjax.types import Array, PRNGKey, PyTree


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logprob_fn: Callable,
    num_steps: int = 1000,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
    **parameters,
) -> AdaptationAlgorithm:

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
