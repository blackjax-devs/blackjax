import functools
from typing import Callable, Union

import jax

from blackjax import nuts, hmc
from blackjax.base import AdaptationAlgorithm
from blackjax.hmc_base import HMCState
from blackjax.stan_warmup import window_adaptation_base, window_adaptation_schedule
from blackjax.types import PRNGKey, Array


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logprob_fn: Callable,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
    **parameters,
) -> AdaptationAlgorithm:

    kernel = algorithm.kernel_gen()

    def kernel_factory(step_size: float, inverse_mass_matrix: Array):
        return jax.jit(
            functools.partial(
                kernel,
                logprob_fn=logprob_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                **parameters,
            ),
            static_argnames=["logprob_fn"],
        )

    def run(rng_key: PRNGKey, init_state: HMCState, num_steps: int = 1000):

        schedule_fn = window_adaptation_schedule(num_steps)
        init, update, final = window_adaptation_base(
            kernel_factory,
            schedule_fn,
            is_mass_matrix_diagonal,
            target_acceptance_rate=target_acceptance_rate,
        )

        def one_step(carry, rng_key):
            state, warmup_state = carry
            state, warmup_state, info = update(rng_key, state, warmup_state)
            return ((state, warmup_state), (state, warmup_state, info))

        warmup_state = init(rng_key, init_state, initial_step_size)
        keys = jax.random.split(rng_key, num_steps + 1)[1:]
        last_state, warmup_chain = jax.lax.scan(
            one_step,
            (init_state, warmup_state),
            keys,
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = final(last_warmup_state)

        return last_chain_state, (step_size, inverse_mass_matrix), warmup_chain

    return AdaptationAlgorithm(run)
