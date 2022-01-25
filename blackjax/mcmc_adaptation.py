import functools
from typing import Callable, Union

import jax
import jax.numpy as jnp

from blackjax import nuts, hmc
from blackjax.base import AdaptationAlgorithm
from blackjax.hmc_base import HMCState
from blackjax.stan_warmup import stan_warmup, stan_warmup_schedule
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
        init, update, final = stan_warmup(
            kernel_factory,
            is_mass_matrix_diagonal,
            target_acceptance_rate=target_acceptance_rate,
        )

        def one_step(carry, interval):
            rng_key, state, warmup_state = carry
            stage, is_middle_window_end = interval

            _, rng_key = jax.random.split(rng_key)
            state, warmup_state, info = update(
                rng_key, stage, is_middle_window_end, state, warmup_state
            )

            return ((rng_key, state, warmup_state), (state, warmup_state, info))

        schedule = jnp.array(stan_warmup_schedule(num_steps))

        warmup_state = init(rng_key, init_state, initial_step_size)
        last_state, warmup_chain = jax.lax.scan(
            one_step, (rng_key, init_state, warmup_state), schedule
        )
        _, last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = final(last_warmup_state)

        return last_chain_state, (step_size, inverse_mass_matrix), warmup_chain

    return AdaptationAlgorithm(run)
