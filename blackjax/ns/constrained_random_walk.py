from typing import Callable, NamedTuple

import jax
from blackjax.mcmc.random_walk import build_additive_step
from blackjax.mcmc.random_walk import init as init_rw
from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "build_constrained_random_walk",
]


class CRWState(NamedTuple):
    position: ArrayTree
    logdensity: ArrayTree
    loglikelihood: ArrayTree


class CRWInfo(NamedTuple):
    evals: Array


def init(particles: ArrayTree, logdensity_fn: Callable, loglikelihood: Array):
    logdensity = logdensity_fn(particles)
    return CRWState(particles, logdensity, loglikelihood)


def build_constrained_random_walk(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    logL0: Array,
    proposal_distribution: Callable,
) -> Callable:
    unconstrained_kernel = build_additive_step()

    def kernel(rng_key: PRNGKey, state: CRWState):
        def unconstrained_step(rng_key, state):
            return unconstrained_kernel(
                rng_key,
                state,
                logprior_fn,
                proposal_distribution,
            )

        rng_key, step_key = jax.random.split(rng_key)

        def body_fn(carry):
            state, key, is_accepted, logl, counter, _ = carry
            counter += 1
            key, proposal_key = jax.random.split(key)
            new_state, info = unconstrained_step(proposal_key, state)
            logl = loglikelihood_fn(new_state.position)
            return state, key, info.is_accepted, logl, counter, new_state

        def cond_fn(carry):
            _, _, is_accepted, logl, _, _ = carry
            return ~(is_accepted & (logl > logL0))

        walk_state = init_rw(state.position, logprior_fn)
        new_walk_state = init_rw(state.position, logprior_fn)
        carry = (walk_state, step_key, False, state.loglikelihood, 0, new_walk_state)
        _, _, _, logl, info, new_walk_state = jax.lax.while_loop(
            cond_fn, body_fn, carry
        )
        return CRWState(
            new_walk_state.position, new_walk_state.logdensity, logl
        ), CRWInfo(info)

    return kernel
