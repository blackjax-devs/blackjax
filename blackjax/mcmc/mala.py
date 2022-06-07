"""Public API for Metropolis Adjusted Langevin kernels."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.mcmc.diffusion import overdamped_langevin
from blackjax.types import PRNGKey, PyTree

__all__ = ["MALAState", "MALAInfo", "init", "kernel"]


class MALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: PyTree
    logprob: float
    logprob_grad: PyTree


class MALAInfo(NamedTuple):
    """Additional information on the MALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_probability: float
    is_accepted: bool


def init(position: PyTree, logprob_fn: Callable) -> MALAState:
    grad_fn = jax.value_and_grad(logprob_fn)
    logprob, logprob_grad = grad_fn(position)
    return MALAState(position, logprob, logprob_grad)


def kernel():
    """Build a MALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def transition_probability(state, new_state, step_size):
        """Transition probability to go from `state` to `new_state`"""
        theta = jax.tree_util.tree_map(
            lambda new_x, x, g: new_x - x - step_size * g,
            new_state.position,
            state.position,
            state.logprob_grad,
        )
        theta_ravel, _ = ravel_pytree(theta)
        return -0.25 * (1.0 / step_size) * jnp.dot(theta_ravel, theta_ravel)

    def one_step(
        rng_key: PRNGKey, state: MALAState, logprob_fn: Callable, step_size: float
    ) -> Tuple[MALAState, MALAInfo]:
        """Generate a new sample with the MALA kernel.

        TODO expand the docstring.

        """
        grad_fn = jax.value_and_grad(logprob_fn)
        integrator = overdamped_langevin(grad_fn)

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)

        delta = (
            new_state.logprob
            - state.logprob
            + transition_probability(new_state, state, step_size)
            - transition_probability(state, new_state, step_size)
        )
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1)

        do_accept = jax.random.bernoulli(key_rmh, p_accept)

        new_state = MALAState(*new_state)
        info = MALAInfo(p_accept, do_accept)

        return jax.lax.cond(
            do_accept,
            lambda _: (new_state, info),
            lambda _: (state, info),
            operand=None,
        )

    return one_step
