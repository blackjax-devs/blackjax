from typing import Callable, Tuple

import jax
from jax import numpy as jnp

from blackjax.mcmc.rmh import RWInfo, RWState, rmh
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["kernel", "normal"]


def kernel(random_step):
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel with a gaussian
    proposal distribution.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(
        rng_key: PRNGKey, state: RWState, logdensity_fn: Callable
    ) -> Tuple[RWState, RWInfo]:
        def proposal_generator(key_proposal, position):
            move_proposal = random_step(key_proposal, position)
            new_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
            return new_position

        kernel = rmh(logdensity_fn, proposal_generator)
        return kernel(rng_key, state)

    return one_step


def normal(sigma: Array) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    Parameter
    ---------
    sigma:
        vector or matrix that contains the standard deviation of the centered
        normal distribution from which we draw the move proposals.

    """
    if jnp.ndim(sigma) > 2:
        raise ValueError("sigma must be a vector or a matrix.")

    def propose(rng_key: PRNGKey, position: PyTree) -> PyTree:
        return generate_gaussian_noise(rng_key, position, sigma=sigma)

    return propose
