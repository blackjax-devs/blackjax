"""Metropolis-adjusted discrete Langevin sampling on binary PyTrees.

The proposal follows equations (2) and (3) of Zhang, Liu and Liu (2022),
https://proceedings.mlr.press/v162/zhang22t.html. Only the binary domain is
implemented; categorical, unadjusted and preconditioned variants are not.
"""

import operator
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from blackjax.types import Array, ArrayLikeTree, ArrayTree, Numeric, PRNGKey

__all__ = ["DMALAState", "DMALAInfo", "init", "build_kernel", "as_top_level_api"]


class DMALAState(NamedTuple):
    """Binary position and cached log density and gradient of its extension."""

    position: ArrayTree
    logdensity: Array
    logdensity_grad: ArrayTree


class DMALAInfo(NamedTuple):
    """Acceptance probability, acceptance decision and proposed chain state."""

    acceptance_rate: Array
    is_accepted: Array
    proposal: DMALAState


def init(
    position: ArrayLikeTree, logdensity_fn: Callable[[ArrayTree], Numeric]
) -> DMALAState:
    """Initialise a binary chain, converting positions to floating-point arrays.

    Parameters
    ----------
    position
        PyTree with entries equal to zero or one. Floating-point representation
        enables differentiation of the log-density extension.
    logdensity_fn
        Scalar differentiable extension of the target log probability to real
        inputs. Its values on the binary domain define the target distribution.

    Returns
    -------
    state
        Position, log density and gradient cached for the first transition.
    """
    array_position = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=jnp.result_type(x, jnp.float32)), position
    )
    logdensity, gradient = jax.value_and_grad(logdensity_fn)(array_position)
    return DMALAState(array_position, jnp.asarray(logdensity), gradient)


def _flip_logits(
    position: ArrayTree, gradient: ArrayTree, step_size: Numeric
) -> ArrayTree:
    return jax.tree.map(
        lambda x, g: (
            0.5 * g * (1 - 2 * x) - 0.5 / jnp.asarray(step_size, dtype=x.dtype)
        ),
        position,
        gradient,
    )


def _proposal_logprob(
    source: DMALAState, destination: DMALAState, step_size: Numeric
) -> Array:
    logits = _flip_logits(source.position, source.logdensity_grad, step_size)
    terms = jax.tree.map(
        lambda x, y, logit: jnp.sum(
            jnp.where(x != y, jax.nn.log_sigmoid(logit), jax.nn.log_sigmoid(-logit))
        ),
        source.position,
        destination.position,
        logits,
    )
    return jax.tree.reduce(operator.add, terms)


def build_kernel() -> Callable:
    """Build a Metropolis-adjusted binary Langevin transition.

    Returns
    -------
    kernel
        Callable taking a random key, state, log-density extension and positive
        finite scalar step size, and returning a state and ``DMALAInfo``.
    """

    def kernel(
        rng_key: PRNGKey,
        state: DMALAState,
        logdensity_fn: Callable[[ArrayTree], Numeric],
        step_size: Numeric,
    ) -> tuple[DMALAState, DMALAInfo]:
        proposal_key, accept_key = jax.random.split(rng_key)
        leaves, treedef = jax.tree.flatten(state.position)
        logits = jax.tree.leaves(
            _flip_logits(state.position, state.logdensity_grad, step_size)
        )
        keys = jax.random.split(proposal_key, len(leaves))
        proposed_position = jax.tree.unflatten(
            treedef,
            [
                jnp.where(jax.random.bernoulli(key, jax.nn.sigmoid(logit)), 1 - x, x)
                for key, x, logit in zip(keys, leaves, logits)
            ],
        )
        proposed_state = init(proposed_position, logdensity_fn)
        log_forward = _proposal_logprob(state, proposed_state, step_size)
        log_reverse = _proposal_logprob(proposed_state, state, step_size)
        log_accept = safe_energy_diff(
            proposed_state.logdensity + log_reverse, state.logdensity + log_forward
        )
        new_state, (accepted, probability, _) = static_binomial_sampling(
            accept_key, log_accept, state, proposed_state
        )
        return new_state, DMALAInfo(probability, accepted, proposed_state)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable[[ArrayTree], Numeric], step_size: Numeric
) -> SamplingAlgorithm:
    """Construct the binary discrete Metropolis-adjusted Langevin algorithm.

    The chain's positions must have entries zero or one. The log-density
    function must have a scalar differentiable extension to real inputs;
    gradients of this extension guide the proposal. Metropolis correction
    uses its values on binary positions. The extension need not be normalised.

    Parameters
    ----------
    logdensity_fn
        Differentiable scalar extension of the target log probability.
    step_size
        Positive finite scalar alpha in the paper's proposal: a bit at x flips
        with log odds ``gradient * (1 - 2*x) / 2 - 1 / (2*alpha)``.

    Returns
    -------
    algorithm
        ``SamplingAlgorithm`` with ``init`` and ``step`` functions. These are
        compatible with ``jax.jit`` and ``jax.vmap``.
    """
    return build_sampling_algorithm(
        build_kernel(), init, logdensity_fn, kernel_args=(step_size,)
    )
