"""Independent draws by rejection sampling with a user-supplied envelope."""

import operator
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayLikeTree, ArrayTree, Numeric, PRNGKey

__all__ = ["RejectionInfo", "build_kernel"]


class RejectionInfo(NamedTuple):
    """Outcome of one rejection-sampling run.

    ``is_accepted`` indicates whether the returned value is a target draw. If it
    is false, the returned value is only the last proposal and must not be used
    as a sample. ``num_proposals`` counts all proposals evaluated.
    ``is_bound_valid`` is false if an evaluated proposal violates the envelope
    or has an undefined density ratio. A true value does not establish the
    envelope globally; the caller must supply a valid bound.
    """

    is_accepted: Array
    num_proposals: Array
    is_bound_valid: Array


class _RejectionState(NamedTuple):
    rng_key: PRNGKey
    candidate: ArrayTree
    info: RejectionInfo


def build_kernel(
    logdensity_fn: Callable[[ArrayTree], Numeric],
    proposal_sampler: Callable[[PRNGKey], ArrayLikeTree],
    proposal_logdensity_fn: Callable[[ArrayTree], Numeric],
    log_bound: Numeric,
    *,
    max_steps: int = 1000,
) -> Callable[[PRNGKey], tuple[ArrayTree, RejectionInfo]]:
    """Build a kernel that returns one independent target draw and its outcome.

    The proposal sampler must draw from the *normalised* density ``q`` described
    by ``proposal_logdensity_fn``. The target density ``f`` may be unnormalised.
    Supply a finite ``log_bound = log(M)`` satisfying ``f(x) <= M * q(x)``
    everywhere, with the proposal supporting the entire target distribution.
    No bound is estimated from the evaluated proposals.

    Each proposal is accepted with probability ``f(x) / (M * q(x))``. Proposals
    with zero target density are rejected. If a sampled ratio exceeds one, or
    is undefined, the run stops and reports an invalid bound rather than
    silently clipping the probability. Densities must return scalar values.

    ``max_steps`` is a positive static integer no greater than ``2**31 - 1``
    limiting the number of proposals.
    Exhausting it reports ``is_accepted=False``. The caller must check the
    returned information before using the last proposal. The kernel supports
    ``jax.jit`` and ``jax.vmap``. All supplied functions must be JAX-compatible,
    including for uncompiled calls, because the loop body is traced. Proposal PyTrees
    must have a fixed structure, shape, and dtype across calls. Independent
    random keys give independent draws. Rejection sampling can be inefficient
    when the envelope is loose, particularly in high dimensions.

    Parameters
    ----------
    logdensity_fn
        Scalar log density of the target, possibly unnormalised.
    proposal_sampler
        Draws a proposal PyTree from a random key.
    proposal_logdensity_fn
        Scalar log density of the normalised proposal distribution.
    log_bound
        Finite scalar logarithm of a known global envelope constant.
    max_steps
        Static proposal budget, between 1 and ``2**31 - 1`` inclusive.

    Returns
    -------
    kernel
        A random-key callable returning the last proposal and ``RejectionInfo``.
        The proposal is a target draw only when ``info.is_accepted`` is true.

    Examples
    --------
    Sample a Beta(2, 1) target using a uniform proposal on [0, 1)::

        kernel = build_kernel(
            lambda x: jnp.where((x >= 0) & (x < 1), jnp.log(2 * x), -jnp.inf),
            lambda key: jax.random.uniform(key),
            lambda x: jnp.where((x >= 0) & (x < 1), 0.0, -jnp.inf),
            jnp.log(2.0),
        )
        keys = jax.random.split(jax.random.key(0), 100)
        samples, info = jax.jit(jax.vmap(kernel))(keys)
        # Check info.is_accepted and info.is_bound_valid before using samples.
    """
    max_steps = operator.index(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_steps > 2**31 - 1:
        raise ValueError("max_steps must not exceed 2**31 - 1")
    if jnp.ndim(log_bound) != 0:
        raise ValueError("log_bound must be a scalar")

    def propose(rng_key: PRNGKey, num_proposals: Array) -> _RejectionState:
        next_key, proposal_key, accept_key = jax.random.split(rng_key, 3)
        candidate = jax.tree.map(jnp.asarray, proposal_sampler(proposal_key))
        log_f = logdensity_fn(candidate)
        log_q = proposal_logdensity_fn(candidate)
        log_ratio = jnp.asarray(
            log_f - log_q - log_bound,
            dtype=jnp.result_type(log_f, log_q, log_bound, float),
        )
        if log_ratio.ndim != 0:
            raise ValueError("target and proposal log densities must be scalars")
        is_valid = (
            jnp.isfinite(log_bound)
            & jnp.isfinite(log_q)
            & ~jnp.isnan(log_ratio)
            & (log_ratio <= 0)
        )
        log_uniform = jnp.log(jax.random.uniform(accept_key, dtype=log_ratio.dtype))
        is_accepted = is_valid & (log_uniform < log_ratio)
        info = RejectionInfo(is_accepted, num_proposals + 1, is_valid)
        return _RejectionState(next_key, candidate, info)

    def kernel(rng_key: PRNGKey) -> tuple[ArrayTree, RejectionInfo]:
        def continue_sampling(state: _RejectionState) -> Array:
            _, _, info = state
            return (
                ~info.is_accepted
                & info.is_bound_valid
                & (info.num_proposals < max_steps)
            )

        def step(state: _RejectionState) -> _RejectionState:
            key, _, info = state
            return propose(key, info.num_proposals)

        initial_state = propose(rng_key, jnp.array(0, dtype=jnp.int32))
        _, sample, info = jax.lax.while_loop(continue_sampling, step, initial_state)
        return sample, info

    return kernel
