# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NS particle-update strategies that wrap a generic MCMC kernel under the
likelihood constraint."""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.base import delete_fn as default_delete_fn

__all__ = [
    "MCMCUpdateInfo",
    "ConstrainedMCMCInfo",
    "update_with_mcmc_take_last",
    "reject_constrained_step",
    "build_kernel",
]


class MCMCUpdateInfo(NamedTuple):
    """Thin layer to hold all the info pertaining to the update step."""

    mcmc_states: NamedTuple
    mcmc_infos: NamedTuple


class ConstrainedMCMCInfo(NamedTuple):
    """Info for a constrained MCMC proposal.

    Attributes
    ----------
    info
        The underlying MCMC info (e.g., RWInfo for random walk).
    is_accepted
        True if both the MCMC proposal was accepted and the proposed
        point is above the likelihood threshold.
    """

    info: NamedTuple
    is_accepted: jnp.ndarray


def update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
    num_delete,
):
    """An update strategy for NS that uses MCMC to update the particles.
    For now we will not keep the states as they will be too large to store.
    Similar to the update_and_take_last from SMC.

    Parameters
    ----------
    constrained_mcmc_step_fn
        Wrapped MCMC step function that enforces the NS likelihood constraint.
    num_mcmc_steps
        Number of MCMC proposals per particle.
    num_delete
        Number of particles to replace per step.

    Returns
    -------
    An update function that proposes new particles by running the constrained
    MCMC kernel from survivor start points and returns the final states and
    infos.
    """

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, sample_key = random.split(rng_key)
        particles = state.particles

        # Select start particles from survivors
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = random.choice(
            choice_key,
            len(weights),
            shape=(num_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_state = jax.tree.map(lambda x: x[start_idx], particles)

        shared_mcmc_step_fn = partial(
            constrained_mcmc_step_fn,
            loglikelihood_0=loglikelihood_0,
            **step_parameters,
        )

        def mcmc_kernel(rng_key, state):
            keys = random.split(rng_key, num_mcmc_steps)

            def body_fn(state, rng_key):
                new_state, info = shared_mcmc_step_fn(rng_key, state)
                return new_state, info

            final_state, infos = jax.lax.scan(body_fn, state, keys)
            return final_state, infos

        sample_keys = random.split(sample_key, num_delete)
        return jax.vmap(mcmc_kernel)(sample_keys, start_state)

    return update_function


def reject_constrained_step(
    init_state_fn: Callable,
    logdensity_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_step_fn: Callable,
) -> Callable:
    """Constrained inner step wrapping a generic MCMC kernel (propose-then-reject).

    Proposes one ``mcmc_step_fn`` move and accepts it only if the MCMC step
    accepted AND the proposed point is above the likelihood threshold; otherwise
    the particle stays put. The complement to :func:`slice_constrained_step` for
    kernels that cannot gate the constraint inside their own proposal.
    """

    def step(rng_key, state, loglikelihood_0, **params):
        mcmc_state = mcmc_init_fn(state.position, logdensity_fn)
        new_mcmc_state, mcmc_info = mcmc_step_fn(
            rng_key, mcmc_state, logdensity_fn, **params
        )
        proposed_state = init_state_fn(
            new_mcmc_state.position, loglikelihood_birth=loglikelihood_0
        )
        within_contour = proposed_state.loglikelihood > loglikelihood_0
        proposal_accepted = getattr(mcmc_info, "is_accepted", True)
        is_accepted = proposal_accepted & within_contour
        new_state = jax.lax.cond(
            is_accepted,
            lambda: proposed_state,
            lambda: state,
        )
        return new_state, ConstrainedMCMCInfo(mcmc_info, is_accepted)

    return step


def build_kernel(
    constrained_step_fn: Callable,
    num_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    num_delete: int = 1,
    delete_fn: Callable = default_delete_fn,
) -> Callable:
    """Build a Nested Sampling kernel from a constrained inner step.

    The generic NS engine: run ``constrained_step_fn`` (a move that reports its
    in-contour ``is_valid``) for ``num_inner_steps`` from survivor start points,
    take the last, and accumulate the evidence via the adaptive kernel. Build the
    step with :func:`reject_constrained_step` (generic MCMC) or, for the slice
    family, :func:`~blackjax.ns.nss.slice_constrained_step`.

    Parameters
    ----------
    constrained_step_fn
        Constrained inner step ``(rng_key, state, loglikelihood_0, **params) ->
        (new_state, info)``.
    num_inner_steps
        Number of inner steps per particle replacement.
    update_inner_kernel_params_fn
        Recomputes the inner-kernel parameters from the live points each step.
    num_delete
        Number of particles replaced per NS iteration.
    delete_fn
        Selects which particles to delete (default: the lowest-likelihood ones).

    Returns
    -------
    A Nested Sampling kernel ``kernel(rng_key, state) -> (new_state, info)``.
    """
    inner_kernel = update_with_mcmc_take_last(
        constrained_step_fn, num_inner_steps, num_delete
    )
    delete_fn = partial(delete_fn, num_delete=num_delete)
    return build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
