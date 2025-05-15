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
"""Nested Slice Sampling (NSS) algorithm.

This module implements the Nested Slice Sampling algorithm, which combines the
Nested Sampling framework with an inner Hit-and-Run Slice Sampling (HRSS) kernel
for exploring the constrained prior distribution at each likelihood level.

The key idea is to leverage the efficiency of slice sampling for constrained
sampling tasks. The parameters of the HRSS kernel, specifically the covariance
matrix for proposing slice directions, are adaptively tuned based on the current
set of live particles.
"""
from functools import partial
from typing import Callable, Dict, Tuple, Any, Optional

import jax
from jax.flatten_util import ravel_pytree

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import (
    default_generate_slice_direction_fn as ss_default_generate_slice_direction_fn,
    build_kernel as build_slice_kernel
)
from blackjax.mcmc.ss import default_stepper_fn
from blackjax.mcmc.ss import init as slice_init
from blackjax.mcmc.ss import SliceState, SliceInfo
from blackjax.ns.adaptive import build_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.utils import get_first_row, repeat_kernel
from blackjax.smc.tuning.from_particles import (
    particles_as_rows,
    particles_covariance_matrix,
)
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride

__all__ = [
    "init",
    "as_top_level_api",
    "default_generate_slice_direction_fn",
    "default_adapt_direction_params_fn",
]


def default_generate_slice_direction_fn(
    rng_key: PRNGKey, **kernel_args: Any
) -> ArrayTree:
    """Default function to generate a normalized slice direction for NSS.

    This function is designed to work with covariance parameters adapted by
    `default_adapt_direction_params_fn`. It expects `kernel_args` to contain
    a key 'cov', where `kernel_args['cov']` is a PyTree structured identically
    to a single particle. Each leaf of this 'cov' PyTree should represent the
    rows of the full, flattened covariance matrix that correspond to that leaf's
    elements in the flattened particle vector.
    (Specifically, if the full DxD covariance matrix of flattened particles is
    `M_flat`, and `unravel_fn` un-flattens a D-vector to the particle PyTree,
    then the input `cov = kernel_args['cov']` is effectively
    `jax.vmap(unravel_fn)(M_flat)`).

    The function reassembles the full (D,D) covariance matrix from this
    PyTree structure. It then samples a flat direction vector `d_flat` from
    a multivariate Gaussian N(0, M_reassembled), normalizes `d_flat` using
    the Mahalanobis norm defined by M_reassembled_inv, and finally un-flattens
    this normalized direction back into the particle's PyTree structure.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    **kernel_args
        Keyword arguments, must contain:
        - `cov`: An `ArrayTree` (structured like a particle) whose leaves
                 represent rows of the covariance matrix, typically the output
                 of `default_adapt_direction_params_fn`.

    Returns
    -------
    ArrayTree
        A Mahalanobis-normalized direction vector (PyTree, matching the
        structure of a single particle), to be used by the slice sampler.
    """
    cov = kernel_args["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov)
    d = ss_default_generate_slice_direction_fn(rng_key, cov)
    return unravel_fn(d)


def default_adapt_direction_params_fn(
    state: NSState, info: Optional[NSInfo]  # info is None at initialization
) -> Dict[str, ArrayTree]:
    """Default function to adapt/tune the slice direction proposal parameters.

    This function computes the empirical covariance matrix from the current set of
    live particles in `state.particles`. This covariance matrix, structured as a
    PyTree, is then returned and can be used by the slice direction generation
    function (e.g., `default_generate_slice_direction_fn`) in the next
    Nested Sampling iteration.

    Parameters
    ----------
    state
        The current `NSState` of the Nested Sampler, containing the live particles.
    info
        The `NSInfo` from the last Nested Sampling step. Unused by this function,
        and will be `None` when called during initialization.

    Returns
    -------
    Dict[str, ArrayTree]
        A dictionary `{'cov': cov_pytree}`. `cov_pytree` is an `ArrayTree` with the
        same structure as a single particle. If the full DxD covariance matrix
        of the flattened particles is `M_flat` (a 2D Array), and `unravel_fn` is the
        function to un-flatten a D-vector to the particle's PyTree structure, then
        `cov_pytree` is equivalent to `jax.vmap(unravel_fn)(M_flat)`.
        This means each leaf of `cov_pytree` will have a shape `(D, *leaf_original_dims)`.
    """
    cov = particles_covariance_matrix(state.particles)
    single_particle = get_first_row(state.particles)
    _, unravel_fn = ravel_pytree(single_particle)
    return {"cov": jax.vmap(unravel_fn)(cov)}


def as_top_level_api(
    logprior_fn: Callable[[ArrayTree], float],
    loglikelihood_fn: Callable[[ArrayTree], float],
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable[[ArrayTree, ArrayTree, float], ArrayTree] = default_stepper_fn,
    adapt_direction_params_fn: Callable[
        [NSState, Optional[NSInfo]], Dict[str, ArrayTree]
    ] = default_adapt_direction_params_fn,
    generate_slice_direction_fn: Callable[
        [PRNGKey, Any], ArrayTree
    ] = default_generate_slice_direction_fn,
) -> SamplingAlgorithm[StateWithParameterOverride, NSInfo, Any]:
    """Creates an adaptive Nested Slice Sampling (NSS) algorithm.

    This function configures a Nested Sampling algorithm that uses Hit-and-Run
    Slice Sampling (HRSS) as its inner kernel. The parameters for the HRSS
    direction proposal (specifically, the covariance matrix structured as a PyTree)
    are adaptively tuned at each step using `adapt_direction_params_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_inner_steps
        The number of HRSS steps to run for each new particle generation.
        A common heuristic is a multiple of the parameter space dimension (e.g., 3*dim).
    num_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> x_new` for the HRSS kernel.
        Defaults to `blackjax.mcmc.ss.default_stepper_fn`.
    adapt_direction_params_fn
        A function `(ns_state, ns_info_or_none) -> dict_of_params` that computes/adapts
        the parameters (e.g., covariance matrix as a PyTree) for the slice
        direction proposal, based on the current NS state. `ns_info_or_none`
        is `None` at initialization. Defaults to `default_adapt_direction_params_fn`.
    generate_slice_direction_fn
        A function `(rng_key, **params) -> direction_pytree` that generates a
        normalized direction for HRSS, using parameters (e.g. `params['cov']`)
        from `adapt_direction_params_fn`.
        Defaults to `default_generate_slice_direction_fn`.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Slice Sampler. The state managed by this
        algorithm is `StateWithParameterOverride[NSState, Dict[str, ArrayTree]]`.
    """
    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    @repeat_kernel(num_inner_steps)
    def inner_kernel(rng_key, state, logdensity_fn, **kwargs):
        generate_slice_direction_fn_ = partial(generate_slice_direction_fn, **kwargs)
        slice_kernel = build_slice_kernel(generate_slice_direction_fn_, stepper_fn)
        return slice_kernel(rng_key, state, logdensity_fn)

    inner_init_fn = slice_init
    update_inner_kernel = adapt_direction_params_fn

    def init_fn(particles: ArrayLikeTree, rng_key: PRNGKey = None) -> NSState:
        del rng_key
        return init(particles, loglikelihood_fn, logprior_fn, update_inner_kernel)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_init_fn,
        inner_kernel,
        update_inner_kernel,
    )

    def step_fn(
        rng_key: PRNGKey,
        state: StateWithParameterOverride[NSState, Dict[str, ArrayTree]],
    ) -> Tuple[StateWithParameterOverride[NSState, Dict[str, ArrayTree]], NSInfo]:
        """Performs one step of the adaptive Nested Slice Sampling algorithm."""
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore
