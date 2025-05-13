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
import functools
from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import default_generate_slice_direction_fn as ss_default_generate_slice_direction_fn
from blackjax.mcmc.ss import default_stepper_fn
from blackjax.mcmc.ss import init as slice_init
from blackjax.ns.adaptive import build_kernel, init
from blackjax.ns.base import NSInfo, NSState, delete_fn
from blackjax.ns.utils import get_first_row
from blackjax.smc.tuning.from_particles import (
    particles_as_rows,
    particles_covariance_matrix,
)
from blackjax.types import ArrayLikeTree, PRNGKey, ArrayTree
from jax.flatten_util import ravel_pytree

__all__ = ["init", "as_top_level_api"]


def default_generate_slice_direction_fn(rng_key: PRNGKey, **kernel_args: ArrayTree) -> ArrayTree:
    """Default function to generate a normalized slice direction.

    This function generates a random direction for the Hit-and-Run Slice Sampler.
    It samples from a zero-mean multivariate Gaussian distribution with the provided
    covariance matrix (`cov` in `kernel_args`) and then normalizes the direction
    with respect to the Mahalanobis norm defined by `inv(cov)`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    **kernel_args
        Keyword arguments, expected to contain `cov`: the covariance matrix (PyTree
        where each leaf is a row (or column) of the covariance matrix). This is 
        used for sampling the initial direction. The structure of `cov` should
        match the particle structure after `particles_as_rows` and then unravelling
        the first row.

    Returns
    -------
    ArrayTree
        A normalized direction vector (PyTree, matching the structure of a single particle),
        to be used by the slice sampler.
    """
    cov = kernel_args["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov)
    d = ss_default_generate_slice_direction_fn(rng_key, cov)
    return unravel_fn(d)


def default_adapt_direction_params_fn(state: NSState, info: NSInfo) -> Dict[str, ArrayTree]:
    """Default function to adapt/tune the slice direction proposal parameters.

    This function computes the empirical covariance matrix from the current set of
    live particles in `state.particles`. This covariance matrix is then returned
    and can be used by the slice direction generation function (e.g.,
    `default_generate_slice_direction_fn`) in the next Nested Sampling iteration.

    Parameters
    ----------
    state
        The current `NSState` of the Nested Sampler, containing the live particles.
    info
        The `NSInfo` from the last Nested Sampling step (currently unused by this function).

    Returns
    -------
    Dict[str, ArrayTree]
        A dictionary containing the adapted parameters. Specifically, it returns
        `{'cov': cov}` where `cov`
        is a PyTree. Each leaf of this PyTree is a row (or column) of the
        covariance matrix (2D Array) corresponding to a flattened component of
        the particles.
        If particles are simple N-D arrays, `cov` will be a single 2D array.
        If particles are PyTrees, `cov` will be a PyTree where each leaf is the
        covariance of that leaf across particles, reshaped appropriately.
    """
    cov = particles_covariance_matrix(state.particles)
    single_particle = get_first_row(state.particles)
    _, unravel_fn = ravel_pytree(single_particle)
    return {"cov": jax.vmap(unravel_fn)(cov)}


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_mcmc_steps: int,
    n_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = default_adapt_direction_params_fn,
    generate_slice_direction_fn: Callable = default_generate_slice_direction_fn,
) -> SamplingAlgorithm:
    """Creates an adaptive Nested Slice Sampling (NSS) algorithm.

    This function configures a Nested Sampling algorithm that uses Hit-and-Run
    Slice Sampling (HRSS) as its inner MCMC kernel. The parameters for the HRSS
    direction proposal (specifically, the covariance matrix) are adaptively tuned
    at each step using `adapt_direction_params_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_mcmc_steps
        The number of HRSS steps to run for each new particle generation.
        The paper suggests this is `p`, e.g., `3 * d` where `d` is dimension.
    n_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> x_new` for the HRSS kernel.
        Defaults to `default_stepper`.
    adapt_direction_params_fn
        A function `(ns_state, ns_info) -> dict_of_params` that computes/adapts
        the parameters (e.g., covariance matrix) for the slice direction proposal,
        based on the current NS state. Defaults to `default_train_fn`.
    generate_slice_direction_fn
        A function `(rng_key, **params) -> direction_pytree` that generates a
        normalized direction for HRSS, using parameters from `adapt_direction_params_fn`.
        Defaults to `default_generate_slice_direction_fn`.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Slice Sampler. The state managed by this
        algorithm is `StateWithParameterOverride`.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    def mcmc_build_kernel(**kwargs):
        return build_slice_kernel(partial(generate_slice_direction_fn, **kwargs), stepper_fn)

    mcmc_init_fn = slice_init
    mcmc_parameter_update_fn = adapt_direction_params_fn

    def init_fn(particles: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(particles, loglikelihood_fn, logprior_fn, mcmc_parameter_update_fn)

    step_fn = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_build_kernel,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
    )

    return SamplingAlgorithm(init_fn, step_fn)
