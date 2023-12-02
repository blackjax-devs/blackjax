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
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L.

"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.diagnostics import effective_sample_size  # type: ignore
from blackjax.util import pytree_size
from jax.flatten_util import ravel_pytree



class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    Attributes:
        L (float): The momentum decoherent rate for the MCLMC algorithm.
        step_size (float): The step size used for the MCLMC algorithm.
    """

    L: float
    step_size: float


def mclmc_find_L_and_step_size(
    kernel,
    num_steps,
    state,
    part1_key,
    part2_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
):
    """
    Finds the optimal value of L (step size) for the MCLMC algorithm.

    Args:
        kernel: The kernel function used for the MCMC algorithm.
        num_steps: The number of MCMC steps that will subsequently be run, after tuning
        state: The initial state of the MCMC algorithm.
        frac_tune1: The fraction of tuning for the first step of the adaptation.
        frac_tune2: The fraction of tuning for the second step of the adaptation.
        frac_tune3: The fraction of tuning for the third step of the adaptation.

    Returns:
        state: The final state of the MCMC algorithm.
        params: The final hyperparameters of the MCMC algorithm.
    """
    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(jnp.sqrt(dim), jnp.sqrt(dim) * 0.25)
    varEwanted = 5e-4

    state, params = make_L_step_size_adaptation(
        kernel=kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        varEwanted=varEwanted,
        sigma_xi=1.5,
        num_effective_samples=150,
    )(state, params, num_steps, part1_key)

    if frac_tune3 != 0:
        state, params = make_adaptation_L(kernel, frac=frac_tune3, Lfactor=0.4)(
            state, params, num_steps, part2_key
        )

    return state, params


def make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    varEwanted=1e-3,
    sigma_xi=1.5,
    num_effective_samples=150,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""

    gamma_forget = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(state_old, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        W, F, step_size_max = adaptive_state

        # dynamics
        state_new, info = kernel(
            rng_key=rng_key, state=state_old, L=params.L, step_size=params.step_size
        )
        energy_change = info.dE
        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            state_old, state_new, params.step_size, step_size_max, energy_change
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (dim * varEwanted)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma_forget * F + w * (xi / jnp.power(params.step_size, 6.0))
        W = gamma_forget * W + w
        step_size = jnp.power(
            F / W, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        return state, params_new, params_new, (W, F, step_size_max), success

    def update_kalman(x, state, outer_weight, success, step_size):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * step_size * success
        zero_prevention = 1 - outer_weight
        F1 = (W * F1 + w * x) / (
            W + w + zero_prevention
        )  # Update <f(x)> with a Kalman filter
        F2 = (W * F2 + w * jnp.square(x)) / (
            W + w + zero_prevention
        )  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)

    adap0 = (0.0, 0.0, jnp.inf)

    def step(iteration_state, weight_and_key):
        outer_weight, rng_key = weight_and_key
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""
        state, params, adaptive_state, kalman_state = iteration_state
        state, params, params_final, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )
        position, _ = ravel_pytree(state.position)
        kalman_state = update_kalman(
            position, kalman_state, outer_weight, success, params.step_size
        )

        return (state, params_final, adaptive_state, kalman_state), None

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = int(num_steps * frac_tune1), int(
            num_steps * frac_tune2
        )
        # L_step_size_adaptation_keys = jax.random.split(rng_key, num_steps1 + num_steps2)
        L_step_size_adaptation_keys = jnp.array([rng_key] * (num_steps1 + num_steps2))

        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # initial state of the kalman filter
        kalman_state = (0.0, jnp.zeros(dim), jnp.zeros(dim))

        # run the steps
        kalman_state = jax.lax.scan(
            step,
            init=(state, params, adap0, kalman_state),
            xs=(outer_weights, L_step_size_adaptation_keys),
            length=num_steps1 + num_steps2,
        )[0]
        state, params, _, kalman_state_output = kalman_state

        L = params.L
        # determine L
        if num_steps2 != 0.0:
            _, F1, F2 = kalman_state_output
            variances = F2 - jnp.square(F1)
            L = jnp.sqrt(jnp.sum(variances))

        return state, MCLMCAdaptationState(L, params.step_size)

    return L_step_size_adaptation


def make_adaptation_L(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        # adaptation_L_keys = jax.random.split(key, num_steps)
        adaptation_L_keys = jnp.array([key] * (num_steps))

        # run kernel in the normal way
        state, info = jax.lax.scan(
            f=lambda s, k: (
                kernel(rng_key=k, state=s, L=params.L, step_size=params.step_size)
            ),
            init=state,
            xs=adaptation_L_keys,
        )
        samples = info.transformed_position  # tranform is the identity here
        flat_samples, unravel_fn = ravel_pytree(samples)
        ESS = 0.5 * effective_sample_size(
            jnp.array([flat_samples, flat_samples])
        )  # TODO: should only use a single chain here

        return state, params._replace(
            L=Lfactor * params.step_size * jnp.average(num_steps / ESS)
        )

    return adaptation_L


def handle_nans(state_old, state_new, step_size, step_size_max, kinetic_change):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""

    reduced_step_size = 0.8
    p, unravel_fn = ravel_pytree(state_new.position)
    nonans = (jnp.all(jnp.isfinite(p)))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (state_new, step_size_max, kinetic_change),
        (state_old, step_size * reduced_step_size, 0.0),
    )

    return nonans, state, step_size, kinetic_change
