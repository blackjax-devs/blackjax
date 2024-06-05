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
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from blackjax.util import pytree_size, streaming_average_update


class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    L
        The momentum decoherent rate for the MCLMC algorithm.
    step_size
        The step size used for the MCLMC algorithm.
    sqrt_diag_cov
        A matrix used for preconditioning.
    """

    L: float
    step_size: float
    sqrt_diag_cov: float


def mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    diagonal_preconditioning=True,
):
    """
    Finds the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_va
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.

    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final hyperparameters.

    Example
    -------
    .. code::
        kernel = lambda std_mat : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        std_mat=std_mat,
        )

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=preconditioning,
        )
    """
    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(
        jnp.sqrt(dim), jnp.sqrt(dim) * 0.25, sqrt_diag_cov=jnp.ones((dim,))
    )
    part1_key, part2_key = jax.random.split(rng_key, 2)

    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        diagonal_preconditioning=diagonal_preconditioning,
    )(state, params, num_steps, part1_key)

    if frac_tune3 != 0:
        state, params = make_adaptation_L(
            mclmc_kernel(params.sqrt_diag_cov), frac=frac_tune3, Lfactor=0.4
        )(state, params, num_steps, part2_key)

    return state, params


def make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    diagonal_preconditioning,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        time, x_average, step_size_max = adaptive_state

        # dynamics
        next_state, info = kernel(params.sqrt_diag_cov)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )
        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        streaming_avg = streaming_average_update(
            current_value=jnp.array([x, jnp.square(x)]),
            previous_weight_and_average=streaming_avg,
            weight=(1 - mask) * success * params.step_size,
            zero_prevention=mask,
        )

        return (state, params, adaptive_state, streaming_avg), None

    run_steps = lambda xs, state, params: jax.lax.scan(
        step,
        init=(
            state,
            params,
            (0.0, 0.0, jnp.inf),
            (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
        ),
        xs=xs,
    )[0]

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = (
            int(num_steps * frac_tune1) + 1,
            int(num_steps * frac_tune2) + 1,
        )
        L_step_size_adaptation_keys = jax.random.split(
            rng_key, num_steps1 + num_steps2 + 1
        )
        L_step_size_adaptation_keys, final_key = (
            L_step_size_adaptation_keys[:-1],
            L_step_size_adaptation_keys[-1],
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = 1 - jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # run the steps
        state, params, _, (_, average) = run_steps(
            xs=(mask, L_step_size_adaptation_keys), state=state, params=params
        )

        L = params.L
        # determine L
        sqrt_diag_cov = params.sqrt_diag_cov
        if num_steps2 != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            L = jnp.sqrt(jnp.sum(variances))

            if diagonal_preconditioning:
                sqrt_diag_cov = jnp.sqrt(variances)
                params = params._replace(sqrt_diag_cov=sqrt_diag_cov)
                L = jnp.sqrt(dim)

                # readjust the stepsize
                steps = num_steps2 // 3  # we do some small number of steps
                keys = jax.random.split(final_key, steps)
                state, params, _, (_, average) = run_steps(
                    xs=(jnp.ones(steps), keys), state=state, params=params
                )

        return state, MCLMCAdaptationState(L, params.step_size, sqrt_diag_cov)

    return L_step_size_adaptation


def make_adaptation_L(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                L=params.L,
                step_size=params.step_size,
            )

            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])

        return state, params._replace(
            L=Lfactor * params.step_size * jnp.mean(num_steps / ess)
        )

    return adaptation_L


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case."""

    reduced_step_size = 0.8
    p, unravel_fn = ravel_pytree(next_state.position)
    nonans = jnp.all(jnp.isfinite(p))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )
    return nonans, state, step_size, kinetic_change
