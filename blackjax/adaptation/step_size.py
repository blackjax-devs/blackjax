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
"""Step size adaptation"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.hmc import HMCState
from blackjax.optimizers.dual_averaging import dual_averaging
from blackjax.types import PRNGKey

__all__ = [
    "DualAveragingAdaptationState",
    "dual_averaging_adaptation",
    "find_reasonable_step_size",
]


# -------------------------------------------------------------------
#                        DUAL AVERAGING
# -------------------------------------------------------------------


class DualAveragingAdaptationState(NamedTuple):
    """State carried through the dual averaging procedure.

    log_step_size
        The logarithm of the current value of the step size.
    log_step_size_avg
        The time-weighted average of the values that the logarithm of the step
        size has taken so far.
    step
        The current iteration step.
    avg_err
        The time average of the value of the quantity :math:`H_t`, the
        difference between the target acceptance rate and the current
        acceptance rate.
    mu
        Arbitrary point the values of log_step_size are shrunk towards. Chose
        to be :math:`\\log(10 \\epsilon_0)` where :math:`\\epsilon_0` is chosen
        in this context to be the step size given by the
        `find_reasonable_step_size` procedure.

    """

    log_step_size: float
    log_step_size_avg: float
    step: int
    avg_error: float
    mu: float


def dual_averaging_adaptation(
    target: float, t0: int = 10, gamma: float = 0.05, kappa: float = 0.75
) -> tuple[Callable, Callable, Callable]:
    """Tune the step size in order to achieve a desired target acceptance rate.

    Let us note :math:`\\epsilon` the current step size, :math:`\\alpha_t` the
    metropolis acceptance rate at time :math:`t` and :math:`\\delta` the desired
    aceptance rate. We define:

    .. math:
        H_t = \\delta - \\alpha_t

    the error at time t. We would like to find a procedure that adapts the
    value of :math:`\\epsilon` such that :math:`h(x) =\\mathbb{E}\\left[H_t|\\epsilon\\right] = 0`

    Following :cite:p:`nesterov2009primal`, the authors of :cite:p:`hoffman2014no` proposed the following update scheme. If
    we note :math:`x = \\log \\epsilon` we follow:

    .. math:
        x_{t+1} \\LongLeftArrow \\mu - \\frac{\\sqrt{t}}{\\gamma} \\frac{1}{t+t_0} \\sum_{i=1}^t H_i
        \\overline{x}_{t+1} \\LongLeftArrow x_{t+1}\\, t^{-\\kappa}  + \\left(1-t^\\kappa\\right)\\overline{x}_t

    :math:`\\overline{x}_{t}` is guaranteed to converge to a value such that
    :math:`h(\\overline{x}_t)` converges to 0, i.e. the Metropolis acceptance
    rate converges to the desired rate.

    See reference :cite:p:`hoffman2014no` (section 3.2.1) for a detailed discussion.

    Parameters
    ----------
    t0: float >= 0
        Free parameter that stabilizes the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in :cite:p:`hoffman2014no` with a default
        value of 10.
    gamma:
        Controls the speed of convergence of the scheme. The authors of :cite:p:`hoffman2014no` recommend
        a value of 0.05.
    kappa: float in [0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in :cite:p:`hoffman2014no`, with a recommended value of .75
    target:
        Target acceptance rate.

    Returns
    -------
    init
        A function that initializes the state of the dual averaging scheme.
    update
        A function that updates the state of the dual averaging scheme.

    """
    da_init, da_update, da_final = dual_averaging(t0, gamma, kappa)

    def init(inital_step_size: float) -> DualAveragingAdaptationState:
        """Initialize the state of the dual averaging scheme.

        The parameter :math:`\\mu` is set to :math:`\\log(10 \\epsilon_1)`
        where :math:`\\epsilon_1` is the initial value of the step size.
        """
        return DualAveragingAdaptationState(*da_init(inital_step_size))

    def update(
        da_state: DualAveragingAdaptationState, acceptance_rate: float
    ) -> DualAveragingAdaptationState:
        """Update the state of the Dual Averaging adaptive algorithm.

        Parameters
        ----------
        da_state:
            The current state of the dual averaging algorithm.
        acceptance_rate: float in [0, 1]
            The current metropolis acceptance rate.

        Returns
        -------
        The updated state of the dual averaging algorithm.

        """
        gradient = target - acceptance_rate
        return DualAveragingAdaptationState(*da_update(da_state, gradient))

    def final(da_state: DualAveragingAdaptationState) -> float:
        return jnp.exp(da_state.log_step_size_avg)

    return init, update, final


# -------------------------------------------------------------------
#                 REASONABLE FIRST STEP SIZE
# -------------------------------------------------------------------


class ReasonableStepSizeState(NamedTuple):
    """State carried through the search for a reasonable first step size.

    step
        The current iteration step.
    direction: {-1, 1}
        Determines whether the step size should be increased or decreased during the
        previous step search. If direction = 1 it will be increased, otherwise decreased.
    previous_direction
        The previous direction. It is necessary to carry it because the choice of step
        size is made at the end of the search update.
    step_size
        The current step size in the search.

    """

    step: int
    direction: int
    previous_direction: int
    step_size: float


def find_reasonable_step_size(
    rng_key: PRNGKey,
    kernel_generator: Callable[[float], Callable],
    reference_state: HMCState,
    initial_step_size: float,
    target_accept: float = 0.65,
) -> float:
    """Find a reasonable initial step size during warmup.

    While the dual averaging scheme is guaranteed to converge to a reasonable
    value for the step size starting from any value, choosing a good first
    value can speed up the convergence. This heuristics doubles and halves the
    step size until the acceptance probability of the HMC proposal crosses the
    target value :cite:p:`hoffman2014no`.

    Parameters
    ----------
    rng_key
       Key used by JAX's random number generator.
    kernel_generator
        A function that takes a step size as an input and returns the corresponding
        sampling kernel.
    reference_hmc_state
        The location (HMC state) where this first step size must be found. This function
        never advances the chain.
    inverse_mass_matrix
        The inverse mass matrix relative to which the step size must be found.
    initial_step_size
        The first step size used to start the search.
    target_accept
        Once that value of the metropolis acceptance probability is reached we
        estimate that we have found a "reasonable" first step size.

    Returns
    -------
    float
        A reasonable first value for the step size.

    """
    fp_limit = jnp.finfo(jax.lax.dtype(initial_step_size))

    def do_continue(rss_state: ReasonableStepSizeState) -> bool:
        """Decides whether the search should continue.

        The search stops when it crosses the `target_accept` threshold, i.e.
        when the current direction is opposite to the previous direction.

        Note
        ----
        Per JAX's documentation :cite:p:`jax_finfo` the `jnp.finfo` object is cached so we do not
        occur any performance penalty when calling it repeatedly inside this
        function.

        """
        _, direction, previous_direction, step_size = rss_state

        not_too_large = (step_size < fp_limit.max) | (direction <= 0)
        not_too_small = (step_size > fp_limit.tiny) | (direction >= 0)
        is_step_size_not_extreme = not_too_large & not_too_small
        has_acceptance_rate_not_crossed_threshold = (previous_direction == 0) | (
            direction == previous_direction
        )
        return is_step_size_not_extreme & has_acceptance_rate_not_crossed_threshold

    def update(rss_state: ReasonableStepSizeState) -> ReasonableStepSizeState:
        """Perform one step of the step size search."""
        i, direction, _, step_size = rss_state
        subkey = jax.random.fold_in(rng_key, i)

        step_size = (2.0**direction) * step_size
        kernel = kernel_generator(step_size)
        _, info = kernel(subkey, reference_state)

        new_direction = jnp.where(target_accept < info.acceptance_rate, 1, -1)
        return ReasonableStepSizeState(i + 1, new_direction, direction, step_size)

    rss_state = ReasonableStepSizeState(0, 0, 0, initial_step_size)
    rss_state = jax.lax.while_loop(do_continue, update, rss_state)

    return rss_state.step_size
