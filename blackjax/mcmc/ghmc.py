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
"""Public API for the Generalized (Non-reversible w/ persistent momentum) HMC Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.proposal import nonreversible_slice_sampling
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["GHMCState", "init", "build_kernel", "as_top_level_api"]


class GHMCState(NamedTuple):
    """State of the Generalized HMC algorithm.

    The Generalized HMC algorithm is persistent on its momentum, hence
    taking as input a position and momentum pair, updating and returning
    it for the next iteration. The algorithm also uses a persistent slice
    to perform a non-reversible Metropolis Hastings update, thus we also
    store the current slice variable and return its updated version after
    each iteration. To make computations more efficient, we also store
    the current logdensity as well as the current gradient of the
    logdensity.

    """

    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    slice: float


def init(
    position: ArrayLikeTree,
    rng_key: PRNGKey,
    logdensity_fn: Callable,
) -> GHMCState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)

    key_mometum, key_slice = jax.random.split(rng_key)
    momentum = generate_gaussian_noise(key_mometum, position)
    slice = jax.random.uniform(key_slice, minval=-1.0, maxval=1.0)

    return GHMCState(position, momentum, logdensity, logdensity_grad, slice)


def build_kernel(
    noise_fn: Callable = lambda _: 0.0,
    divergence_threshold: float = 1000,
):
    """Build a Generalized HMC kernel.

    The Generalized HMC kernel performs a similar procedure to the standard HMC
    kernel with the difference of a persistent momentum variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step. This means that; apart from momentum and slice variables that are dependent
    on the previous momentum and slice variables, and a Metropolis-Hastings step
    performed (equivalently) as slice sampling; the standard HMC's implementation can
    be re-used to perform Generalized HMC sampling.

    Parameters
    ----------
    noise_fn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key, a Pytree that contains the current state
    of the chain, and free parameters of the sampling mechanism; and that
    returns a new state of the chain along with information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: GHMCState,
        logdensity_fn: Callable,
        step_size: float,
        momentum_inverse_scale: ArrayLikeTree,
        alpha: float,
        delta: float,
    ) -> tuple[GHMCState, hmc.HMCInfo]:
        """Generate new sample with the Generalized HMC kernel.

        Parameters
        ----------
        rng_key
            JAX's pseudo random number generating key.
        state
            Current state of the chain.
        logdensity_fn
            (Unnormalized) Log density function being targeted.
        step_size
            Variable specifying the size of the integration step.
        momentum_inverse_scale
            Pytree with the same structure as the targeted position variable
            specifying the per dimension inverse scaling transformation applied
            to the persistent momentum variable prior to the integration step.
        alpha
            Variable specifying the degree of persistent momentum, complementary
            to independent new momentum.
        delta
            Fixed (non-random) amount of translation added at each new iteration
            to the slice variable for non-reversible slice sampling.

        """

        flat_inverse_scale = jax.flatten_util.ravel_pytree(momentum_inverse_scale)[0]
        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            flat_inverse_scale**2
        )

        symplectic_integrator = integrators.velocity_verlet(
            logdensity_fn, kinetic_energy_fn
        )
        proposal_generator = hmc.hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            divergence_threshold=divergence_threshold,
            sample_proposal=nonreversible_slice_sampling,
        )

        key_momentum, key_noise = jax.random.split(rng_key)
        position, momentum, logdensity, logdensity_grad, slice = state
        # New momentum is persistent
        momentum = update_momentum(key_momentum, state, alpha, momentum_generator)
        # Slice is non-reversible
        slice = ((slice + 1.0 + delta + noise_fn(key_noise)) % 2) - 1.0

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        # Note that ghmc use nonreversible_slice_sampling, which overloads the pattern
        # of SampleProposal and do not actually return the acceptance rate.
        proposal, info, slice_next = proposal_generator(slice, integrator_state)
        proposal = hmc.flip_momentum(proposal)
        state = GHMCState(
            position=proposal.position,
            momentum=proposal.momentum,
            logdensity=proposal.logdensity,
            logdensity_grad=proposal.logdensity_grad,
            slice=slice_next,
        )

        return state, info

    return kernel


def update_momentum(rng_key, state, alpha, momentum_generator):
    """Persistent update of the momentum variable.

    Performs a persistent update of the momentum, taking as input the previous
    momentum, a random number generating key, the parameter alpha and the
    momentum generator function. Outputs
    an updated momentum that is a mixture of the previous momentum a new sample
    from a Gaussian density (dependent on alpha). The weights of the mixture of
    these two components are a function of alpha.

    """
    position, momentum, *_ = state

    momentum = jax.tree.map(
        lambda prev_momentum, shifted_momentum: prev_momentum * jnp.sqrt(1.0 - alpha)
        + jnp.sqrt(alpha) * shifted_momentum,
        momentum,
        momentum_generator(rng_key, position),
    )

    return momentum


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    momentum_inverse_scale: ArrayLikeTree,
    alpha: float,
    delta: float,
    *,
    divergence_threshold: int = 1000,
    noise_gn: Callable = lambda _: 0.0,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Generalized HMC kernel.

    The Generalized HMC kernel performs a similar procedure to the standard HMC
    kernel with the difference of a persistent momentum variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step.

    This means that the sampling of the momentum variable depends on the previous
    momentum, the rate of persistence depends on the alpha parameter, and that the
    Metropolis-Hastings accept/reject step is done through slice sampling with a
    non-reversible slice variable also dependent on the previous slice, the determinisitc
    transformation is defined by the delta parameter.

    The Generalized HMC does not have a trajectory length parameter, it always performs
    one iteration of the velocity verlet integrator with a given step size, making
    the algorithm a good candiate for running many chains in parallel.

    Examples
    --------

    A new Generalized HMC kernel can be initialized and used with the following code:

    .. code::

        ghmc_kernel = blackjax.ghmc(logdensity_fn, step_size, alpha, delta)
        state = ghmc_kernel.init(rng_key, position)
        new_state, info = ghmc_kernel.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(ghmc_kernel.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        A PyTree of the same structure as the target PyTree (position) with the
        values used for as a step size for each dimension of the target space in
        the velocity verlet integrator.
    alpha
        The value defining the persistence of the momentum variable.
    delta
        The value defining the deterministic translation of the slice variable.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    noise_gn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(noise_gn, divergence_threshold)

    def init_fn(position: ArrayLikeTree, rng_key: PRNGKey):
        return init(position, rng_key, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            momentum_inverse_scale,
            alpha,
            delta,
        )

    return SamplingAlgorithm(init_fn, step_fn)
