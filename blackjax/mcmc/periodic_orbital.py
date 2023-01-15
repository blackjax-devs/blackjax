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
"""Public API for Periodic Orbital Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.types import Array, PRNGKey, PyTree


class PeriodicOrbitalState(NamedTuple):
    """State of the periodic orbital algorithm.

    The periodic orbital algorithm takes one orbit with weights,
    samples from the points on that orbit according to their weights
    and returns another weighted orbit of the same period.

    positions
        a collection of points on the orbit, representing samples from
        the target distribution.
    weights
        weights of each point on the orbit, reweights points to ensure
        they are from the target distribution.
    directions
        an integer indicating the position on the orbit of each point.
    logdensities
        vector with logdensities (negative potential energies) for each point in
        the orbit.
    logdensities_grad
        matrix where each row is a vector with gradients of the logdensity
        function for each point in the orbit.
    """

    positions: PyTree
    weights: Array
    directions: Array
    logdensities: Array
    logdensities_grad: PyTree


class PeriodicOrbitalInfo(NamedTuple):
    """Additional information on the states in the orbit.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum
        the momentum that was sampled and used to integrate the trajectory.
    weights_mean
        mean of the the unnormalized weights of the orbit, ideally close
        to the (unknown) constant of proportionally missing from the target.
    weights_variance
        variance of the unnormalized weights of the orbit, ideally close to 0.
    """

    momentums: PyTree
    weights_mean: float
    weights_variance: float


def init(
    position: PyTree, logdensity_fn: Callable, period: int
) -> PeriodicOrbitalState:
    """Create a periodic orbital state from a position.

    Parameters
    ----------
    position
        the current values of the random variables whose posterior we want to
        sample from. Can be anything from a list, a (named) tuple or a dict of
        arrays. The arrays can either be Numpy or JAX arrays.
    logdensity_fn
        a function that returns the value of the log posterior when called
        with a position.
    period
        the number of steps used to build the orbit

    Returns
    -------
    A periodic orbital state that repeats the same position for `period` times,
    sets equal weights to all positions, assigns to each position a direction from
    0 to period-1, calculates the potential energies for each position and its
    gradient.
    """

    positions = jax.tree_util.tree_map(
        lambda position: jnp.array([position for _ in range(period)]), position
    )

    weights = jnp.array([1 / period for _ in range(period)])

    directions = jnp.arange(period)

    logdensities, logdensities_grad = jax.vmap(jax.value_and_grad(logdensity_fn))(
        positions
    )

    return PeriodicOrbitalState(
        positions, weights, directions, logdensities, logdensities_grad
    )


def kernel(
    bijection: Callable = integrators.velocity_verlet,
):
    """Build a Periodic Orbital kernel [1]_.

    Parameters
    ----------
    bijection
        transformation used to build the orbit (given a step size).

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    References
    ----------
    .. [1]: Kirill Neklyudov and Max Welling "Orbital MCMC." arXiv preprint
            arXiv:2010.08047 (2021).
    """

    def one_step(
        rng_key: PRNGKey,
        state: PeriodicOrbitalState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        period: int,
    ) -> Tuple[PeriodicOrbitalState, PeriodicOrbitalInfo]:
        """Generate a new orbit with the Periodic Orbital kernel.

        Choose a step from the orbit with probability proportional to its weights.
        Then shift the direction (or alternatively sample a new direction randomly),
        in order to make the algorithm irreversible, and compute a new orbit from
        the selected step and its direction.

        Parameters
        ----------
        rng_key
            pseudo random number generating key.
        state
            initial orbit.
        logdensity_fn
            log probability function we wish to sample from.
        step_size
            space between steps of the orbit.
        inverse_mass_matrix
            or a 1D array containing elements of its diagonal.
        period
            total steps used to build the orbit.

        Returns
        -------
        A kernel that chooses a step from the orbit and outputs a periodic orbital
        state and information about the iteration.

        """

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
        bijection_fn = bijection(logdensity_fn, kinetic_energy_fn)
        proposal_generator = periodic_orbital_proposal(
            bijection_fn, kinetic_energy_fn, period, step_size
        )

        key_choice, key_momentum = jax.random.split(rng_key, 2)

        (
            positions,
            weights,
            directions,
            logdensities,
            logdensities_grad,
        ) = state

        choice_indx = jax.random.choice(key_choice, len(weights), p=weights)
        position = jax.tree_util.tree_map(
            lambda positions: positions[choice_indx], positions
        )
        direction = directions[choice_indx]
        period = jnp.max(directions) + 1
        direction = jnp.mod(direction + jnp.array(period / 2, int), period)
        logdensity = logdensities[choice_indx]
        logdensity_grad = jax.tree_util.tree_map(
            lambda p_energy_grad: p_energy_grad[choice_indx], logdensities_grad
        )

        momentum = momentum_generator(key_momentum, position)

        augmented_state = integrators.IntegratorState(
            position,
            momentum,
            logdensity,
            logdensity_grad,
        )
        proposal, info = proposal_generator(direction, augmented_state)

        return proposal, info

    return one_step


def periodic_orbital_proposal(
    bijection: Callable,
    kinetic_energy_fn: Callable,
    period: int,
    step_size: float,
) -> Callable:
    """Periodic Orbital algorithm.

    The algorithm builds and orbit and computes the weights for each of its steps
    by applying a bijection `period` times, both forwards and backwards depending
    on the direction of the initial state.

    Parameters
    ----------
    bijection
        continuous, differentialble and bijective transformation used to build
        the orbit step by step.
    kinetic_energy_fn
        function that computes the kinetic energy.
    period
        total steps used to build the orbit.
    step_size
        size between each step of the orbit.

    Returns
    -------
    A kernel that generates a new periodic orbital state and information
    about the transition.

    """

    def generate(
        direction: int, init_state: integrators.IntegratorState
    ) -> Tuple[PeriodicOrbitalState, PeriodicOrbitalInfo]:
        """Generate orbit by applying bijection forwards and backwards on period.

        As described in algorithm 2 of [1]_, each iteration of the periodic orbital
        MCMC takes a position and its direction, i.e. its step in the orbit, then
        it runs the bijection backwards until it reaches the direction 0 and forwards
        until it reaches the direction period-1. For each step it calculates its
        weight using the target density, the auxilary variable's density and the
        bijection.

        References
        ----------
        .. [1]: Kirill Neklyudov and Max Welling "Orbital MCMC." arXiv preprint
                arXiv:2010.08047 (2021).
        """

        index_steps = jnp.arange(period) - direction

        def orbit_fn(state, i):
            state = jax.lax.cond(
                i != 0,
                lambda _: bijection(state, jnp.sign(i) * step_size),
                lambda _: init_state,
                operand=None,
            )
            kinetic_energy = kinetic_energy_fn(state.momentum)
            weight = state.logdensity - kinetic_energy
            return state, (state, jnp.exp(weight))

        _, (states, weights) = jax.lax.scan(orbit_fn, init_state, index_steps)

        directions = jnp.where(
            index_steps < 0, -(index_steps + 1), index_steps + direction
        )

        state = PeriodicOrbitalState(
            states.position,
            weights / jnp.sum(weights),
            directions,
            states.logdensity,
            states.logdensity_grad,
        )
        info = PeriodicOrbitalInfo(
            states.momentum,
            jnp.mean(weights),
            jnp.var(weights),
        )
        return state, info

    return generate
