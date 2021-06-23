from typing import Callable

import jax

from blackjax.inference.integrators import IntegratorState
from blackjax.inference.trajectory import Trajectory


def constant_time_harvester(
    momentum_generator: Callable,
    trajectory_sampler: Callable,
    num_integration_steps: int,  # will be replaced with is_criterion_met
    num_before_harvest: int,
):
    """This kernel samples a fixed number of states from the trajectory and
    returns the last proposal as a sample.

    If the trajectory sampler reaches its termination condition the harvester
    samples a new value from the momentum and continues integrating.

    """

    def reinitialize_trajectory(rng_key, state, trajectory):
        momentum = momentum_generator(rng_key)
        new_state = IntegratorState(
            state.position,
            momentum,
            state.potential_energy,
            state.potential_energy_grad,
        )
        new_trajectory = Trajectory(new_state, new_state, new_state[1], 1)

        return new_trajectory

    def harvest(rng_key, harvest_state):
        state, proposal, trajectory = harvest_state

        def body_fn(state, rng_key):
            state, proposal, trajectory = state

            resample_key, rng_key = jax.random.split(rng_key)

            # Resample momentum if we have reached the termination criterion
            trajectory = jax.lax.cond(
                trajectory.states >= num_integration_steps,
                (resample_key, state, trajectory),
                lambda x: x[2],
                reinitialize_trajectory,
            )

            new_state, new_proposal, new_trajectory = trajectory_sampler(
                rng_key, proposal, trajectory
            )
            return new_state, new_proposal, new_trajectory

        keys = jax.random.split(rng_key, num_before_harvest)
        last_harvest_state, _ = jax.lax.scan(
            body_fn, (state, proposal, trajectory), keys
        )

        return last_harvest_state

    return harvest
