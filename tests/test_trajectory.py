"""Test the trajectory integration"""
import jax
import jax.numpy as jnp
import pytest

import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.termination as termination
import blackjax.inference.trajectory as trajectory

divergence_threshold = 1000


@pytest.mark.parametrize("case", [(0.0001, False), (1000, True)])
def test_dynamic_progressive_integration_divergence(case):
    rng_key = jax.random.PRNGKey(0)

    def potential_fn(x):
        return jax.scipy.stats.norm.logpdf(x)

    step_size, should_diverge = case
    position = 1.0
    inverse_mass_matrix = jnp.array([1.0])

    momentum_generator, kinetic_energy_fn, uturn_check_fn = metrics.gaussian_euclidean(
        inverse_mass_matrix
    )

    integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    (
        new_criterion_state,
        update_criterion_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = trajectory.dynamic_progressive_integration(
        integrator,
        kinetic_energy_fn,
        update_criterion_state,
        is_criterion_met,
        divergence_threshold,
    )

    # Initialize
    direction = 1
    initial_state = integrators.new_integrator_state(
        potential_fn, position, momentum_generator(rng_key, position)
    )
    termination_state = new_criterion_state(initial_state, 10)
    max_num_steps = 100

    _, _, _, is_diverging, _ = trajectory_integrator(
        rng_key, initial_state, direction, termination_state, max_num_steps, step_size
    )

    assert is_diverging.item() is should_diverge
