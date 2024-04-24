"""Test the trajectory integration"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory

divergence_threshold = 1000


class TrajectoryTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([(0.0001, False), (1000, True)])
    def test_dynamic_progressive_integration_divergence(
        self, step_size, should_diverge
    ):
        rng_key = jax.random.key(0)

        logdensity_fn = jax.scipy.stats.norm.logpdf

        position = 1.0
        inverse_mass_matrix = jnp.array([1.0])

        (
            momentum_generator,
            kinetic_energy_fn,
            uturn_check_fn,
        ) = metrics.gaussian_euclidean(inverse_mass_matrix)

        integrator = integrators.velocity_verlet(logdensity_fn, kinetic_energy_fn)
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
            logdensity_fn, position, momentum_generator(rng_key, position)
        )
        initial_energy = -initial_state.logdensity + kinetic_energy_fn(
            initial_state.momentum
        )
        termination_state = new_criterion_state(initial_state, 10)
        max_num_steps = 100

        _, _, _, is_diverging, _ = self.variant(trajectory_integrator)(
            rng_key,
            initial_state,
            direction,
            termination_state,
            max_num_steps,
            step_size,
            initial_energy,
        )

        assert is_diverging.item() is should_diverge

    def test_dynamic_progressive_equal_recursive(self):
        rng_key = jax.random.key(23133)

        def logdensity_fn(x):
            return -((1.0 - x[0]) ** 2) - 1.5 * (x[1] - x[0] ** 2) ** 2

        inverse_mass_matrix = jnp.asarray([[1.0, 0.5], [0.5, 1.25]])
        (
            momentum_generator,
            kinetic_energy_fn,
            uturn_check_fn,
        ) = metrics.gaussian_euclidean(inverse_mass_matrix)

        integrator = integrators.velocity_verlet(logdensity_fn, kinetic_energy_fn)
        (
            new_criterion_state,
            update_criterion_state,
            is_criterion_met,
        ) = termination.iterative_uturn_numpyro(uturn_check_fn)
        (
            integrator,
            kinetic_energy_fn,
            update_criterion_state,
            is_criterion_met,
            uturn_check_fn,
        ) = (
            jax.jit(x)
            for x in (
                integrator,
                kinetic_energy_fn,
                update_criterion_state,
                is_criterion_met,
                uturn_check_fn,
            )
        )

        trajectory_integrator = trajectory.dynamic_progressive_integration(
            integrator,
            kinetic_energy_fn,
            update_criterion_state,
            is_criterion_met,
            divergence_threshold,
        )
        buildtree_integrator = trajectory.dynamic_recursive_integration(
            integrator,
            kinetic_energy_fn,
            uturn_check_fn,
            divergence_threshold,
        )

        for i in range(50):
            subkey = jax.random.fold_in(rng_key, i)
            (
                rng_buildtree,
                rng_direction,
                rng_tree_depth,
                rng_step_size,
                rng_position,
                rng_momentum,
            ) = jax.random.split(subkey, 6)
            direction = jax.random.choice(rng_direction, jnp.array([-1, 1]))
            tree_depth = jax.random.choice(rng_tree_depth, np.arange(2, 5))
            initial_state = integrators.new_integrator_state(
                logdensity_fn,
                jax.random.normal(rng_position, [2]),
                jax.random.normal(rng_momentum, [2]),
            )
            step_size = jnp.abs(jax.random.normal(rng_step_size, [])) * 0.1
            initial_energy = -initial_state.logdensity + kinetic_energy_fn(
                initial_state.momentum
            )

            termination_state = new_criterion_state(initial_state, tree_depth)
            (
                proposal0,
                trajectory0,
                _,
                is_diverging0,
                has_terminated0,
            ) = trajectory_integrator(
                rng_buildtree,
                initial_state,
                direction,
                termination_state,
                2**tree_depth,
                step_size,
                initial_energy,
            )

            (
                _,
                proposal1,
                trajectory1,
                is_diverging1,
                has_terminated1,
            ) = buildtree_integrator(
                rng_buildtree,
                initial_state,
                direction,
                tree_depth,
                step_size,
                initial_energy,
            )
            # Assert that the trajectory being built is the same
            chex.assert_trees_all_close(trajectory0, trajectory1, rtol=1e-5)

            assert is_diverging0 == is_diverging1
            assert has_terminated0 == has_terminated1
            # We dont expect the proposal to be the same (even with the same PRNGKey
            # as the order of selection is different). but the property associate
            # with the full trajectory should be the same.
            np.testing.assert_allclose(proposal0.weight, proposal1.weight, rtol=1e-5)
            np.testing.assert_allclose(
                proposal0.sum_log_p_accept, proposal1.sum_log_p_accept, rtol=1e-5
            )

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        [
            (0.0000000001, False, False, 10),
            (1, False, True, 2),
            (100000, True, True, 1),
        ],
    )
    def test_dynamic_progressive_expansion(
        self, step_size, should_diverge, should_turn, expected_doublings
    ):
        rng_key = jax.random.key(0)

        def logdensity_fn(x):
            return -0.5 * x**2

        position = 0.0
        inverse_mass_matrix = jnp.array([1.0])

        (
            momentum_generator,
            kinetic_energy_fn,
            uturn_check_fn,
        ) = metrics.gaussian_euclidean(inverse_mass_matrix)

        integrator = integrators.velocity_verlet(logdensity_fn, kinetic_energy_fn)
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

        expand = trajectory.dynamic_multiplicative_expansion(
            trajectory_integrator, uturn_check_fn
        )

        state = integrators.new_integrator_state(
            logdensity_fn, position, momentum_generator(rng_key, position)
        )
        energy = -state.logdensity + kinetic_energy_fn(state.momentum)
        initial_proposal = proposal.Proposal(state, energy, 0.0, -np.inf)
        initial_termination_state = new_criterion_state(state, 10)
        initial_trajectory = trajectory.Trajectory(
            state,
            state,
            state.momentum,
            0,
        )
        initial_expansion_state = trajectory.DynamicExpansionState(
            0, initial_proposal, initial_trajectory, initial_termination_state
        )

        expansion_state, (is_diverging, is_turning) = self.variant(expand)(
            rng_key, initial_expansion_state, energy, step_size
        )

        assert is_diverging == should_diverge
        assert expansion_state.step == expected_doublings
        assert is_turning == should_turn

    def test_static_integration_variable_num_steps(self):
        rng_key = jax.random.key(0)

        logdensity_fn = jax.scipy.stats.norm.logpdf
        position = 1.0
        inverse_mass_matrix = jnp.array([1.0])

        (
            momentum_generator,
            kinetic_energy_fn,
            _,
        ) = metrics.gaussian_euclidean(inverse_mass_matrix)
        initial_state = integrators.new_integrator_state(
            logdensity_fn, position, momentum_generator(rng_key, position)
        )

        integrator = integrators.velocity_verlet(logdensity_fn, kinetic_energy_fn)
        static_integration = trajectory.static_integration(integrator)

        # When not jitted, the number of steps is static and this integration is
        # performed using a scan
        scan_state = static_integration(initial_state, 0.1, 10)

        # When jitted, the number of steps is no longer static - make sure that
        # we still get the same result
        fori_state = jax.jit(static_integration)(initial_state, 0.1, 10)

        chex.assert_trees_all_close(fori_state, scan_state, rtol=1e-5)

    def test_dynamic_hmc_integration_steps(self):
        rng_key = jax.random.key(0)
        num_step_key, sample_key = jax.random.split(rng_key)
        initial_position = jnp.array(3.0)
        parameters = {"step_size": 3.9, "inverse_mass_matrix": jnp.array([1.0])}

        unique_integration_steps = jnp.asarray([5, 10, 20])
        unique_probs = jnp.asarray([0.1, 0.8, 0.1])
        num_step_fn = lambda key: jax.random.choice(
            key, unique_integration_steps, p=unique_probs
        )
        kernel_factory = dynamic_hmc.build_kernel(integration_steps_fn=num_step_fn)

        logprob = jax.scipy.stats.norm.logpdf
        hmc_kernel = lambda key, state: kernel_factory(
            key, state, logprob, **parameters
        )
        init_state = dynamic_hmc.init(initial_position, logprob, num_step_key)

        def one_step(state, rng_key):
            state, info = hmc_kernel(rng_key, state)
            return state, info

        num_iter = 1000
        keys = jax.random.split(sample_key, num_iter)
        _, infos = jax.lax.scan(one_step, init_state, keys)
        _, unique_counts = np.unique(infos.num_integration_steps, return_counts=True)

        np.testing.assert_allclose(unique_counts / num_iter, unique_probs, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
