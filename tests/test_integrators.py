import itertools

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import blackjax.mcmc.integrators as integrators


def HarmonicOscillator(inv_mass_matrix, k=1.0, m=1.0):
    """Potential and Kinetic energy of an harmonic oscillator."""

    def potential_energy(q):
        return jnp.sum(0.5 * k * jnp.square(q["x"]))

    def kinetic_energy(p):
        v = jnp.multiply(inv_mass_matrix, p["x"])
        return jnp.sum(0.5 * jnp.dot(v, p["x"]))

    return potential_energy, kinetic_energy


def FreeFall(inv_mass_matrix, g=1.0):
    """Potential and kinetic energy of a free-falling object."""

    def potential_energy(q):
        return jnp.sum(g * q["x"])

    def kinetic_energy(p):
        v = jnp.multiply(inv_mass_matrix, p["x"])
        return jnp.sum(0.5 * jnp.dot(v, p["x"]))

    return potential_energy, kinetic_energy


def PlanetaryMotion(inv_mass_matrix):
    """Potential and kinetic energy for planar planetary motion."""

    def potential_energy(q):
        return -1.0 / jnp.power(q["x"] ** 2 + q["y"] ** 2, 0.5)

    def kinetic_energy(p):
        z = jnp.stack([p["x"], p["y"]], axis=-1)
        return 0.5 * jnp.dot(inv_mass_matrix, z**2)

    return potential_energy, kinetic_energy


algorithms = {
    "velocity_verlet": {"algorithm": integrators.velocity_verlet, "precision": 1e-4},
    "mclachlan": {"algorithm": integrators.mclachlan, "precision": 1e-5},
    "yoshida": {"algorithm": integrators.yoshida, "precision": 1e-6},
}


examples = {
    "free_fall": {
        "model": FreeFall,
        "num_steps": 100,
        "step_size": 0.01,
        "q_init": {"x": 0.0},
        "p_init": {"x": 1.0},
        "q_final": {"x": 0.5},
        "p_final": {"x": 1.0},
        "inv_mass_matrix": jnp.array([1.0]),
    },
    "harmonic_oscillator": {
        "model": HarmonicOscillator,
        "num_steps": 100,
        "step_size": 0.01,
        "q_init": {"x": 0.0},
        "p_init": {"x": 1.0},
        "q_final": {"x": jnp.sin(1.0)},
        "p_final": {"x": jnp.cos(1.0)},
        "inv_mass_matrix": jnp.array([1.0]),
    },
    "planetary_motion": {
        "model": PlanetaryMotion,
        "num_steps": 628,
        "step_size": 0.01,
        "q_init": {"x": 1.0, "y": 0.0},
        "p_init": {"x": 0.0, "y": 1.0},
        "q_final": {"x": 1.0, "y": 0.0},
        "p_final": {"x": 0.0, "y": 1.0},
        "inv_mass_matrix": jnp.array([1.0, 1.0]),
    },
}


class IntegratorTest(chex.TestCase):
    """Test the numerical accuracy of trajectory integrators.

    We compare the evolution of the trajectory to analytical integration, and
    the conservation of energy. JAX's default float precision is 32bit; it is
    possible to change it to 64bit but only at startup. It is thus impossible
    to test both in the same run; we run the tests with the lower precision.
    """

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        itertools.product(
            ["free_fall", "harmonic_oscillator", "planetary_motion"],
            ["velocity_verlet", "mclachlan", "yoshida"],
        )
    )
    def test_integrator(self, example_name, integrator_name):
        integrator = algorithms[integrator_name]
        example = examples[example_name]

        model = example["model"]
        potential, kinetic_energy = model(example["inv_mass_matrix"])

        step = self.variant(integrator["algorithm"](potential, kinetic_energy))

        step_size = example["step_size"]

        q = example["q_init"]
        p = example["p_init"]
        initial_state = integrators.IntegratorState(
            q, p, potential(q), jax.grad(potential)(q)
        )
        final_state = jax.lax.fori_loop(
            0,
            example["num_steps"],
            lambda _, state: step(state, step_size),
            initial_state,
        )

        # We make sure that the particle moved from its initial position.
        chex.assert_tree_all_close(final_state.position, example["q_final"], atol=1e-2)

        # We now check the conservation of energy, the property that matters the most in HMC.
        energy = potential(q) + kinetic_energy(p)
        new_energy = potential(final_state.position) + kinetic_energy(
            final_state.momentum
        )
        self.assertAlmostEqual(energy, new_energy, delta=integrator["precision"])


if __name__ == "__main__":
    absltest.main()
