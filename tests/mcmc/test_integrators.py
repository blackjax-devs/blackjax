import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.integrators as integrators
from blackjax.mcmc.integrators import esh_dynamics_momentum_update_one_step
from blackjax.util import generate_unit_vector


def HarmonicOscillator(inv_mass_matrix, k=1.0, m=1.0):
    """Potential and Kinetic energy of an harmonic oscillator."""

    def neg_potential_energy(q):
        return -jnp.sum(0.5 * k * jnp.square(q["x"]))

    def kinetic_energy(p):
        v = jnp.multiply(inv_mass_matrix, p["x"])
        return jnp.sum(0.5 * jnp.dot(v, p["x"]))

    return neg_potential_energy, kinetic_energy


def FreeFall(inv_mass_matrix, g=1.0):
    """Potential and kinetic energy of a free-falling object."""

    def neg_potential_energy(q):
        return -jnp.sum(g * q["x"])

    def kinetic_energy(p):
        v = jnp.multiply(inv_mass_matrix, p["x"])
        return jnp.sum(0.5 * jnp.dot(v, p["x"]))

    return neg_potential_energy, kinetic_energy


def PlanetaryMotion(inv_mass_matrix):
    """Potential and kinetic energy for planar planetary motion."""

    def neg_potential_energy(q):
        return 1.0 / jnp.power(q["x"] ** 2 + q["y"] ** 2, 0.5)

    def kinetic_energy(p):
        z = jnp.stack([p["x"], p["y"]], axis=-1)
        return 0.5 * jnp.dot(inv_mass_matrix, z**2)

    return neg_potential_energy, kinetic_energy


def MultivariateNormal(inv_mass_matrix):
    """Potential and kinetic energy for a multivariate normal distribution."""

    def log_density(q):
        q, _ = ravel_pytree(q)
        return stats.multivariate_normal.logpdf(q, jnp.zeros_like(q), inv_mass_matrix)

    def kinetic_energy(p):
        p, _ = ravel_pytree(p)
        return 0.5 * p.T @ inv_mass_matrix @ p

    return log_density, kinetic_energy


mvnormal_position_init = {
    "a": 0.0,
    "b": jnp.asarray([1.0, 2.0, 3.0]),
    "c": jnp.ones((2, 1)),
}
_, unravel_fn = ravel_pytree(mvnormal_position_init)
key0, key1 = jax.random.split(jax.random.key(52))
mvnormal_momentum_init = unravel_fn(jax.random.normal(key0, (6,)))
a = jax.random.normal(key1, (6, 6))
cov = jnp.matmul(a.T, a)
# Validated numerically
mvnormal_position_end = unravel_fn(
    jnp.asarray([0.38887993, 0.85231394, 2.7879136, 3.0339851, 0.5856687, 1.9291426])
)
mvnormal_momentum_end = unravel_fn(
    jnp.asarray([0.46576163, 0.23854092, 1.2518811, -0.35647452, -0.742138, 1.2552949])
)

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
    "multivariate_normal": {
        "model": MultivariateNormal,
        "num_steps": 16,
        "step_size": 0.005,
        "q_init": mvnormal_position_init,
        "p_init": mvnormal_momentum_init,
        "q_final": mvnormal_position_end,
        "p_final": mvnormal_momentum_end,
        "inv_mass_matrix": cov,
    },
}

algorithms = {
    "velocity_verlet": {"algorithm": integrators.velocity_verlet, "precision": 1e-4},
    "mclachlan": {"algorithm": integrators.mclachlan, "precision": 1e-5},
    "yoshida": {"algorithm": integrators.yoshida, "precision": 1e-6},
    "noneuclidean_leapfrog": {"algorithm": integrators.noneuclidean_leapfrog},
    "noneuclidean_mclachlan": {"algorithm": integrators.noneuclidean_mclachlan},
    "noneuclidean_yoshida": {"algorithm": integrators.noneuclidean_yoshida},
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
            [
                "free_fall",
                "harmonic_oscillator",
                "planetary_motion",
                "multivariate_normal",
            ],
            [
                "velocity_verlet",
                "mclachlan",
                "yoshida",
            ],
        )
    )
    def test_euclidean_integrator(self, example_name, integrator_name):
        integrator = algorithms[integrator_name]
        example = examples[example_name]

        model = example["model"]
        neg_potential, kinetic_energy = model(example["inv_mass_matrix"])

        step = self.variant(integrator["algorithm"](neg_potential, kinetic_energy))

        step_size = example["step_size"]

        q = example["q_init"]
        p = example["p_init"]
        initial_state = integrators.IntegratorState(
            q, p, neg_potential(q), jax.grad(neg_potential)(q)
        )

        final_state = jax.lax.fori_loop(
            0,
            example["num_steps"],
            lambda _, state: step(state, step_size),
            initial_state,
        )

        # We make sure that the particle moved from its initial position.
        chex.assert_trees_all_close(final_state.position, example["q_final"], atol=1e-2)

        # We now check the conservation of energy, the property that matters the most in HMC.
        energy = -neg_potential(q) + kinetic_energy(p)
        new_energy = -neg_potential(final_state.position) + kinetic_energy(
            final_state.momentum
        )
        self.assertAlmostEqual(energy, new_energy, delta=integrator["precision"])

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([3, 5])
    def test_esh_momentum_update(self, dims):
        """
        Test the numerically efficient version of the momentum update currently
        implemented match the naive implementation according to the Equation 16 in
        :cite:p:`robnik2023microcanonical`
        """
        step_size = 1e-3
        key0, key1 = jax.random.split(jax.random.key(62))
        gradient = jax.random.uniform(key0, shape=(dims,))
        momentum = jax.random.uniform(key1, shape=(dims,))
        momentum /= jnp.linalg.norm(momentum)

        # Navie implementation
        gradient_norm = jnp.linalg.norm(gradient)
        gradient_normalized = gradient / gradient_norm
        delta = step_size * gradient_norm / (dims - 1)
        next_momentum = (
            momentum
            + gradient_normalized
            * (
                jnp.sinh(delta)
                + jnp.dot(gradient_normalized, momentum * (jnp.cosh(delta) - 1))
            )
        ) / (jnp.cosh(delta) + jnp.dot(gradient_normalized, momentum * jnp.sinh(delta)))

        # Efficient implementation
        update_stable = self.variant(esh_dynamics_momentum_update_one_step)
        next_momentum1, *_ = update_stable(momentum, gradient, step_size, 1.0)
        np.testing.assert_array_almost_equal(next_momentum, next_momentum1)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        [
            "noneuclidean_leapfrog",
            "noneuclidean_mclachlan",
            "noneuclidean_yoshida",
        ],
    )
    def test_noneuclidean_integrator(self, integrator_name):
        integrator = algorithms[integrator_name]
        cov = jnp.asarray([[1.0, 0.5], [0.5, 2.0]])
        logdensity_fn = lambda x: stats.multivariate_normal.logpdf(
            x, jnp.zeros([2]), cov
        )

        step = self.variant(integrator["algorithm"](logdensity_fn))

        rng = jax.random.key(4263456)
        key0, key1 = jax.random.split(rng, 2)
        position_init = jax.random.normal(key0, (2,))
        momentum_init = generate_unit_vector(key1, position_init)
        step_size = 0.0001
        initial_state = integrators.new_integrator_state(
            logdensity_fn, position_init, momentum_init
        )

        final_state, kinetic_energy_change = jax.lax.scan(
            lambda state, _: step(state, step_size),
            initial_state,
            xs=None,
            length=15,
        )

        # Check the conservation of energy.
        potential_energy_change = final_state.logdensity - initial_state.logdensity
        energy_change = kinetic_energy_change[-1] + potential_energy_change
        self.assertAlmostEqual(energy_change, 0, delta=1e-3)


if __name__ == "__main__":
    absltest.main()
