"""Tests for :mod:`blackjax.mcmc.coupled_hmc`.

The tests are grouped so that the two groups fail for different reasons.

:class:`CoupledHMCMathTest` checks the mathematics the coupling relies on:
that externalising the momentum draw reproduces the metric's own momentum
law, that the reflection is a norm-preserving involution which leaves the
standard normal distribution intact, that the default direction really is the
inverse momentum map, and that the trajectory agrees with an independent
NumPy leapfrog.

:class:`CoupledHMCContractTest` checks the interface contract: that each
marginal's returned state *and* info equal what an independently reconstructed
uncoupled marginal produces from the same inputs, that the shared uniform
drives each marginal's own comparison rather than a shared verdict, and that
invalid input is refused rather than quietly turned into a Metropolis
rejection.

The uncoupled reference in the second group is rebuilt from BlackJAX
primitives (``trajectory.static_integration``, ``hmc.flip_momentum``,
``hmc_energy``, ``safe_energy_diff``) rather than by calling the module's own
prescribed helper a second time, so agreement is evidence about the module
instead of a tautology.
"""
import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import coupled_hmc, integrators, metrics, trajectory
from blackjax.mcmc.hmc import HMCState, flip_momentum
from blackjax.mcmc.proposal import safe_energy_diff
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.util import generate_gaussian_noise, run_inference_algorithm
from tests.fixtures import BlackJAXTest, smooth_skewed_logdensity

# ---------------------------------------------------------------------------
# Targets and metric payloads
# ---------------------------------------------------------------------------
_DIM = 4
_RANK = 2

# Deterministic controls that must land on a chosen side of an acceptance
# probability get their own fixed seed. `BlackJAXTest` seeds from today's date,
# which is right for a test whose outcome must hold for any draw and wrong for
# a control whose whole point is that `p_accept` is interior -- the same test
# would pass or fail depending on the day it ran. This is a local fixture, not
# a change to the shared key management.
_CONTROL_SEED = 20260907

# The reference comparison needs a fixture whose acceptance probabilities are
# interior under BOTH couplings, since the second marginal receives a reflected
# innovation and so has a different probability there. These values were chosen
# by scanning seeds and step sizes for that property; the test asserts it rather
# than trusting this comment.
_REFERENCE_SEED = 11
_REFERENCE_STEP_SIZES = (0.75, 0.90)

# `test_jit_vmap_and_scan`'s margin control needs both marginals interior;
# `_CONTROL_SEED` and `_REFERENCE_SEED` saturate one to 1.0 on that fixture.
_JIT_MARGIN_SEED = 10248


def _standard_normal_logdensity(x):
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return -0.5 * jnp.sum(flat**2)


def _tilted_logdensity(x):
    """A second, genuinely different target: anisotropic with a linear tilt."""
    flat, _ = jax.flatten_util.ravel_pytree(x)
    scales = jnp.arange(1, flat.size + 1, dtype=flat.dtype)
    return -0.5 * jnp.sum((flat / scales) ** 2) + jnp.sum(0.3 * flat)


def _stiff_logdensity(x):
    """Sharply curved, so a large step size makes rejection near-certain."""
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return -0.5 * jnp.sum((flat * 60.0) ** 2)


def _skewed_anisotropic_logdensity(x):
    """A second non-Gaussian target, scaled differently from the first.

    The reference comparison needs both marginals to have an acceptance
    probability strictly inside (0, 1), so that one uniform can be chosen to
    accept and another to reject. On the Gaussian targets first tried here, a
    coarse step-size grid moved from ``p_accept`` 1.0 to 0.0 between 0.70 and
    0.80 without landing inside, which made the fixture hard to pick. That is
    an observation about the grid that was scanned, NOT evidence that no
    interior interval exists: for a fixed leapfrog count the energy error
    varies continuously with the step size, so a narrower interval may well be
    there and simply was not sampled. This skewed target is a controlled
    choice that made an interior probability easy to find and pin, not a claim
    about Gaussian targets being unusable.
    """
    flat, _ = jax.flatten_util.ravel_pytree(x)
    scales = jnp.arange(1, flat.size + 1, dtype=flat.dtype)
    y = flat / scales
    return jnp.sum(y - jnp.exp(y))


def _moderately_stiff_logdensity(x):
    """Curved just enough that a moderate step size gives an intermediate
    acceptance probability, rather than a saturated 0 or 1."""
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return -0.5 * jnp.sum((flat * 2.0) ** 2)


def _metric_payloads(dtype):
    """Return one payload per supported metric kind, in ``dtype``."""
    rng = np.random.default_rng(20260907)
    diagonal = jnp.asarray(np.abs(rng.normal(size=_DIM)) + 0.5, dtype=dtype)
    root = rng.normal(size=(_DIM, _DIM))
    dense = jnp.asarray(root @ root.T + _DIM * np.eye(_DIM), dtype=dtype)
    basis, _ = np.linalg.qr(rng.normal(size=(_DIM, _RANK)))
    low_rank = metrics.LowRankInverseMassMatrix(
        jnp.asarray(np.abs(rng.normal(size=_DIM)) + 0.5, dtype=dtype),
        jnp.asarray(basis, dtype=dtype),
        jnp.asarray(np.abs(rng.normal(size=_RANK)) + 0.5, dtype=dtype),
    )
    return {"diagonal": diagonal, "dense": dense, "low_rank": low_rank}


def _x64():
    return jax.enable_x64()


# ---------------------------------------------------------------------------
# An uncoupled reference transition, rebuilt from BlackJAX primitives
# ---------------------------------------------------------------------------
def _oracle_marginal(
    logdensity_fn,
    inverse_mass_matrix,
    step_size,
    num_integration_steps,
    state,
    standard_normal,
    uniform,
    divergence_threshold=1000.0,
):
    """One uncoupled HMC transition from a prescribed ``(z, u)``.

    Deliberately does not touch ``coupled_hmc``: the trajectory, endpoint
    flip, energies and acceptance test are reassembled here from the library's
    own pieces so that agreement is an independent check.

    Returns ``(state, momentum, acceptance_rate, is_accepted, is_divergent,
    energy, proposal)`` -- everything the production ``HMCInfo`` carries.
    """
    metric = metrics.default_metric(inverse_mass_matrix)
    integrator = integrators.velocity_verlet(logdensity_fn, metric.kinetic_energy)
    build_trajectory = trajectory.static_integration(integrator)
    energy_fn = hmc_energy(metric.kinetic_energy)

    _, unravel = jax.flatten_util.ravel_pytree(state.position)
    momentum = metric.scale(
        state.position, unravel(standard_normal), inv=False, trans=False
    )
    start = integrators.IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )
    end = flip_momentum(build_trajectory(start, step_size, num_integration_steps))

    initial_energy = energy_fn(start)
    new_energy = energy_fn(end)
    delta_energy = safe_energy_diff(initial_energy, new_energy)
    is_divergent = -delta_energy > divergence_threshold

    p_accept = jnp.clip(jnp.exp(delta_energy), max=1)
    is_accepted = uniform < p_accept
    selected = jax.tree.map(lambda a, b: jnp.where(is_accepted, a, b), end, start)
    new_state = HMCState(
        selected.position, selected.logdensity, selected.logdensity_grad
    )
    return (
        new_state,
        momentum,
        p_accept,
        is_accepted,
        is_divergent,
        new_energy,
        end,
    )


def _reference_unit(state, first_mass, coupling):
    """The reflection unit, built independently of the module's own report.

    Uses the documented rule -- ``whitened_difference`` on the FIRST marginal's
    metric -- then normalises in NumPy, so a test comparing against it can see
    both a wrong reflection map and a policy wired to the wrong metric.
    """
    if coupling != "reflection":
        return None
    direction = np.asarray(
        coupled_hmc.whitened_difference(
            state.first, state.second, metrics.default_metric(first_mass)
        )
    )
    scale = np.max(np.abs(direction))
    direction = direction / (scale if scale else 1.0)
    norm = np.linalg.norm(direction)
    return jnp.asarray(direction / (norm if norm else 1.0), jnp.float64)


def _reflect_with_numpy(noise, unit, coupling):
    """Reconstruct the second marginal's innovation without the module's help.

    Deliberately NumPy: calling ``coupled_hmc._reflect`` here would make every
    comparison that uses this value circular for a defect in ``_reflect``.
    """
    if coupling != "reflection":
        return noise
    unit_array = np.asarray(unit)
    noise_array = np.asarray(noise)
    return jnp.asarray(
        noise_array - 2 * unit_array * float(noise_array @ unit_array),
        dtype=noise.dtype,
    )


# ---------------------------------------------------------------------------
# Mathematics and numerics
# ---------------------------------------------------------------------------
class CoupledHMCMathTest(BlackJAXTest):
    """Checks the algebra the coupling relies on."""

    @parameterized.parameters("diagonal", "dense", "low_rank")
    def test_scale_reproduces_the_momentum_law(self, kind):
        """``scale(z, inv=False, trans=False)`` is ``sample_momentum``'s map.

        This is the identity that lets the momentum draw be externalised
        without perturbing either marginal's law.
        """
        with _x64():
            payload = _metric_payloads(jnp.float64)[kind]
            metric = metrics.default_metric(payload)
            key = self.next_key()
            position = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)

            reference = metric.sample_momentum(key, position)
            noise = generate_gaussian_noise(key, position)
            via_scale = metric.scale(position, noise, inv=False, trans=False)

            chex.assert_trees_all_equal(reference, via_scale)

    def test_scale_reproduces_the_momentum_law_on_a_pytree(self):
        with _x64():
            position = {
                "a": jnp.ones((2,), jnp.float64),
                "b": jnp.ones((3,), jnp.float64),
            }
            metric = metrics.default_metric(jnp.ones((5,), jnp.float64))
            key = self.next_key()
            reference = metric.sample_momentum(key, position)
            via_scale = metric.scale(
                position, generate_gaussian_noise(key, position), inv=False, trans=False
            )
            chex.assert_trees_all_equal(reference, via_scale)

    def test_reflection_is_a_norm_preserving_involution(self):
        with _x64():
            direction = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            unit = coupled_hmc._reflection_unit(direction)
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)

            reflected = coupled_hmc._reflect(noise, unit)
            np.testing.assert_allclose(
                float(jnp.linalg.norm(reflected)),
                float(jnp.linalg.norm(noise)),
                rtol=1e-12,
            )
            chex.assert_trees_all_close(
                coupled_hmc._reflect(reflected, unit), noise, atol=1e-12
            )
            np.testing.assert_allclose(float(jnp.linalg.norm(unit)), 1.0, rtol=1e-12)

    def test_reflection_negates_the_component_along_the_unit(self):
        """The reflection must actually reflect.

        Norm preservation, involution and a unit-norm ``e`` are all satisfied
        by the identity map, so asserting only those leaves the map itself
        unconstrained. What distinguishes a reflection is that the component
        along ``e`` flips sign while the orthogonal complement is untouched.
        """
        with _x64():
            unit = coupled_hmc._reflection_unit(
                jnp.asarray([1.0, 2.0, -0.5, 0.25], jnp.float64)
            )
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            reflected = coupled_hmc._reflect(noise, unit)

            along_before = float(jnp.dot(unit, noise))
            along_after = float(jnp.dot(unit, reflected))
            self.assertNotAlmostEqual(along_before, 0.0, places=6)
            np.testing.assert_allclose(along_after, -along_before, rtol=1e-12)

            # The component orthogonal to `unit` is unchanged.
            orthogonal_before = noise - unit * along_before
            orthogonal_after = reflected - unit * along_after
            chex.assert_trees_all_close(orthogonal_after, orthogonal_before, atol=1e-12)

    def test_reflection_preserves_the_standard_normal(self):
        """Finite-sample check that the reflected innovations are still N(0, I).

        This is the *only* property reflection is claimed to have.
        """
        with _x64():
            unit = coupled_hmc._reflection_unit(
                jnp.asarray([1.0, 2.0, -0.5, 0.25], jnp.float64)
            )
            noise = jax.random.normal(self.next_key(), (40000, _DIM), jnp.float64)
            reflected = jax.vmap(coupled_hmc._reflect, in_axes=(0, None))(noise, unit)

            mean = np.asarray(reflected.mean(axis=0))
            covariance = np.cov(np.asarray(reflected), rowvar=False)
            # 40000 draws: the standard error on a mean is 5e-3, on a
            # covariance entry about 7e-3; these thresholds are ~6 sigma.
            np.testing.assert_allclose(mean, 0.0, atol=3e-2)
            np.testing.assert_allclose(covariance, np.eye(_DIM), atol=5e-2)

    @parameterized.parameters("diagonal", "dense", "low_rank")
    def test_whitened_difference_is_the_inverse_momentum_map(self, kind):
        r"""``whitened_difference`` returns :math:`A^{-1}(x_1-x_2)`.

        Verified through the identity that motivates it,
        :math:`\langle \Delta x, \partial K/\partial p\rangle = \langle
        A^{-1}\Delta x, z\rangle`, using autodiff on the library's own kinetic
        energy rather than a hand-written formula.
        """
        with _x64():
            payload = _metric_payloads(jnp.float64)[kind]
            metric = metrics.default_metric(payload)
            first = HMCState(
                jax.random.normal(self.next_key(), (_DIM,), jnp.float64), 0.0, None
            )
            second = HMCState(
                jax.random.normal(self.next_key(), (_DIM,), jnp.float64), 0.0, None
            )
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)

            direction = coupled_hmc.whitened_difference(first, second, metric)
            momentum = metric.scale(first.position, noise, inv=False, trans=False)
            velocity = jax.grad(metric.kinetic_energy)(momentum)
            delta = first.position - second.position

            np.testing.assert_allclose(
                float(jnp.dot(delta, velocity)),
                float(jnp.dot(direction, noise)),
                rtol=1e-10,
                atol=1e-12,
            )

    def test_zero_direction_is_the_identity(self):
        with _x64():
            unit = coupled_hmc._reflection_unit(jnp.zeros((_DIM,), jnp.float64))
            chex.assert_trees_all_equal(unit, jnp.zeros((_DIM,), jnp.float64))
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            chex.assert_trees_all_equal(coupled_hmc._reflect(noise, unit), noise)

    @parameterized.named_parameters(
        {"testcase_name": "tiny", "scale": 1e-300},
        {"testcase_name": "huge", "scale": 1e300},
        {"testcase_name": "ordinary", "scale": 1.0},
    )
    def test_reflection_unit_normalises_extreme_directions(self, scale):
        """A direction whose norm would overflow or underflow still normalises."""
        with _x64():
            direction = scale * jnp.asarray([1.0, -2.0, 0.5, 3.0], jnp.float64)
            unit = coupled_hmc._reflection_unit(direction)
            np.testing.assert_allclose(float(jnp.linalg.norm(unit)), 1.0, rtol=1e-12)
            # The unit vector is scale-free: it matches the ordinary case.
            reference = coupled_hmc._reflection_unit(
                jnp.asarray([1.0, -2.0, 0.5, 3.0], jnp.float64)
            )
            chex.assert_trees_all_close(unit, reference, atol=1e-12)

    @parameterized.named_parameters(
        {"testcase_name": "nan", "bad": float("nan")},
        {"testcase_name": "inf", "bad": float("inf")},
    )
    def test_reflection_unit_does_not_launder_a_non_finite_direction(self, bad):
        """A non-finite direction must stay visible, not become a plausible unit."""
        with _x64():
            direction = jnp.asarray([bad, 1.0, 0.0, 0.0], jnp.float64)
            unit = coupled_hmc._reflection_unit(direction)
            self.assertFalse(bool(jnp.all(jnp.isfinite(unit))))

    def test_low_rank_unit_eigenvalues_are_neutral(self):
        """``lam = 1`` columns leave the low-rank metric equal to its diagonal."""
        with _x64():
            sigma = jnp.asarray([1.0, 2.0, 0.5, 1.5], jnp.float64)
            basis, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(_DIM, _RANK)))
            padded = metrics.LowRankInverseMassMatrix(
                sigma,
                jnp.asarray(basis, jnp.float64),
                jnp.ones((_RANK,), jnp.float64),
            )
            low_rank_metric = metrics.default_metric(padded)
            diagonal_metric = metrics.default_metric(sigma**2)

            position = jnp.zeros((_DIM,), jnp.float64)
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            chex.assert_trees_all_close(
                low_rank_metric.scale(position, noise, inv=False, trans=False),
                diagonal_metric.scale(position, noise, inv=False, trans=False),
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# The interface contract
# ---------------------------------------------------------------------------
class CoupledHMCContractTest(BlackJAXTest):
    """Checks that coupling changes only the joint law of the inputs."""

    def _pair_state(self, dtype=jnp.float64, first_fn=None, second_fn=None, key=None):
        """Build a paired state.

        Pass ``key`` for a control that needs reproducible positions rather
        than a fresh draw per day.
        """
        first_fn = first_fn or _standard_normal_logdensity
        second_fn = second_fn or _standard_normal_logdensity
        position_key = self.next_key() if key is None else key
        first_key, second_key = jax.random.split(position_key)
        first_position = jax.random.normal(first_key, (_DIM,), dtype)
        second_position = jax.random.normal(second_key, (_DIM,), dtype)
        return coupled_hmc.init(
            (first_position, second_position), (first_fn, second_fn)
        )

    # -- each marginal is unchanged by coupling -----------------------------
    @parameterized.named_parameters(
        {"testcase_name": "sync_accept", "coupling": "synchronous", "accept": True},
        {"testcase_name": "sync_reject", "coupling": "synchronous", "accept": False},
        {"testcase_name": "reflect_accept", "coupling": "reflection", "accept": True},
        {"testcase_name": "reflect_reject", "coupling": "reflection", "accept": False},
    )
    def test_marginals_match_an_independent_reference(self, coupling, accept):
        """Both marginals' state AND info equal the uncoupled reference's.

        Heterogeneous on purpose: different targets, metrics, step sizes and
        integration counts, so a parameter leaking across the pair shows up.
        """
        with _x64():
            payloads = _metric_payloads(jnp.float64)
            # Two different non-Gaussian targets, two different metrics, two
            # different step sizes and integration counts. These settings are a
            # fixture found by scanning, chosen because BOTH reference
            # acceptance probabilities are
            # strictly interior under both couplings (about 0.53 for the first,
            # and 0.82 synchronous / 0.75 reflected for the second), which is
            # what lets one uniform accept and another reject.
            first_fn = smooth_skewed_logdensity
            second_fn = _skewed_anisotropic_logdensity
            first_mass, second_mass = payloads["dense"], payloads["low_rank"]
            step_sizes, integration_steps = _REFERENCE_STEP_SIZES, (3, 5)

            control_key, noise_key = jax.random.split(jax.random.key(_REFERENCE_SEED))
            first_key, second_key = jax.random.split(control_key)
            state = coupled_hmc.init(
                (
                    0.5 * jax.random.normal(first_key, (_DIM,), jnp.float64),
                    0.5 * jax.random.normal(second_key, (_DIM,), jnp.float64),
                ),
                (first_fn, second_fn),
            )
            step = coupled_hmc._build_prescribed_pair(
                (first_fn, second_fn),
                (first_mass, second_mass),
                step_sizes,
                integration_steps,
                integrators.velocity_verlet,
                1000.0,
                coupling,
                coupled_hmc.whitened_difference if coupling == "reflection" else None,
            )
            noise = jax.random.normal(noise_key, (_DIM,), jnp.float64)

            # Running only an accepting case would leave the rejection branch
            # uncompared, so a kernel that always returned the proposal would
            # still match the reference. The uniform is therefore chosen from
            # the REFERENCE's acceptance probabilities rather than hard-coded:
            # the states are seeded from the current date, so a fixed value is
            # not reliably on the side of the threshold it was picked for.
            _, probe_info = step(state, noise, jnp.asarray(0.0, jnp.float64))
            # The reference innovation is built entirely outside the module:
            # the direction from `whitened_difference` on the FIRST marginal's
            # metric, normalised and reflected in NumPy. Nothing is read back
            # from the production info, so this comparison stays blind to
            # neither a wrong `_reflect` nor a policy wired to the wrong metric
            # -- reading the reported unit back would let both be wrong
            # together and still agree.
            reference_unit = _reference_unit(state, first_mass, coupling)
            second_noise = _reflect_with_numpy(noise, reference_unit, coupling)
            probe_first = _oracle_marginal(
                first_fn,
                first_mass,
                step_sizes[0],
                integration_steps[0],
                state.first,
                noise,
                0.0,
            )[2]
            probe_second = _oracle_marginal(
                second_fn,
                second_mass,
                step_sizes[1],
                integration_steps[1],
                state.second,
                second_noise,
                0.0,
            )[2]
            # Both reference probabilities must be strictly INTERIOR, or this
            # control cannot force both outcomes and asserts nothing. A
            # probability that has rounded to exactly 1 makes the rejecting case
            # unreachable; the fixture is chosen so that does not happen, and
            # the test fails loudly rather than being skipped or clipped into
            # passing if it ever does.
            for label, probability in (
                ("first", float(probe_first)),
                ("second", float(probe_second)),
            ):
                self.assertGreater(
                    probability,
                    0.0,
                    f"{label} reference p_accept is 0, no accepting uniform exists",
                )
                self.assertLess(
                    probability,
                    1.0,
                    f"{label} reference p_accept rounded to 1, no rejecting "
                    "uniform exists -- choose a different control fixture rather "
                    "than skipping or clipping this case",
                )
            highest = max(float(probe_first), float(probe_second))
            # Reject with a margin, not at `highest`: oracle and module p_accept
            # can disagree by 1 ULP, making `uniform = highest` a coin flip.
            uniform = jnp.asarray(0.0 if accept else 0.5 * (highest + 1.0), jnp.float64)

            new_state, info = step(state, noise, uniform)
            # Confirm this case exercises the branch it is named for.
            self.assertEqual(bool(info.first.is_accepted), accept)
            self.assertEqual(bool(info.second.is_accepted), accept)

            # `second_noise` was already reconstructed above with NumPy rather
            # than by calling `coupled_hmc._reflect`, which would make the
            # comparison circular for exactly the defect of `_reflect` being
            # wrong. Confirm the unit did not change between the probe and the
            # real call, since the direction is measured from the incoming
            # states only and must not depend on the uniform.
            # The reported unit must match the independently built one, and
            # must not have changed between the probe call and this one, since
            # the direction depends on the incoming states alone.
            if coupling == "reflection":
                chex.assert_trees_all_close(
                    info.reflection_unit, reference_unit, atol=1e-12
                )
            else:
                # Synchronous reports an exactly zero unit, which is the
                # documented encoding of "the identity map".
                chex.assert_trees_all_equal(
                    info.reflection_unit, jnp.zeros((_DIM,), jnp.float64)
                )
            chex.assert_trees_all_equal(
                info.reflection_unit, probe_info.reflection_unit
            )
            for label, marginal_state, marginal_info, fn, mass, size, count, z in (
                (
                    "first",
                    state.first,
                    info.first,
                    first_fn,
                    first_mass,
                    step_sizes[0],
                    integration_steps[0],
                    noise,
                ),
                (
                    "second",
                    state.second,
                    info.second,
                    second_fn,
                    second_mass,
                    step_sizes[1],
                    integration_steps[1],
                    second_noise,
                ),
            ):
                (
                    oracle_state,
                    oracle_momentum,
                    oracle_rate,
                    oracle_accepted,
                    oracle_divergent,
                    oracle_energy,
                    oracle_proposal,
                ) = _oracle_marginal(fn, mass, size, count, marginal_state, z, uniform)

                produced = new_state.first if label == "first" else new_state.second
                with self.subTest(marginal=label):
                    chex.assert_trees_all_close(
                        produced, oracle_state, atol=1e-12, rtol=1e-12
                    )
                    chex.assert_trees_all_close(
                        marginal_info.momentum, oracle_momentum, atol=1e-12
                    )
                    chex.assert_trees_all_close(
                        marginal_info.acceptance_rate, oracle_rate, atol=1e-12
                    )
                    self.assertEqual(
                        bool(marginal_info.is_accepted), bool(oracle_accepted)
                    )
                    self.assertEqual(
                        bool(marginal_info.is_divergent), bool(oracle_divergent)
                    )
                    chex.assert_trees_all_close(
                        marginal_info.energy, oracle_energy, atol=1e-12
                    )
                    chex.assert_trees_all_close(
                        marginal_info.proposal, oracle_proposal, atol=1e-12
                    )
                    self.assertEqual(marginal_info.num_integration_steps, count)

    # -- planted defects: confirm these checks actually bite ----------------
    def test_crossing_the_caches_changes_the_transition(self):
        """Swapping the two marginals' caches must change the result.

        If it did not, the comparison against the independent reference could
        not detect a crossed cache, and the checks above would be vacuous.
        """
        with _x64():
            state = self._pair_state(
                first_fn=_standard_normal_logdensity, second_fn=_tilted_logdensity
            )
            crossed = coupled_hmc.CoupledHMCState(
                HMCState(
                    state.first.position,
                    state.second.logdensity,
                    state.second.logdensity_grad,
                ),
                state.second,
            )
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            uniform = jnp.asarray(0.3, jnp.float64)
            mass = jnp.ones((_DIM,), jnp.float64)

            honest, _, *_ = _oracle_marginal(
                _standard_normal_logdensity, mass, 0.1, 4, state.first, noise, uniform
            )
            crossed_result, _, *_ = _oracle_marginal(
                _standard_normal_logdensity, mass, 0.1, 4, crossed.first, noise, uniform
            )
            # The planted defect changes the transition, so the check bites.
            self.assertFalse(
                bool(jnp.allclose(honest.position, crossed_result.position, atol=1e-10))
            )

    def _intermediate_pair(self):
        """A pair whose second marginal accepts with probability strictly inside
        ``(0, 1)``.

        This matters: a control built on a second marginal with ``p_accept ==
        0`` would report "the two disagree" even if the uniform were ignored
        entirely, because a zero probability rejects for every uniform.  Only
        an intermediate probability makes the outcome actually depend on the
        shared variate.  The settings below give ``p_accept`` near 0.69.
        """
        first_fn = _standard_normal_logdensity
        second_fn = _moderately_stiff_logdensity
        position = jnp.full((_DIM,), 0.5, jnp.float64)
        state = coupled_hmc.init((position, position), (first_fn, second_fn))
        step = coupled_hmc._build_prescribed_pair(
            (first_fn, second_fn),
            (jnp.ones((_DIM,), jnp.float64), jnp.ones((_DIM,), jnp.float64)),
            (1e-6, 0.4),
            (3, 3),
            integrators.velocity_verlet,
            1000.0,
            "synchronous",
            None,
        )
        noise = jax.random.normal(jax.random.key(7), (_DIM,), jnp.float64)
        return state, step, noise

    def test_the_shared_uniform_drives_each_marginal_separately(self):
        """One uniform, two probabilities: the second flips where the first cannot.

        Sweeping the shared variate across the second marginal's acceptance
        probability makes its verdict change while the first marginal, which
        accepts with probability one, never does.  A pair that shared its
        *decision* rather than its uniform could not produce this pattern.
        """
        with _x64():
            state, step, noise = self._intermediate_pair()
            _, probe = step(state, noise, jnp.asarray(0.5, jnp.float64))
            second_probability = float(probe.second.acceptance_rate)
            self.assertGreater(second_probability, 0.05)
            self.assertLess(second_probability, 0.95)
            np.testing.assert_allclose(
                float(probe.first.acceptance_rate), 1.0, rtol=1e-9
            )

            _, below = step(
                state,
                noise,
                jnp.asarray(second_probability - 1e-6, jnp.float64),
            )
            _, above = step(
                state,
                noise,
                jnp.asarray(second_probability + 1e-6, jnp.float64),
            )

            self.assertTrue(bool(below.second.is_accepted))
            self.assertFalse(bool(above.second.is_accepted))
            # The rule itself, stated exactly, for both marginals on both sides:
            # each verdict is that marginal's own comparison against the shared
            # variate. This is what a pair sharing its *decision* could not do.
            for label, info in (
                ("below", below),
                ("above", above),
            ):
                for marginal in (info.first, info.second):
                    with self.subTest(side=label):
                        self.assertEqual(
                            bool(marginal.is_accepted),
                            float(info.uniform) < float(marginal.acceptance_rate),
                        )
            # The first marginal is unmoved by the same sweep.
            self.assertTrue(bool(below.first.is_accepted))
            self.assertTrue(bool(above.first.is_accepted))

    def test_a_zero_probability_marginal_rejects_whatever_the_partner_does(self):
        """The ``p_accept == 0`` case, kept as an endpoint rather than a control."""
        with _x64():
            state = self._pair_state(
                first_fn=_standard_normal_logdensity, second_fn=_stiff_logdensity
            )
            step = coupled_hmc._build_prescribed_pair(
                (_standard_normal_logdensity, _stiff_logdensity),
                (jnp.ones((_DIM,), jnp.float64), jnp.ones((_DIM,), jnp.float64)),
                (0.05, 0.9),
                (3, 3),
                integrators.velocity_verlet,
                1000.0,
                "synchronous",
                None,
            )
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            _, info = step(state, noise, jnp.asarray(0.5, jnp.float64))

            self.assertEqual(float(info.uniform), 0.5)
            self.assertTrue(bool(info.first.is_accepted))
            self.assertFalse(bool(info.second.is_accepted))
            self.assertGreater(
                float(info.first.acceptance_rate), float(info.second.acceptance_rate)
            )

    def test_each_marginal_keeps_its_own_cache(self):
        """Returned caches match each marginal's OWN target, not the other's."""
        with _x64():
            first_fn, second_fn = _standard_normal_logdensity, _tilted_logdensity
            state = self._pair_state(first_fn=first_fn, second_fn=second_fn)
            kernel = coupled_hmc.build_kernel()
            mass = jnp.ones((_DIM,), jnp.float64)
            new_state, _ = kernel(
                self.next_key(),
                state,
                (first_fn, second_fn),
                (0.1, 0.1),
                (mass, mass),
                (4, 4),
            )
            for produced, fn in (
                (new_state.first, first_fn),
                (new_state.second, second_fn),
            ):
                chex.assert_trees_all_close(
                    produced.logdensity, fn(produced.position), atol=1e-12
                )
                chex.assert_trees_all_close(
                    produced.logdensity_grad,
                    jax.grad(fn)(produced.position),
                    atol=1e-12,
                )
            # The two targets really do differ, so the check above has bite.
            self.assertNotAlmostEqual(
                float(first_fn(new_state.first.position)),
                float(second_fn(new_state.first.position)),
            )

    # -- the Metropolis rule and its endpoints ------------------------------
    def test_acceptance_probability_matches_ordinary_hmc(self):
        """The mathematical MH rule is ordinary HMC's.

        The acceptance *probability* must agree exactly.  The realised
        accept/reject draw is not compared: ordinary HMC uses
        ``jax.random.bernoulli`` and this kernel uses a uniform comparison, so
        no draw-for-draw parity exists or is claimed.
        """
        with _x64():
            mass = jnp.ones((_DIM,), jnp.float64)
            position = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)
            state = HMCState(
                position,
                _tilted_logdensity(position),
                jax.grad(_tilted_logdensity)(position),
            )
            noise = jax.random.normal(self.next_key(), (_DIM,), jnp.float64)

            _, prescribed_step = coupled_hmc._build_prescribed_marginal(
                _tilted_logdensity,
                mass,
                0.35,
                4,
                integrators.velocity_verlet,
                1000.0,
            )
            _, info = prescribed_step(state, noise, jnp.asarray(0.5, jnp.float64))
            (
                _,
                _,
                oracle_rate,
                *_,
            ) = _oracle_marginal(_tilted_logdensity, mass, 0.35, 4, state, noise, 0.5)
            chex.assert_trees_all_close(info.acceptance_rate, oracle_rate, atol=1e-14)

    @parameterized.named_parameters(
        {"testcase_name": "u_zero", "uniform": 0.0},
        {"testcase_name": "u_just_below_one", "uniform": None},
    )
    def test_certain_acceptance_accepts_at_both_uniform_endpoints(self, uniform):
        """``p_accept == 1`` accepts for every admissible uniform in [0, 1).

        Pinned at a stationary point (position = momentum = 0, energy
        difference exactly 0.0); a small step size alone left ~1e-16 sign
        noise that failed 128/200 seeded days.
        """
        with _x64():
            if uniform is None:
                uniform = float(jnp.nextafter(jnp.float64(1.0), jnp.float64(0.0)))
            mass = jnp.ones((_DIM,), jnp.float64)
            position = jnp.zeros((_DIM,), jnp.float64)
            state = HMCState(
                position,
                _standard_normal_logdensity(position),
                jax.grad(_standard_normal_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _standard_normal_logdensity,
                mass,
                1e-8,
                2,
                integrators.velocity_verlet,
                1000.0,
            )
            noise = jnp.zeros((_DIM,), jnp.float64)  # see docstring
            _, info = step(state, noise, jnp.asarray(uniform, jnp.float64))
            self.assertEqual(
                float(info.acceptance_rate),
                1.0,
                "p_accept must be bit-exact 1.0 at a stationary point, not merely close",
            )
            self.assertTrue(bool(info.is_accepted))

    def test_certain_rejection_rejects_at_the_zero_uniform(self):
        """``p_accept == 0`` (a divergence) rejects even at ``u == 0``."""
        with _x64():
            mass = jnp.ones((_DIM,), jnp.float64)
            position = jnp.full((_DIM,), 3.0, jnp.float64)
            state = HMCState(
                position,
                _stiff_logdensity(position),
                jax.grad(_stiff_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _stiff_logdensity, mass, 5.0, 8, integrators.velocity_verlet, 1000.0
            )
            noise = jax.random.normal(
                jax.random.key(_CONTROL_SEED), (_DIM,), jnp.float64
            )
            new_state, info = step(state, noise, jnp.asarray(0.0, jnp.float64))
            self.assertEqual(float(info.acceptance_rate), 0.0)
            self.assertFalse(bool(info.is_accepted))
            # The returned STATE must be the old one. Asserting only the flag
            # would leave the rejection branch itself unconstrained: a kernel
            # that reported `is_accepted=False` while returning the proposal
            # would pass.
            chex.assert_trees_all_equal(new_state.position, state.position)
            chex.assert_trees_all_equal(new_state.logdensity, state.logdensity)
            chex.assert_trees_all_equal(
                new_state.logdensity_grad, state.logdensity_grad
            )
            # And the proposal really did move, so the check above has bite.
            self.assertFalse(
                bool(jnp.allclose(info.proposal.position, state.position, atol=1e-10))
            )

    def test_prescribed_uniform_is_not_narrowed(self):
        """A float64 uniform must be refused by a float32 marginal, not cast.

        Casting would map a value strictly below one onto exactly ``1.0f``,
        which then loses a certain acceptance.  The refusal is what prevents
        that, so it is asserted directly -- including on the exact value that
        exhibits the collapse.
        """
        with _x64():
            just_below_one = jnp.nextafter(jnp.float64(1.0), jnp.float64(0.0))
            # The collapse this guards against is real:
            self.assertTrue(bool(just_below_one < 1))
            self.assertEqual(float(just_below_one.astype(jnp.float32)), 1.0)

            position = jnp.zeros((_DIM,), jnp.float32)
            state = HMCState(
                position,
                _standard_normal_logdensity(position),
                jax.grad(_standard_normal_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _standard_normal_logdensity,
                jnp.ones((_DIM,), jnp.float32),
                0.1,
                2,
                integrators.velocity_verlet,
                1000.0,
            )
            noise = jnp.zeros((_DIM,), jnp.float32)
            with self.assertRaisesRegex(TypeError, "no narrowing"):
                step(state, noise, just_below_one)

    @parameterized.named_parameters(
        {"testcase_name": "one", "uniform": 1.0},
        {"testcase_name": "above_one", "uniform": 1.5},
        {"testcase_name": "negative", "uniform": -0.1},
        {"testcase_name": "nan", "uniform": float("nan")},
    )
    def test_an_out_of_domain_uniform_is_refused_not_clipped(self, uniform):
        """An inadmissible uniform must raise, not become a rejection.

        Clipping or comparing it anyway would turn caller error into an
        ordinary Metropolis rejection, which is indistinguishable from a
        legitimate one and so would never be noticed.
        """
        with _x64():
            position = jnp.zeros((_DIM,), jnp.float64)
            state = HMCState(
                position,
                _standard_normal_logdensity(position),
                jax.grad(_standard_normal_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _standard_normal_logdensity,
                jnp.ones((_DIM,), jnp.float64),
                0.1,
                2,
                integrators.velocity_verlet,
                1000.0,
            )
            noise = jnp.zeros((_DIM,), jnp.float64)
            with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\)"):
                step(state, noise, jnp.asarray(uniform, jnp.float64))

    def test_a_non_finite_innovation_is_refused(self):
        with _x64():
            position = jnp.zeros((_DIM,), jnp.float64)
            state = HMCState(
                position,
                _standard_normal_logdensity(position),
                jax.grad(_standard_normal_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _standard_normal_logdensity,
                jnp.ones((_DIM,), jnp.float64),
                0.1,
                2,
                integrators.velocity_verlet,
                1000.0,
            )
            bad = jnp.asarray([jnp.inf, 0.0, 0.0, 0.0], jnp.float64)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                step(state, bad, jnp.asarray(0.5, jnp.float64))

    def test_prescribed_innovations_must_match_shape_and_dtype(self):
        with _x64():
            position = jnp.zeros((_DIM,), jnp.float64)
            state = HMCState(
                position,
                _standard_normal_logdensity(position),
                jax.grad(_standard_normal_logdensity)(position),
            )
            _, step = coupled_hmc._build_prescribed_marginal(
                _standard_normal_logdensity,
                jnp.ones((_DIM,), jnp.float64),
                0.1,
                2,
                integrators.velocity_verlet,
                1000.0,
            )
            good_noise = jnp.zeros((_DIM,), jnp.float64)
            good_uniform = jnp.asarray(0.5, jnp.float64)

            with self.assertRaisesRegex(ValueError, "flat vector matching"):
                step(state, jnp.zeros((_DIM + 1,), jnp.float64), good_uniform)
            with self.assertRaisesRegex(TypeError, "effective dtype"):
                step(state, jnp.zeros((_DIM,), jnp.float32), good_uniform)
            with self.assertRaisesRegex(ValueError, "must be a scalar"):
                step(state, good_noise, jnp.zeros((2,), jnp.float64))

    # -- input discipline ---------------------------------------------------
    def test_metric_kinds_outside_the_tested_set_are_refused(self):
        mass = jnp.ones((_DIM,))
        with self.assertRaisesRegex(TypeError, "callable"):
            coupled_hmc._check_metric_kind(lambda position: mass)
        with self.assertRaisesRegex(TypeError, "pre-built Metric"):
            coupled_hmc._check_metric_kind(metrics.default_metric(mass))

    @parameterized.parameters(
        "logdensity_fn", "step_size", "inverse_mass_matrix", "num_integration_steps"
    )
    def test_every_per_marginal_parameter_must_be_an_explicit_pair(self, name):
        mass = jnp.ones((_DIM,))
        arguments = {
            "logdensity_fn": (_standard_normal_logdensity,) * 2,
            "step_size": (0.1, 0.1),
            "inverse_mass_matrix": (mass, mass),
            "num_integration_steps": (4, 4),
        }
        arguments[name] = arguments[name][0]
        with self.assertRaisesRegex(TypeError, f"`{name}` must be an explicit"):
            coupled_hmc.as_top_level_api(**arguments)

    def test_paired_positions_must_agree(self):
        fns = (_standard_normal_logdensity,) * 2
        with self.assertRaisesRegex(ValueError, "one pytree structure"):
            coupled_hmc.init((jnp.ones((_DIM,)), {"a": jnp.ones((_DIM,))}), fns)
        with self.assertRaisesRegex(ValueError, "matching leaf shapes"):
            coupled_hmc.init((jnp.ones((_DIM,)), jnp.ones((_DIM + 1,))), fns)
        # Same structure and same total size, but split differently across
        # leaves: the shared flat innovation would be unravelled across
        # different leaf boundaries in each marginal.
        with self.assertRaisesRegex(ValueError, "matching leaf shapes"):
            coupled_hmc.init(
                (
                    {"a": jnp.ones((2,)), "b": jnp.ones((3,))},
                    {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                ),
                fns,
            )
        with _x64():
            with self.assertRaisesRegex(TypeError, "matching floating dtypes"):
                coupled_hmc.init(
                    (
                        jnp.ones((_DIM,), jnp.float64),
                        jnp.ones((_DIM,), jnp.float32),
                    ),
                    fns,
                )

    def test_coupling_and_direction_options_are_checked_at_build_time(self):
        with self.assertRaisesRegex(ValueError, "synchronous.*reflection"):
            coupled_hmc.build_kernel(coupling="antithetic")
        with self.assertRaisesRegex(ValueError, "does not use a direction"):
            coupled_hmc.build_kernel(
                coupling="synchronous", direction_fn=coupled_hmc.whitened_difference
            )

    def test_eager_validation_rejects_inadmissible_metrics(self):
        good = jnp.ones((_DIM,))
        with self.assertRaisesRegex(ValueError, "exactly symmetric"):
            coupled_hmc.validate_marginal_inputs(
                jnp.asarray([[1.0, 0.5], [0.4, 1.0]]), 0.1, 4
            )
        with self.assertRaisesRegex(ValueError, "positive definite"):
            coupled_hmc.validate_marginal_inputs(
                jnp.asarray([[1.0, 2.0], [2.0, 1.0]]), 0.1, 4
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            coupled_hmc.validate_marginal_inputs(jnp.asarray([1.0, -1.0]), 0.1, 4)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            coupled_hmc.validate_marginal_inputs(
                metrics.LowRankInverseMassMatrix(
                    jnp.ones((_DIM,)),
                    jnp.ones((_DIM, _RANK)),
                    jnp.asarray([2.0, 3.0]),
                ),
                0.1,
                4,
            )
        with self.assertRaisesRegex(ValueError, "step_size.*positive"):
            coupled_hmc.validate_marginal_inputs(good, -0.1, 4)
        with self.assertRaisesRegex(ValueError, "num_integration_steps.*positive"):
            coupled_hmc.validate_marginal_inputs(good, 0.1, 0)

    def test_warmup_output_is_accepted_without_hand_casting(self):
        """A real ``window_adaptation`` payload must drive the pair directly.

        Warmup returns ``step_size`` as a zero-dimensional JAX array, so a
        validator that insisted on a Python float would force every caller to
        cast ordinary BlackJAX output before use.
        """
        warmup = blackjax.window_adaptation(
            blackjax.hmc, _standard_normal_logdensity, num_integration_steps=5
        )
        (_, parameters), _ = warmup.run(
            self.next_key(), jnp.ones((_DIM,)), num_steps=120
        )
        step_size = parameters["step_size"]
        inverse_mass_matrix = parameters["inverse_mass_matrix"]
        # Precondition for this test to mean anything: it is not a plain float.
        self.assertFalse(isinstance(step_size, float))
        self.assertEqual(jnp.ndim(step_size), 0)

        algorithm = coupled_hmc.as_top_level_api(
            (_standard_normal_logdensity,) * 2,
            (step_size, step_size),
            (inverse_mass_matrix, inverse_mass_matrix),
            (5, 5),
        )
        state = algorithm.init((jnp.ones((_DIM,)), -jnp.ones((_DIM,))))
        new_state, _ = algorithm.step(self.next_key(), state)
        self.assertTrue(bool(jnp.all(jnp.isfinite(new_state.first.position))))

    def test_step_size_accepts_ordinary_scalar_forms(self):
        """One representative of each kind that reaches this in practice.

        Not every spelling: a bare Python float (weakly typed), a NumPy scalar,
        and a zero-dimensional JAX array, which is what ``window_adaptation``
        returns. Adding the remaining spellings would protect no distinct
        failure mode.
        """
        for value in (0.1, np.float32(0.1), jnp.asarray(0.1, jnp.float32)):
            with self.subTest(value=repr(value)):
                coupled_hmc.validate_marginal_inputs(jnp.ones((_DIM,)), value, 4)

    @parameterized.named_parameters(
        {"testcase_name": "bool", "value": True, "error": TypeError},
        {"testcase_name": "numpy_bool", "value": np.bool_(True), "error": TypeError},
        {"testcase_name": "complex", "value": 1 + 2j, "error": TypeError},
        {
            "testcase_name": "nonscalar",
            "value": jnp.asarray([0.1, 0.2]),
            "error": ValueError,
        },
        {"testcase_name": "nan", "value": float("nan"), "error": ValueError},
        {"testcase_name": "inf", "value": float("inf"), "error": ValueError},
    )
    def test_step_size_refuses_inadmissible_scalar_forms(self, value, error):
        """Accepting array forms must not also let bad values through."""
        with self.assertRaises(error):
            coupled_hmc.validate_marginal_inputs(jnp.ones((_DIM,)), value, 4)

    @parameterized.named_parameters(
        {"testcase_name": "python_int", "value": 4},
        {"testcase_name": "jax_scalar_i32", "value": jnp.asarray(4, jnp.int32)},
    )
    def test_integration_count_accepts_integer_scalar_forms(self, value):
        """The count contract: any concrete integer scalar, matching the engine.

        All three forms already run through ``static_integration`` unchanged,
        so the validator must not be stricter than the machinery it guards.
        """
        coupled_hmc.validate_marginal_inputs(jnp.ones((_DIM,)), 0.1, value)
        _, step = coupled_hmc._build_prescribed_marginal(
            _standard_normal_logdensity,
            jnp.ones((_DIM,)),
            0.1,
            value,
            integrators.velocity_verlet,
            1000.0,
        )
        position = jnp.ones((_DIM,))
        state = HMCState(
            position,
            _standard_normal_logdensity(position),
            jax.grad(_standard_normal_logdensity)(position),
        )
        _, info = step(state, jnp.zeros((_DIM,)), jnp.asarray(0.5, position.dtype))
        self.assertEqual(int(info.num_integration_steps), 4)

    @parameterized.named_parameters(
        {"testcase_name": "bool", "value": True, "error": TypeError},
        {"testcase_name": "float", "value": 4.0, "error": TypeError},
        {
            "testcase_name": "jax_float_scalar",
            "value": jnp.asarray(4.0, jnp.float32),
            "error": TypeError,
        },
        {
            "testcase_name": "nonscalar",
            "value": jnp.asarray([4, 5]),
            "error": ValueError,
        },
    )
    def test_integration_count_refuses_inadmissible_forms(self, value, error):
        """A floating count is refused, never silently truncated."""
        with self.assertRaises(error):
            coupled_hmc.validate_marginal_inputs(jnp.ones((_DIM,)), 0.1, value)

    def test_eager_validation_refuses_a_tracer(self):
        """Eager checks cannot read a traced value, and say so rather than pass."""

        def under_jit(value):
            coupled_hmc.validate_marginal_inputs(jnp.ones((_DIM,)), value, 4)
            return value

        with self.assertRaisesRegex(TypeError, "must be concrete"):
            jax.jit(under_jit)(jnp.asarray(0.1))

    def test_the_convenience_api_validates_and_build_kernel_does_not(self):
        """Eager checks belong to the convenience API, not to the traced path.

        ``build_kernel`` performs no eager numerical validation, which is what
        makes it usable with traced metrics; the price is that admissibility
        becomes the caller's responsibility there.
        """
        bad = jnp.asarray([[1.0, 0.5], [0.4, 1.0]])
        fns = (_standard_normal_logdensity,) * 2
        with self.assertRaisesRegex(ValueError, "exactly symmetric"):
            coupled_hmc.as_top_level_api(fns, (0.1, 0.1), (bad, bad), (2, 2))
        # Building a kernel does not inspect the metric numerically at all.
        self.assertTrue(callable(coupled_hmc.build_kernel()))

    # -- policy timing ------------------------------------------------------
    def test_direction_policy_sees_only_the_incoming_states(self):
        """``direction_fn`` is handed the pre-transition pair, nothing else."""
        with _x64():
            seen = {}

            def recording_direction_fn(first, second, first_metric):
                seen["first"] = first.position
                seen["second"] = second.position
                return coupled_hmc.whitened_difference(first, second, first_metric)

            state = self._pair_state()
            kernel = coupled_hmc.build_kernel(
                coupling="reflection", direction_fn=recording_direction_fn
            )
            mass = jnp.ones((_DIM,), jnp.float64)
            new_state, _ = kernel(
                self.next_key(),
                state,
                (_standard_normal_logdensity,) * 2,
                (0.2, 0.2),
                (mass, mass),
                (4, 4),
            )
            chex.assert_trees_all_equal(seen["first"], state.first.position)
            chex.assert_trees_all_equal(seen["second"], state.second.position)
            # And it is genuinely the pre-transition state, not the result.
            self.assertFalse(
                bool(jnp.allclose(seen["first"], new_state.first.position))
            )

    def test_common_normal_reports_the_first_marginal_s_innovation(self):
        """``common_normal`` is the innovation the FIRST marginal received.

        The field's docstring promises exactly that, and under reflection the
        two marginals receive different vectors, so reporting the second one
        would make the field contradict its own documentation with nothing to
        catch it.
        """
        with _x64():
            mass = jnp.ones((_DIM,), jnp.float64)
            first_position = jnp.ones((_DIM,), jnp.float64)
            second_position = -jnp.ones((_DIM,), jnp.float64)
            state = coupled_hmc.init(
                (first_position, second_position),
                (_standard_normal_logdensity,) * 2,
            )
            for coupling in ("synchronous", "reflection"):
                step = coupled_hmc._build_prescribed_pair(
                    (_standard_normal_logdensity,) * 2,
                    (mass, mass),
                    (0.2, 0.2),
                    (3, 3),
                    integrators.velocity_verlet,
                    1000.0,
                    coupling,
                    (
                        coupled_hmc.whitened_difference
                        if coupling == "reflection"
                        else None
                    ),
                )
                noise = jax.random.normal(
                    jax.random.key(_CONTROL_SEED), (_DIM,), jnp.float64
                )
                _, info = step(state, noise, jnp.asarray(0.3, jnp.float64))
                with self.subTest(coupling=coupling):
                    chex.assert_trees_all_equal(info.common_normal, noise)
                    if coupling == "reflection":
                        # The check has bite only if the second marginal really
                        # received something else.
                        second = _reflect_with_numpy(
                            noise, info.reflection_unit, coupling
                        )
                        self.assertFalse(bool(jnp.allclose(second, noise, atol=1e-10)))

    def test_direction_policy_receives_the_first_marginal_s_metric(self):
        """The default direction is documented as first-metric-based.

        Handing the policy the second marginal's metric would silently change
        which direction is reflected along while leaving every marginal
        property intact, so it has to be asserted directly.
        """
        with _x64():
            payloads = _metric_payloads(jnp.float64)
            first_mass, second_mass = payloads["dense"], payloads["low_rank"]
            seen = {}

            def recording_direction_fn(first, second, first_metric):
                seen["metric"] = first_metric
                return coupled_hmc.whitened_difference(first, second, first_metric)

            state = self._pair_state(key=jax.random.key(_CONTROL_SEED))
            step = coupled_hmc._build_prescribed_pair(
                (_standard_normal_logdensity,) * 2,
                (first_mass, second_mass),
                (0.2, 0.2),
                (3, 3),
                integrators.velocity_verlet,
                1000.0,
                "reflection",
                recording_direction_fn,
            )
            noise = jax.random.normal(
                jax.random.key(_CONTROL_SEED), (_DIM,), jnp.float64
            )
            step(state, noise, jnp.asarray(0.3, jnp.float64))

            # Identify the metric by what it computes, not by object identity.
            probe = jax.random.normal(
                jax.random.key(_CONTROL_SEED + 1), (_DIM,), jnp.float64
            )
            observed = seen["metric"].scale(
                state.first.position, probe, inv=False, trans=False
            )
            expected_first = metrics.default_metric(first_mass).scale(
                state.first.position, probe, inv=False, trans=False
            )
            expected_second = metrics.default_metric(second_mass).scale(
                state.first.position, probe, inv=False, trans=False
            )
            chex.assert_trees_all_close(observed, expected_first, atol=1e-12)
            # The two metrics must actually differ, or this proves nothing.
            self.assertFalse(
                bool(jnp.allclose(expected_first, expected_second, atol=1e-8))
            )

    def test_synchronous_coupling_keeps_identical_states_identical(self):
        """Two chains started at the same point never separate under sync."""
        with _x64():
            position = jnp.ones((_DIM,), jnp.float64)
            state = coupled_hmc.init(
                (position, position), (_standard_normal_logdensity,) * 2
            )
            kernel = coupled_hmc.build_kernel()
            mass = jnp.ones((_DIM,), jnp.float64)
            for _ in range(5):
                state, info = kernel(
                    self.next_key(),
                    state,
                    (_standard_normal_logdensity,) * 2,
                    (0.3, 0.3),
                    (mass, mass),
                    (4, 4),
                )
                chex.assert_trees_all_equal(state.first.position, state.second.position)
                chex.assert_trees_all_equal(
                    info.reflection_unit, jnp.zeros((_DIM,), jnp.float64)
                )

    def test_reflection_of_identical_states_degenerates_to_synchronous(self):
        """A zero position difference gives a zero direction, i.e. the identity."""
        with _x64():
            position = jnp.ones((_DIM,), jnp.float64)
            state = coupled_hmc.init(
                (position, position), (_standard_normal_logdensity,) * 2
            )
            kernel = coupled_hmc.build_kernel(coupling="reflection")
            mass = jnp.ones((_DIM,), jnp.float64)
            new_state, info = kernel(
                self.next_key(),
                state,
                (_standard_normal_logdensity,) * 2,
                (0.3, 0.3),
                (mass, mass),
                (4, 4),
            )
            chex.assert_trees_all_equal(
                info.reflection_unit, jnp.zeros((_DIM,), jnp.float64)
            )
            chex.assert_trees_all_equal(
                new_state.first.position, new_state.second.position
            )

    # -- transformations and protocol --------------------------------------
    def test_jit_vmap_and_scan(self):
        mass = jnp.ones((_DIM,))
        algorithm = coupled_hmc.as_top_level_api(
            (_standard_normal_logdensity, _tilted_logdensity),
            (0.2, 0.2),
            (mass, mass),
            (4, 4),
            coupling="reflection",
        )
        state = algorithm.init((jnp.ones((_DIM,)), -jnp.ones((_DIM,))))
        key = self.next_key()

        eager_state, eager_info = algorithm.step(key, state)
        jitted_state, jitted_info = jax.jit(algorithm.step)(key, state)
        # atol=1e-4: ~35x measured float32 eager/jit drift (state 2.86e-6, rate 3.82e-6).
        chex.assert_trees_all_close(eager_state, jitted_state, atol=1e-4)
        chex.assert_trees_all_close(
            eager_info.first.acceptance_rate,
            jitted_info.first.acceptance_rate,
            atol=1e-4,
        )
        chex.assert_trees_all_close(
            eager_info.second.acceptance_rate,
            jitted_info.second.acceptance_rate,
            atol=1e-4,
        )
        chex.assert_trees_all_equal_structs(eager_state, jitted_state)
        chex.assert_trees_all_equal_structs(eager_info, jitted_info)

        # Decisions exercised through the same machinery, but with the uniform
        # held far from p_accept. Interiority is asserted, not trusted.
        margin_step = coupled_hmc._build_prescribed_pair(
            (_standard_normal_logdensity, _tilted_logdensity),
            (mass, mass),
            (0.2, 0.2),
            (4, 4),
            integrators.velocity_verlet,
            1000.0,
            "reflection",
            coupled_hmc.whitened_difference,
        )
        margin_noise = jax.random.normal(jax.random.key(_JIT_MARGIN_SEED), (_DIM,))
        # acceptance_rate is independent of the uniform; this only reads p_accept.
        _, rate_probe_info = margin_step(state, margin_noise, jnp.asarray(0.5))
        first_p_accept = float(rate_probe_info.first.acceptance_rate)
        second_p_accept = float(rate_probe_info.second.acceptance_rate)
        for label, probability in (
            ("first", first_p_accept),
            ("second", second_p_accept),
        ):
            self.assertGreater(
                probability, 0.0, f"{label} p_accept is 0, no accepting uniform exists"
            )
            self.assertLess(
                probability,
                1.0,
                f"{label} p_accept rounded to 1, no rejecting uniform exists -- "
                "choose a different `_JIT_MARGIN_SEED`",
            )
        highest = max(first_p_accept, second_p_accept)
        # Same margin construction as reflect_reject above.
        for uniform, expect_accepted in (
            (jnp.asarray(0.0), True),
            (jnp.asarray(0.5 * (highest + 1.0)), False),
        ):
            eager_margin_state, eager_margin_info = margin_step(
                state, margin_noise, uniform
            )
            jitted_margin_state, jitted_margin_info = jax.jit(margin_step)(
                state, margin_noise, uniform
            )
            for name, info in (
                ("eager", eager_margin_info),
                ("jitted", jitted_margin_info),
            ):
                self.assertEqual(
                    bool(info.first.is_accepted),
                    expect_accepted,
                    f"{name} first marginal at uniform={float(uniform)!r}",
                )
                self.assertEqual(
                    bool(info.second.is_accepted),
                    expect_accepted,
                    f"{name} second marginal at uniform={float(uniform)!r}",
                )
            chex.assert_trees_all_close(
                eager_margin_state, jitted_margin_state, atol=1e-12
            )

        def body(carry, step_key):
            new_state, _ = algorithm.step(step_key, carry)
            return new_state, new_state.first.position

        keys = jax.random.split(self.next_key(), 20)
        final, history = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
        self.assertEqual(history.shape, (20, _DIM))
        self.assertTrue(bool(jnp.all(jnp.isfinite(final.first.position))))

        def run_one(first_position, second_position, batch_key):
            batch_state = algorithm.init((first_position, second_position))
            return algorithm.step(batch_key, batch_state)[0].first.position

        batched = jax.vmap(run_one)(
            jnp.ones((3, _DIM)),
            -jnp.ones((3, _DIM)),
            jax.random.split(self.next_key(), 3),
        )
        self.assertEqual(batched.shape, (3, _DIM))

    def test_conforms_to_the_sampling_algorithm_protocol(self):
        mass = jnp.ones((_DIM,))
        algorithm = coupled_hmc.as_top_level_api(
            (_standard_normal_logdensity,) * 2, (0.2, 0.2), (mass, mass), (4, 4)
        )
        self.assertIsInstance(algorithm, SamplingAlgorithm)

        state = algorithm.init((jnp.ones((_DIM,)), -jnp.ones((_DIM,))))
        self.assertIsInstance(state, coupled_hmc.CoupledHMCState)

        new_state, info = algorithm.step(self.next_key(), state)
        self.assertIsInstance(new_state, coupled_hmc.CoupledHMCState)
        self.assertIsInstance(info, coupled_hmc.CoupledHMCInfo)
        # The pair really is a pytree, so the generic runner drives it.
        final, history = run_inference_algorithm(
            self.next_key(),
            algorithm,
            num_steps=10,
            initial_position=(jnp.ones((_DIM,)), -jnp.ones((_DIM,))),
        )
        self.assertIsInstance(final, coupled_hmc.CoupledHMCState)
        self.assertEqual(history[0].first.position.shape, (10, _DIM))

        # A pytree position drives the same path: the pair is itself a pytree,
        # which is what makes the generic runner work, so this is the one
        # distinct thing the old separate pytree smoke test asserted.
        tree_position = {"a": jnp.ones((2,)), "b": jnp.zeros((2,))}
        tree_algorithm = coupled_hmc.as_top_level_api(
            (_standard_normal_logdensity,) * 2,
            (0.2, 0.2),
            (jnp.ones((4,)), jnp.ones((4,))),
            (4, 4),
            coupling="reflection",
        )
        tree_state = tree_algorithm.init(
            (tree_position, jax.tree.map(lambda leaf: leaf - 1.0, tree_position))
        )
        stepped, tree_info = tree_algorithm.step(self.next_key(), tree_state)
        chex.assert_trees_all_equal_structs(stepped.first.position, tree_position)
        self.assertEqual(tree_info.common_normal.shape, (4,))


if __name__ == "__main__":
    absltest.main()
