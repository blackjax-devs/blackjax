"""Tests for the composed step-size x trajectory-length GIST sampler
(``h`` then ``L``, :mod:`blackjax.mcmc.composed._kernel`).

Correctness evidence in this file is layered, per the module's own design
history: a one-step exact-target paired-flux invariance oracle
(``OneStepPairedFluxOracleTest``) is the *primary* gate -- it tests
``E[f(Z1) - f(Z0)] = 0`` from independent exact-target draws after exactly
one transition, with no burn-in or long-run convergence to hide a real
invariance bug behind. ``MutationControlTest`` runs the same oracle against
four single-factor ablations of the acceptance-ratio construction (each
built directly against the shipped seam, not a hand transcription) to
confirm the oracle actually has power to catch what it claims to catch.
Everything else (regression coverage, edge cases, a light end-to-end
sampling smoke) is secondary, structural coverage.

CI note: run with ``--benchmark-disable`` when parallelizing under xdist
(e.g. ``-n 2``) -- a bare ``-n 2`` run hits an ``INTERNALERROR`` from
pytest-benchmark x xdist under this project's ``filterwarnings = error``
(the same masking-bug family as the probdiffeq migration saga). Not a code
issue in this module; the documented blackjax test command already
includes ``--benchmark-disable``.
"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
from absl.testing import absltest, parameterized
from scipy import stats as sstats

import blackjax
import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.composed import _kernel as gist_composed
from blackjax.mcmc.composed import _seam as gist
from blackjax.mcmc.composed import step_size, trajectory_length
from tests.fixtures import BlackJAXTest, neal_funnel_logdensity, std_normal_logdensity


def run_chain(algo, position, key, n):
    state = algo.init(position)

    def body(s, k):
        s, info = algo.step(k, s)
        return s, (s.position, info)

    _, (positions, infos) = jax.lax.scan(body, state, jax.random.split(key, n))
    return positions, infos


def _dict_position_logdensity(x):
    """A dict-pytree-position standard normal, for the pytree regression case."""
    return -0.5 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))


# ---------------------------------------------------------------------------
# One-step paired-flux oracle: shared machinery.
#
# Two exact-drawable target classes, chosen so no single one is sufficient on
# its own: an isotropic (whitened) Gaussian and Neal's funnel (this module's
# own `tests.fixtures.neal_funnel_logdensity`, y ~ N(0, 3**2), x | y ~
# N(0, exp(y) I_3)). Both ship a pre-registered, fixed functional family
# that sees the JOINT law -- including the funnel's variance-direction
# functionals (`exp(-y) ||x||^2` and its cross moment with `y`), not just
# marginal location/scale.
# ---------------------------------------------------------------------------

_D_GAUSSIAN = 4


def _gaussian_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _gaussian_exact_draws(rng_key, n):
    return jax.random.normal(rng_key, (n, _D_GAUSSIAN))


def _gaussian_functionals(x):
    return {
        "x0": x[..., 0],
        "x0sq": x[..., 0] ** 2,
        "x1sq": x[..., 1] ** 2,
        "x0x1": x[..., 0] * x[..., 1],
        "x1x2": x[..., 1] * x[..., 2],
        "r2": jnp.sum(x**2, axis=-1),
    }


def _funnel_exact_draws(rng_key, n, k=3, sigma=3.0):
    """Independent exact draws from ``tests.fixtures.neal_funnel_logdensity``
    (``sigma=3`` matches that fixture's own fixed neck scale)."""
    key_y, key_x = jax.random.split(rng_key)
    y = sigma * jax.random.normal(key_y, (n,))
    x = jax.random.normal(key_x, (n, k)) * jnp.exp(0.5 * y)[:, None]
    return jnp.concatenate([y[:, None], x], axis=1)


def _funnel_functionals(x):
    y = x[..., 0]
    z = x[..., 1:]
    r2 = jnp.exp(-y) * jnp.sum(z**2, axis=-1)
    return {
        "y": y,
        "y2": y**2,
        "r2": r2,  # the variance-direction functional
        "y_r2": y * r2,  # cross moment
        "z0sq": jnp.exp(-y) * z[..., 0] ** 2,
    }


# name -> (logdensity_fn, exact_draws_fn, functionals_fn, dim, a reasonable initial_step_size)
_TARGETS = {
    "gaussian": (
        _gaussian_logdensity,
        _gaussian_exact_draws,
        _gaussian_functionals,
        _D_GAUSSIAN,
        0.5,
    ),
    "funnel": (
        neal_funnel_logdensity,
        _funnel_exact_draws,
        _funnel_functionals,
        4,
        0.15,
    ),
}


def _paired_z(before, after):
    delta = np.asarray(after) - np.asarray(before)
    n = delta.shape[0]
    se = delta.std(ddof=1) / np.sqrt(n)
    return float(delta.mean() / se) if se > 0 else float("nan")


def _family_threshold(m, alpha=0.05):
    """Bonferroni-corrected two-sided z threshold for a pre-registered
    family of ``m`` simultaneous paired-flux functionals, at overall
    false-positive rate ``alpha``."""
    return float(sstats.norm.ppf(1 - alpha / (2 * m)))


def _paired_flux_check(target, algo, n, seed):
    _, draws_fn, functionals_fn, _, _ = _TARGETS[target]
    key_init, key_step = jax.random.split(jax.random.key(seed))
    x0 = draws_fn(key_init, n)
    keys = jax.random.split(key_step, n)

    def one(x, k):
        state = algo.init(x)
        new_state, info = algo.step(k, state)
        return new_state.position, info

    x1, info = jax.vmap(one)(x0, keys)
    f0, f1 = functionals_fn(x0), functionals_fn(x1)
    zs = {name: _paired_z(f0[name], f1[name]) for name in f0}
    threshold = _family_threshold(len(zs))
    return zs, threshold, float(jnp.mean(info.is_accepted))


# ---------------------------------------------------------------------------
# Mutation controls: single-factor ablations of the acceptance-ratio
# construction, built directly against the shipped seam (reuses the real,
# shipped `_tuning_parameter_fn` unmodified -- only `apply_fn`'s acceptance
# construction is mutated). Each mirrors `gist_composed._apply_fn` line for
# line except for exactly the one ablated factor named by `variant`, so each
# is an honest single-factor diff against the nominal kernel, not an
# independent reimplementation that could hide a second, compensating bug.
# ---------------------------------------------------------------------------

_MUTANTS = (
    "drop_h_reversibility",
    "drop_width_ratio",
    "drop_membership",
    "wrong_h_in_reverse",
)


def _mutant_apply_fn(
    integrator,
    initial_step_size,
    h_selector_trial_length,
    max_search_steps,
    criterion,
    max_num_steps,
    path_fraction,
    variant,
):
    selector = step_size.step_size_selector(
        integrator,
        h_selector_trial_length,
        initial_step_size,
        max_search_steps,
        criterion,
    )

    def apply_fn(state, alpha, aux, logdensity_fn, metric):
        a, b, step_index, num_steps = alpha
        h, h_search_exhausted_forward, rollout = aux
        forward_uturn = rollout.num_steps_to_uturn

        proposal_state = jax.tree.map(
            lambda buf: jax.lax.dynamic_index_in_dim(
                buf, num_steps - 1, axis=0, keepdims=False
            ),
            rollout.states,
        )
        proposal_state = hmc.flip_momentum(proposal_state)

        reverse_step_index, h_search_exhausted_reverse = selector(
            proposal_state, a, b, logdensity_fn, metric
        )
        h_search_exhausted = h_search_exhausted_forward | h_search_exhausted_reverse
        is_h_reversible = reverse_step_index == step_index

        # wrong_h_in_reverse: evaluate the reverse rollout at the *base* h0
        # instead of the forward-selected h -- violates "both endpoints use
        # the same, forward-selected h" (module docstring).
        reverse_h = initial_step_size if variant == "wrong_h_in_reverse" else h
        reverse_uturn_fn = trajectory_length.num_steps_to_uturn(
            integrator, reverse_h, metric, max_num_steps
        )
        reverse_uturn = reverse_uturn_fn(proposal_state, logdensity_fn)

        _, width_forward = trajectory_length._step_distribution(
            forward_uturn, path_fraction
        )
        lo_reverse, width_reverse = trajectory_length._step_distribution(
            reverse_uturn, path_fraction
        )
        is_in_reverse_interval = (num_steps >= lo_reverse) & (
            num_steps <= reverse_uturn
        )

        if variant == "drop_h_reversibility":
            # Never gate on the h-selection reversibility indicator.
            is_valid = is_in_reverse_interval & jnp.logical_not(h_search_exhausted)
        elif variant == "drop_membership":
            # Never gate on the L-interval membership indicator.
            is_valid = is_h_reversible & jnp.logical_not(h_search_exhausted)
        else:
            is_valid = (
                is_h_reversible
                & is_in_reverse_interval
                & jnp.logical_not(h_search_exhausted)
            )

        if variant == "drop_width_ratio":
            log_width_ratio = jnp.asarray(0.0)
        else:
            log_width_ratio = jnp.log(width_forward.astype(jnp.float32)) - jnp.log(
                width_reverse.astype(jnp.float32)
            )

        log_tuning_density_ratio = jnp.where(is_valid, log_width_ratio, -jnp.inf)
        extra_info = gist_composed._ComposedExtra(
            num_integration_steps=num_steps,
            step_size=h,
            reverse_step_index=reverse_step_index,
            h_search_exhausted=h_search_exhausted,
            num_steps_to_uturn_forward=forward_uturn,
            num_steps_to_uturn_reverse=reverse_uturn,
            is_no_return_rejected=jnp.logical_not(is_in_reverse_interval),
        )
        return proposal_state, log_tuning_density_ratio, extra_info

    return apply_fn


_KERNEL_DEFAULTS = dict(
    integrator=integrators.velocity_verlet,
    divergence_threshold=1000.0,
    criterion="symmetric",
    max_search_steps=10,
    h_selector_trial_length=1,
    max_num_steps=64,
    path_fraction=0.2849,
)


def _composed_algorithm(variant, logdensity_fn, initial_step_size, inverse_mass_matrix):
    """A `SamplingAlgorithm` for the nominal kernel (``variant="nominal"``)
    or one of the `_MUTANTS`, built directly against the shipped seam."""
    kw = _KERNEL_DEFAULTS
    if variant == "nominal":
        kernel = gist_composed.build_kernel(
            kw["integrator"],
            kw["divergence_threshold"],
            kw["criterion"],
            kw["max_search_steps"],
            kw["h_selector_trial_length"],
            kw["path_fraction"],
            kw["max_num_steps"],
        )

        def step_fn(rng_key, state):
            return kernel(
                rng_key, state, logdensity_fn, initial_step_size, inverse_mass_matrix
            )

    else:
        tuning_parameter_fn = gist_composed._tuning_parameter_fn(
            kw["integrator"],
            initial_step_size,
            kw["h_selector_trial_length"],
            kw["max_search_steps"],
            kw["criterion"],
            kw["max_num_steps"],
            kw["path_fraction"],
        )
        apply_fn = _mutant_apply_fn(
            kw["integrator"],
            initial_step_size,
            kw["h_selector_trial_length"],
            kw["max_search_steps"],
            kw["criterion"],
            kw["max_num_steps"],
            kw["path_fraction"],
            variant,
        )

        def step_fn(rng_key, state):
            new_state, info, _ = gist._step(
                rng_key,
                state,
                logdensity_fn,
                tuning_parameter_fn,
                apply_fn,
                inverse_mass_matrix,
                kw["divergence_threshold"],
            )
            return new_state, info

    def init_fn(position, rng_key=None):
        del rng_key
        return gist.init(position, logdensity_fn)

    return SamplingAlgorithm(init_fn, step_fn)


class OneStepPairedFluxOracleTest(chex.TestCase):
    """The primary correctness gate for this module (see the file
    docstring). A fixed seed (not date-rotated) is used so the measured
    z-scores documented next to each assertion are exactly reproducible.
    """

    SEED = 0
    N = 200_000

    def _assert(self, target, algo, expect_detected, label):
        zs, threshold, acceptance = _paired_flux_check(target, algo, self.N, self.SEED)
        max_abs_z = max(abs(z) for z in zs.values())
        rounded_zs = {k: round(v, 2) for k, v in zs.items()}
        msg = (
            f"{label} on {target}: max|z|={max_abs_z:.3f} threshold={threshold:.3f} "  # noqa: E231
            f"acceptance={acceptance:.3f} zs={rounded_zs}"  # noqa: E231
        )
        if expect_detected:
            self.assertGreater(max_abs_z, threshold, msg)
        else:
            self.assertLess(max_abs_z, threshold, msg)

    def test_nominal_passes_on_gaussian(self):
        # Measured: max|z|=0.906 (threshold 2.638), acceptance 57.3%.
        _, _, _, d, h0 = _TARGETS["gaussian"]
        algo = _composed_algorithm("nominal", _gaussian_logdensity, h0, jnp.ones(d))
        self._assert("gaussian", algo, expect_detected=False, label="nominal")

    def test_nominal_passes_on_funnel(self):
        # Measured: max|z|=1.883 (threshold 2.576), acceptance 34.2%.
        _, _, _, d, h0 = _TARGETS["funnel"]
        algo = _composed_algorithm("nominal", neal_funnel_logdensity, h0, jnp.ones(d))
        self._assert("funnel", algo, expect_detected=False, label="nominal")

    def test_hmc_negative_control_gaussian(self):
        # Ordinary HMC on the same oracle: confirms the oracle itself is not
        # inherently biased. Measured: max|z|=1.985 (threshold 2.638).
        algo = blackjax.hmc(
            _gaussian_logdensity,
            step_size=0.3,
            inverse_mass_matrix=jnp.ones(_D_GAUSSIAN),
            num_integration_steps=10,
        )
        self._assert("gaussian", algo, expect_detected=False, label="hmc")

    def test_hmc_negative_control_funnel(self):
        # Measured: max|z|=1.283 (threshold 2.576).
        algo = blackjax.hmc(
            neal_funnel_logdensity,
            step_size=0.05,
            inverse_mass_matrix=jnp.ones(4),
            num_integration_steps=10,
        )
        self._assert("funnel", algo, expect_detected=False, label="hmc")


class MutationControlTest(parameterized.TestCase):
    """Each single-factor ablation must be DETECTED (fail the paired-flux
    oracle) -- run on both target classes, since factor detectability is
    class-dependent (a mutation that biases a variance-direction functional
    may be invisible to an isotropic target, and vice versa).

    Measured max|z| at N=200_000, seed=0 (family-calibrated threshold in
    parentheses): drop_h_reversibility gaussian=19.78 (2.64) / funnel=47.70
    (2.58); drop_width_ratio gaussian=38.53 (2.64) / funnel=19.85 (2.58);
    drop_membership gaussian=7.04 (2.64) / funnel=11.22 (2.58);
    wrong_h_in_reverse gaussian=4.02 (2.64) / funnel=4.30 (2.58). Every
    mutant is detected on BOTH target classes at this sample size, though
    wrong_h_in_reverse's margin is much narrower than the other three --
    consistent with detectability being class- and sample-size-dependent in
    general (a smaller N, or a different functional family, could plausibly
    miss it on one class alone).
    """

    SEED = 0
    N = 200_000

    @parameterized.named_parameters(
        *[
            (f"{variant}_{target}", variant, target)
            for variant in _MUTANTS
            for target in ("gaussian", "funnel")
        ]
    )
    def test_mutant_is_detected(self, variant, target):
        logdensity_fn, _, _, d, h0 = _TARGETS[target]
        algo = _composed_algorithm(variant, logdensity_fn, h0, jnp.ones(d))
        zs, threshold, acceptance = _paired_flux_check(target, algo, self.N, self.SEED)
        max_abs_z = max(abs(z) for z in zs.values())
        self.assertGreater(
            max_abs_z,
            threshold,
            f"{variant} on {target} should have been detected: max|z|={max_abs_z:.3f} "  # noqa: E231
            f"threshold={threshold:.3f} acceptance={acceptance:.3f} zs={zs}",  # noqa: E231
        )


class InitTest(chex.TestCase):
    def test_init_stores_position_and_gradients(self):
        position = jnp.array([1.0, 2.0])
        state = gist_composed.init(position, std_normal_logdensity)
        self.assertIsInstance(state, gist.GISTState)
        np.testing.assert_allclose(state.position, position)
        np.testing.assert_allclose(
            float(state.logdensity), float(std_normal_logdensity(position))
        )


class SingleStepTest(chex.TestCase):
    def test_step_shapes_and_types(self):
        algo = blackjax.gist_composed(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.5,
        )
        state = algo.init(jnp.zeros(3))
        new_state, info = algo.step(jax.random.key(0), state)
        self.assertIsInstance(new_state, gist.GISTState)
        self.assertIsInstance(info, gist_composed.GISTComposedInfo)
        self.assertEqual(new_state.position.shape, (3,))
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(std_normal_logdensity(new_state.position)),
            atol=1e-5,
        )

    def test_jit(self):
        algo = blackjax.gist_composed(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.3,
        )
        state = algo.init(jnp.zeros(3))
        new_state, _ = jax.jit(algo.step)(jax.random.key(0), state)
        self.assertEqual(new_state.position.shape, (3,))

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(criterion="not-a-criterion")


class CompilationTest(chex.TestCase):
    def test_no_excess_retracing(self):
        """The logdensity should compile at most 5 times: init, plus 4
        within one kernel trace -- the forward step-size selector, the
        forward no-U-turn rollout, the reverse step-size reversibility
        re-check, and the reverse no-U-turn rollout each build their own
        symplectic-integrator closure (the accepted-move proposal itself is
        a gather from the forward rollout's buffer, not a fifth
        integration). Verified empirically: the count stabilizes at 5 after
        the first ``step()`` call and does not grow on further calls with
        the same shapes. A future reverse-rollout caching change (reusing
        the forward-selected h's trajectory builder across the reverse
        selector re-check and the reverse U-turn rollout) could drop this;
        it is a documented follow-up, not attempted here.
        """

        @chex.assert_max_traces(n=5)
        def logdensity_fn(x):
            return jnp.sum(st.norm.logpdf(x))

        chex.clear_trace_counter()

        algo = blackjax.gist_composed(
            logdensity_fn, inverse_mass_matrix=jnp.ones(2), initial_step_size=0.3
        )
        state = algo.init(jnp.zeros(2))
        step = jax.jit(algo.step)

        rng_key = jax.random.key(0)
        for i in range(5):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = step(sample_key, state)


class RegressionCoverageTest(chex.TestCase):
    """Structural coverage matrix: dict-pytree position, non-identity
    (diagonal and dense) metric classes, and fixed-seed bit-stability. Smoke
    level -- correctness evidence is ``OneStepPairedFluxOracleTest`` and
    ``MutationControlTest`` above.
    """

    @parameterized.named_parameters(
        ("identity_metric", jnp.zeros(3), jnp.ones(3), std_normal_logdensity),
        (
            "diagonal_metric",
            jnp.zeros(3),
            jnp.array([0.3, 1.0, 2.5]),
            std_normal_logdensity,
        ),
        (
            "dense_metric",
            jnp.zeros(2),
            jnp.array([[2.0, 1.2], [1.2, 1.0]]),
            lambda x: -0.5
            * x
            @ jnp.linalg.inv(jnp.array([[2.0, 1.2], [1.2, 1.0]]))
            @ x,
        ),
        (
            "dict_pytree_position",
            {"a": jnp.array(0.0), "b": jnp.array([1.0, -0.5])},
            jnp.ones(3),
            _dict_position_logdensity,
        ),
    )
    def test_runs_finite_over_a_short_chain(
        self, init_position, inverse_mass_matrix, logdensity_fn
    ):
        algo = blackjax.gist_composed(
            logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            initial_step_size=0.3,
            max_num_steps=64,
        )
        pos, infos = run_chain(algo, init_position, jax.random.key(0), 100)
        for leaf in jax.tree.leaves(pos):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
        self.assertTrue(bool(jnp.any(infos.is_accepted)))

    def test_bit_identical_reproducibility(self):
        """Fixed-seed reproducibility: the same rng_key must give the same
        trajectory bit-for-bit across two independent constructions -- a
        cheap, deterministic guard against hidden global/mutable state or
        ordering-dependent randomness (e.g. an accidental key reuse between
        (a, b) and L; see the module docstring's RNG discipline)."""
        algo = blackjax.gist_composed(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.4,
        )
        pos_a, info_a = run_chain(algo, jnp.zeros(3), jax.random.key(42), 30)
        pos_b, info_b = run_chain(algo, jnp.zeros(3), jax.random.key(42), 30)
        chex.assert_trees_all_equal(pos_a, pos_b)
        chex.assert_trees_all_equal(info_a, info_b)

    def test_conditional_frequency_of_L_given_selected_h(self):
        """RNG-discipline regression: L's distribution CONDITIONAL on the
        selected step index must be uniform over its interval -- not merely
        have the right MARGINAL distribution. A bug that derived L's key
        from the same subkey as (a, b) (instead of an independent split)
        would correlate the two and bias this conditional law even if L's
        unconditional histogram still looked roughly uniform (this is the
        statistical failure mode the module docstring's RNG discipline
        paragraph names).

        Fixed momentum/state/step size, empirically verified to put ~65% of
        the mass on step_index=1 (h=1.4, forward no-U-turn count U=2,
        interval [1, 2]) -- a single, known constant interval within the
        dominant bucket, so the check compares directly against closed-form
        discrete-uniform moments rather than needing a general goodness-of-
        fit test across several buckets.
        """
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            state.position,
            jnp.array([0.6, -0.3]),
            state.logdensity,
            state.logdensity_grad,
        )
        tuning_parameter_fn = gist_composed._tuning_parameter_fn(
            integrators.velocity_verlet, 0.7, 1, 10, "symmetric", 64, 0.2849
        )
        n = 20_000
        keys = jax.random.split(jax.random.key(3), n)

        def draw(k):
            alpha, _ = tuning_parameter_fn(
                k, integrator_state, std_normal_logdensity, metric
            )
            return alpha.step_index, alpha.num_integration_steps

        step_indices, num_steps = jax.vmap(draw)(keys)
        step_indices = np.asarray(step_indices)
        num_steps = np.asarray(num_steps)

        dominant_j = 1
        mask = step_indices == dominant_j
        # Guard: the bucket must actually dominate, or the check below is
        # underpowered (and vacuously "passes" on too few samples).
        self.assertGreater(int(mask.sum()), int(n * 0.5))

        h = 0.7 * 2.0**dominant_j
        uturn_fn = trajectory_length.num_steps_to_uturn(
            integrators.velocity_verlet, h, metric, 64
        )
        forward_uturn = int(uturn_fn(integrator_state, std_normal_logdensity))
        lo, width = trajectory_length._step_distribution(
            jnp.asarray(forward_uturn), 0.2849
        )
        lo, width = int(lo), int(width)

        L_sub = num_steps[mask]
        np.testing.assert_allclose(L_sub.mean(), (lo + forward_uturn) / 2, atol=0.05)
        np.testing.assert_allclose(L_sub.var(ddof=1), (width**2 - 1) / 12, rtol=0.1)


class ConstructionValidationTest(chex.TestCase):
    """Construction-time validation: the tuning-density ratio is only a
    well-defined, normalized conditional under these hypotheses, so a
    violation must raise loudly rather than silently alias into an
    ill-formed interval deep inside a traced while_loop."""

    def test_psi_above_one_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(path_fraction=1.5)

    def test_psi_below_zero_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(path_fraction=-0.1)

    def test_max_num_steps_below_one_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(max_num_steps=0)

    def test_h_selector_trial_length_below_one_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(h_selector_trial_length=0)

    def test_negative_max_search_steps_raises(self):
        with self.assertRaises(ValueError):
            gist_composed.build_kernel(max_search_steps=-1)


class EdgeCaseTest(BlackJAXTest):
    def test_search_exhausted_forces_rejection(self):
        algo = blackjax.gist_composed(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(2),
            initial_step_size=1e8,  # forces v=-1 (shrink) for virtually any (a, b)
            max_search_steps=0,
        )
        state = algo.init(jnp.zeros(2))
        new_state, info = jax.jit(algo.step)(self.next_key(), state)
        self.assertTrue(bool(info.h_search_exhausted))
        self.assertFalse(bool(info.is_accepted))
        np.testing.assert_allclose(new_state.position, state.position)

    def test_hard_constraint_boundary_no_crash(self):
        logp = lambda x: jnp.where(x[0] > 0, -0.5 * jnp.sum(x**2), -jnp.inf)
        algo = blackjax.gist_composed(
            logp, inverse_mass_matrix=jnp.ones(2), initial_step_size=0.5
        )
        pos, _ = run_chain(algo, jnp.array([0.5, 0.5]), self.next_key(), 300)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        self.assertTrue(np.all(np.asarray(pos[:, 0]) > 0))

    def test_nan_gradient_region_no_crash(self):
        logp = lambda x: -jnp.sum(jnp.sqrt(x))
        algo = blackjax.gist_composed(
            logp,
            inverse_mass_matrix=jnp.ones(2),
            initial_step_size=0.3,
            max_search_steps=5,
            max_num_steps=16,
        )
        pos, _ = run_chain(algo, jnp.array([1.0, 1.0]), self.next_key(), 300)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))


class MomentRecoveryTest(BlackJAXTest):
    """A light end-to-end sampling smoke -- not this module's primary
    correctness evidence (see the file docstring)."""

    def test_isotropic_std_normal(self):
        algo = blackjax.gist_composed(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.5,
        )
        pos, infos = run_chain(algo, jnp.zeros(3), self.next_key(), 4000)
        s = np.asarray(pos[2000:])
        np.testing.assert_allclose(s.mean(), 0.0, atol=0.15)
        np.testing.assert_allclose(s.std(), 1.0, rtol=0.2)
        self.assertGreater(float(jnp.mean(infos.is_accepted)), 0.1)


if __name__ == "__main__":
    absltest.main()
