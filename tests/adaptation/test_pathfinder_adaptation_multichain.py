# Copyright 2024- The Blackjax Authors.
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
"""Tests for pathfinder_adaptation's num_chains, n_paths, and imm_estimator kwargs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from tests.fixtures import std_normal_logdensity

# ---------------------------------------------------------------------------
# Shared position dimension (kept small to limit JIT compilation footprint)
# ---------------------------------------------------------------------------

DIM = 3


# ---------------------------------------------------------------------------
# 1. Backward compatibility: no num_chains / n_paths kwargs
# ---------------------------------------------------------------------------


def test_backward_compat_single_chain():
    """pathfinder_adaptation with no new kwargs is identical to original API.

    The original single-chain single-path API returns a dense (d, d) IMM
    from lbfgs_inverse_hessian_formula_1.  Scalar step_size.
    """
    rng_key = jax.random.key(0)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=50)
    step_size = params["step_size"]
    imm = params["inverse_mass_matrix"]
    assert step_size.shape == ()  # scalar
    # Original returns dense (d, d) IMM from the L-BFGS inverse Hessian
    assert imm.shape == (DIM, DIM)
    assert float(step_size) > 0
    # Pareto-k should NOT be present in single-path single-chain
    assert "_pathfinder_psis_pareto_k" not in params


# ---------------------------------------------------------------------------
# 2. Multi-chain single-path: num_chains > 1, n_paths=1
# ---------------------------------------------------------------------------


def test_multichain_single_path_shapes():
    """num_chains=4, n_paths=1 returns batched step_size (4,) and shared dense IMM (d, d)."""
    rng_key = jax.random.key(1)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=4,
        n_paths=1,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    assert params["step_size"].shape == (4,)
    # Single-path always returns dense (d, d) from L-BFGS inverse Hessian
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    assert jnp.all(params["step_size"] > 0)
    # state.position should have leading dim 4
    assert state.position.shape == (4, DIM)


def test_multichain_single_path_default_n_paths():
    """num_chains=4 with n_paths=None defaults to n_paths=num_chains=4 (multipathfinder)."""
    rng_key = jax.random.key(7)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=4,
        # n_paths defaults to num_chains=4 -> triggers multipathfinder path
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    # n_paths = num_chains = 4 -> multi-path path
    assert params["step_size"].shape == (4,)
    # Multi-path returns dense (d, d) IMM
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    assert "_pathfinder_psis_pareto_k" in params


# ---------------------------------------------------------------------------
# 3. Single-chain multi-path: num_chains=1, n_paths=4
# ---------------------------------------------------------------------------


def test_single_chain_multipathfinder():
    """num_chains=1, n_paths=4 runs multipathfinder, returns scalar step_size and (d,d) IMM."""
    rng_key = jax.random.key(2)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=1,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    assert params["step_size"].shape == ()  # scalar
    # Multi-path returns dense (d, d) IMM
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    assert float(params["step_size"]) > 0
    assert "_pathfinder_psis_pareto_k" in params
    pareto_k = params["_pathfinder_psis_pareto_k"]
    assert pareto_k.shape == ()
    # pareto_k is a scalar diagnostic (can be NaN on degenerate samples, but
    # should always be a finite-or-nan float, not an array)
    assert pareto_k.ndim == 0


# ---------------------------------------------------------------------------
# 4. Paper-canonical: num_chains=4, n_paths=4
# ---------------------------------------------------------------------------


def test_paper_canonical_multichain_multipathfinder():
    """num_chains=4, n_paths=4: multipathfinder -> PSIS init -> vmap DA."""
    rng_key = jax.random.key(3)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=4,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    # Step sizes: one per chain
    assert params["step_size"].shape == (4,)
    assert jnp.all(params["step_size"] > 0)
    # IMM: shared dense (d, d) matrix
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    # PSIS diagnostic included
    assert "_pathfinder_psis_pareto_k" in params
    # state: batched over 4 chains
    assert state.position.shape == (4, DIM)
    # Empirical means should be near zero on this isotropic Gaussian
    mean = jnp.mean(state.position, axis=0)
    np.testing.assert_allclose(mean, jnp.zeros(DIM), atol=2.0)


# ---------------------------------------------------------------------------
# 5. Edge cases: invalid num_chains / n_paths raise ValueError
# ---------------------------------------------------------------------------


def test_num_chains_zero_raises():
    """num_chains=0 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="num_chains"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            num_chains=0,
        )


def test_num_chains_negative_raises():
    """num_chains=-1 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="num_chains"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            num_chains=-1,
        )


def test_n_paths_zero_raises():
    """n_paths=0 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="n_paths"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            n_paths=0,
        )


def test_n_paths_negative_raises():
    """n_paths=-2 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="n_paths"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            n_paths=-2,
        )


# ---------------------------------------------------------------------------
# 6. imm_estimator="psis_empirical" variant
# ---------------------------------------------------------------------------


def test_psis_empirical_single_chain_multipathfinder():
    """imm_estimator='psis_empirical' returns dense (d, d) IMM via jnp.cov."""
    rng_key = jax.random.key(10)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=1,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
        imm_estimator="psis_empirical",
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    assert params["step_size"].shape == ()
    # psis_empirical also returns dense (d, d)
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    assert "_pathfinder_psis_pareto_k" in params


def test_psis_empirical_multichain_multipathfinder():
    """imm_estimator='psis_empirical', num_chains=4, n_paths=4 returns (d, d) IMM."""
    rng_key = jax.random.key(11)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=4,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
        imm_estimator="psis_empirical",
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    assert params["step_size"].shape == (4,)
    assert params["inverse_mass_matrix"].shape == (DIM, DIM)
    assert "_pathfinder_psis_pareto_k" in params
    assert state.position.shape == (4, DIM)


# ---------------------------------------------------------------------------
# 7. Shape contract uniformity: all dispatch paths return (d, d) IMM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_chains,n_paths,imm_estimator",
    [
        (1, None, "lbfgs_psis_mixture"),  # PATH A: original single-chain
        (1, 1, "lbfgs_psis_mixture"),  # PATH A: explicit n_paths=1
        (4, 1, "lbfgs_psis_mixture"),  # PATH B: multi-chain single-path
        (1, 4, "lbfgs_psis_mixture"),  # PATH C: single-chain multi-path
        (4, 4, "lbfgs_psis_mixture"),  # PATH C: paper-canonical
        (1, 4, "psis_empirical"),  # PATH C: psis_empirical single chain
        (4, 4, "psis_empirical"),  # PATH C: psis_empirical multi-chain
    ],
)
def test_imm_shape_uniformly_dense(num_chains, n_paths, imm_estimator):
    """Across all dispatch paths and estimators, inverse_mass_matrix is (d, d)."""
    rng_key = jax.random.key(42)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=num_chains,
        n_paths=n_paths,
        num_samples_per_path=20,
        psis_imm_n_samples=50,
        imm_estimator=imm_estimator,
    )
    init_pos = jnp.zeros(DIM)
    (_, params), _ = warmup.run(rng_key, init_pos, num_steps=20)
    imm = params["inverse_mass_matrix"]
    assert imm.shape == (DIM, DIM), (
        f"Expected (d, d)=({DIM}, {DIM}) but got {imm.shape} "
        f"for num_chains={num_chains}, n_paths={n_paths}, "
        f"imm_estimator={imm_estimator!r}"
    )


# ---------------------------------------------------------------------------
# 8. PSD check: IMM is symmetric and positive-definite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "imm_estimator",
    ["lbfgs_psis_mixture", "psis_empirical"],
)
def test_imm_is_symmetric_and_psd(imm_estimator):
    """Returned IMM is symmetric and passes Cholesky (positive-definite)."""
    rng_key = jax.random.key(20)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        std_normal_logdensity,
        num_chains=1,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
        imm_estimator=imm_estimator,
    )
    init_pos = jnp.zeros(DIM)
    (_, params), _ = warmup.run(rng_key, init_pos, num_steps=30)
    imm = params["inverse_mass_matrix"]
    # Symmetry
    assert jnp.allclose(imm, imm.T, atol=1e-5), (
        f"IMM not symmetric for imm_estimator={imm_estimator!r}: "
        f"max asymmetry = {jnp.max(jnp.abs(imm - imm.T))}"
    )
    # Positive-definite: Cholesky must not raise
    jnp.linalg.cholesky(imm)


# ---------------------------------------------------------------------------
# 9. Degenerate equivalence: n_paths=1 lbfgs_psis_mixture == single-path L-BFGS
# ---------------------------------------------------------------------------


def test_degenerate_n_paths_1_matches_single_path():
    """With n_paths=1, lbfgs_psis_mixture IMM equals the single-path L-BFGS IMM.

    The between-component term is zero when n_paths=1 (only one mu_i = mu_mix),
    so Sigma_mix = Sigma_0 exactly.
    """
    # We use the same key for both runs so that the L-BFGS optimization follows
    # the same trajectory. The single-path run uses a 3-way key split while the
    # n_paths=1 multi-path uses a 4-way split; keys diverge after the first
    # split. We therefore can't check bit-exact equality, but we CAN verify that
    # the n_paths=1 lbfgs_psis_mixture result is a valid (d, d) PSD matrix
    # and that it equals the lbfgs_inverse_hessian at the same ELBO-argmax
    # iterate from that run.
    from blackjax.adaptation.pathfinder_adaptation import (
        _psis_weighted_mixture_covariance,
    )
    from blackjax.vi.multipathfinder import multi_approximate, psis_weights

    rng_key = jax.random.key(0)
    init_pos = jnp.zeros(DIM)

    # Run multi_approximate with n_paths=1 and extract directly.
    pf_key, _ = jax.random.split(rng_key)
    init_positions = init_pos[None]  # (1, DIM)
    mpf_state, _ = multi_approximate(
        pf_key, std_normal_logdensity, init_positions, num_samples=50
    )
    log_weights, _ = psis_weights(mpf_state)

    # lbfgs_psis_mixture IMM from the helper
    mix_imm = _psis_weighted_mixture_covariance(mpf_state, log_weights)

    # Directly compute via lbfgs_inverse_hessian_formula_1 on the single path
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

    single_imm = lbfgs_inverse_hessian_formula_1(
        mpf_state.path_states.alpha[0],
        mpf_state.path_states.beta[0],
        mpf_state.path_states.gamma[0],
    )

    assert jnp.allclose(mix_imm, single_imm, atol=1e-5), (
        f"n_paths=1 mixture IMM != single-path LBFGS IMM: "
        f"max diff = {jnp.max(jnp.abs(mix_imm - single_imm))}"
    )


# ---------------------------------------------------------------------------
# 10. Both estimators converge on the same Gaussian target
# ---------------------------------------------------------------------------


def test_both_estimators_converge_on_gaussian():
    """On an isotropic Gaussian, both estimators produce similar IMMs (rtol=0.5).

    This test uses generous tolerance because empirical covariance with only
    100 samples can have substantial Monte Carlo noise, and lbfgs_psis_mixture
    is the analytic estimator. We just verify they're in the same ballpark.
    """
    n_paths = 4
    num_samples_per_path = 100
    psis_imm_n_samples = 200

    def run_estimator(key, estimator):
        warmup = blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            num_chains=1,
            n_paths=n_paths,
            num_samples_per_path=num_samples_per_path,
            psis_imm_n_samples=psis_imm_n_samples,
            imm_estimator=estimator,
        )
        (_, params), _ = warmup.run(key, jnp.zeros(DIM), num_steps=30)
        return params["inverse_mass_matrix"]

    key_a, key_b = jax.random.split(jax.random.key(99))
    imm_a = run_estimator(key_a, "lbfgs_psis_mixture")
    imm_b = run_estimator(key_b, "psis_empirical")

    # Both should have the same shape
    assert imm_a.shape == (DIM, DIM)
    assert imm_b.shape == (DIM, DIM)

    # Diagonal elements (variances) should be positive and in the same ballpark.
    diag_a = jnp.diag(imm_a)
    diag_b = jnp.diag(imm_b)
    assert jnp.all(
        diag_a > 0
    ), f"lbfgs_psis_mixture diagonal not all positive: {diag_a}"
    assert jnp.all(diag_b > 0), f"psis_empirical diagonal not all positive: {diag_b}"

    # Trace should be similar within a generous factor (x5)
    trace_a = jnp.trace(imm_a)
    trace_b = jnp.trace(imm_b)
    ratio = jnp.maximum(trace_a, trace_b) / jnp.minimum(trace_a, trace_b)
    assert float(ratio) < 5.0, (
        "Traces differ by more than 5x: "
        f"trace_a={float(trace_a)}, trace_b={float(trace_b)}"
    )


# ---------------------------------------------------------------------------
# 11. Between-component term is non-zero for a bimodal target
# ---------------------------------------------------------------------------


def test_between_component_contributes_for_bimodal_target():
    """lbfgs_psis_mixture IMM has larger trace than within-component alone on bimodal.

    We construct a 1-D bimodal target (mixture of two separated Gaussians) and
    verify that the between-component correction increases the trace relative to
    a within-component-only estimate.
    """
    from blackjax.adaptation.pathfinder_adaptation import (
        _psis_weighted_mixture_covariance,
    )
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
    from blackjax.vi.multipathfinder import multi_approximate, psis_weights

    # 1-D bimodal: two Gaussians at -3 and +3
    def bimodal_logdensity(x):
        log_p1 = jax.scipy.stats.norm.logpdf(x, loc=-3.0, scale=1.0)
        log_p2 = jax.scipy.stats.norm.logpdf(x, loc=3.0, scale=1.0)
        return jax.scipy.special.logsumexp(jnp.array([log_p1, log_p2]))

    # Start one path near each mode
    init_positions = jnp.array([[-3.0], [3.0]])

    rng_key = jax.random.key(5)
    mpf_state, _ = multi_approximate(
        rng_key, bimodal_logdensity, init_positions, num_samples=100
    )
    log_weights, _ = psis_weights(mpf_state)

    # Full mixture covariance
    mix_imm = _psis_weighted_mixture_covariance(mpf_state, log_weights)

    # Within-component only (no between term)
    path_states = mpf_state.path_states
    num_samples_per_path = mpf_state.logp.shape[1]
    n_paths_local = log_weights.shape[0] // num_samples_per_path
    log_weights_per_path = log_weights.reshape(n_paths_local, num_samples_per_path)
    log_path_weights = jax.scipy.special.logsumexp(log_weights_per_path, axis=1)
    log_path_weights_norm = log_path_weights - jax.scipy.special.logsumexp(
        log_path_weights
    )
    w = jnp.exp(log_path_weights_norm)
    sigmas = jax.vmap(lbfgs_inverse_hessian_formula_1)(
        path_states.alpha, path_states.beta, path_states.gamma
    )
    sigma_within = jnp.einsum("i,ijk->jk", w, sigmas)

    # The full mixture covariance should have larger trace
    mix_trace = float(jnp.trace(mix_imm))
    within_trace = float(jnp.trace(sigma_within))
    assert mix_trace > within_trace, (
        f"Expected mixture trace ({mix_trace}) > within-only trace ({within_trace}). "
        "between-component term should add variance for bimodal target"
    )


# ---------------------------------------------------------------------------
# 12. imm_estimator warning on single-chain single-path
# ---------------------------------------------------------------------------


def test_imm_estimator_warns_on_single_path():
    """Passing imm_estimator='psis_empirical' on single-path dispatch raises UserWarning."""
    with pytest.warns(UserWarning, match="imm_estimator"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            num_chains=1,
            n_paths=1,
            imm_estimator="psis_empirical",
        )


# ---------------------------------------------------------------------------
# 13. Pytree position regression: dict-shaped position survives PATH C
# ---------------------------------------------------------------------------


def test_pytree_position_multipathfinder_dispatch():
    """Multi-path dispatch handles dict-shaped pytree positions without shape leak.

    Regression for the bug where ``_psis_weighted_mixture_covariance`` assumed
    ``path_states.position`` was a flat ``(n_paths, d)`` array, but
    ``PathfinderState.position`` actually stores the user's pytree shape.
    For a dict-position model like ``{'x': array}`` the einsum
    ``"i,id->d"`` chokes with a ``ValueError`` from ``opt_einsum`` since
    a dict isn't array-like.

    Surfaced 2026-05-19 during tuningfork's pathfinder shim consolidation
    (PR #33).  The fix ravels each path's pytree position before the einsum.
    """

    def dict_logdensity(pos):
        # 10-D isotropic Gaussian over a single named site.
        return std_normal_logdensity(pos["x"])

    rng_key = jax.random.key(20260519)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        dict_logdensity,
        num_chains=4,
        n_paths=4,
        num_samples_per_path=30,
        psis_imm_n_samples=100,
    )
    init_pos = {"x": jnp.zeros(10)}
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=30)

    # Successful run is the regression check; just verify shape contract.
    assert params["step_size"].shape == (4,)
    assert params["inverse_mass_matrix"].shape == (10, 10)
    assert "_pathfinder_psis_pareto_k" in params
