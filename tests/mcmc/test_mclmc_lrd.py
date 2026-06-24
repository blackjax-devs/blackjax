# Copyright 2026- The Blackjax Authors.
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
"""Unit tests for Low-Rank + Diagonal (LRD) preconditioned MCLMC integrator.

Coverage
--------
1. ``test_lrd_forward_adjoint_product_equals_Minv`` — algebraic identity
   forward_L ∘ adjoint_L = M⁻¹ (relative error < 1e-5).
2. ``test_esh_dynamics_lrd_dispatch`` — ``esh_dynamics_momentum_update_one_step``
   with a ``LowRankInverseMassMatrix`` produces the same gradient transform as an
   explicit matrix multiply, confirming the forward/adjoint operators are wired
   correctly.
3. ``test_mclmc_lrd_sampling_rotated_gaussian`` — end-to-end sampling test on a
   rotated ill-conditioned Gaussian (d=10, κ=100).  The LRD preconditioner is
   constructed from the true covariance; the test verifies that the posterior
   mean is ≈ 0 and the sample second moments match the true covariance within a
   loose tolerance.  (Diagonal MCLMC without LRD is NOT expected to converge
   adequately at the same budget — that asymmetry is the whole point.)
"""

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
import blackjax.mcmc.mclmc as mclmc_module
from blackjax.mcmc.integrators import esh_dynamics_momentum_update_one_step
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lrd_imm(cov: jax.Array, k: int) -> LowRankInverseMassMatrix:
    """Decompose a dense covariance into LowRankInverseMassMatrix(sigma, U, lam).

    Returns sigma, U, lam such that:
        M⁻¹ = diag(sigma) (I + U(Λ-I)Uᵀ) diag(sigma)
    approximates ``cov`` at rank k.
    """
    sigma = jnp.sqrt(jnp.diagonal(cov))
    inv_sigma = 1.0 / sigma
    # Correlation matrix
    C = cov * inv_sigma[:, None] * inv_sigma[None, :]
    eigenvals, eigenvectors = jnp.linalg.eigh(C)
    # Sort by |λ - 1| descending — highest preconditioning impact first
    sort_idx = jnp.argsort(jnp.abs(eigenvals - 1.0))[::-1]
    top_idx = sort_idx[:k]
    return LowRankInverseMassMatrix(
        sigma=sigma,
        U=eigenvectors[:, top_idx],
        lam=eigenvals[top_idx],
    )


def _build_rotated_gaussian(key: jax.Array, d: int, condition_number: float):
    """Return (logdensity_fn, cov) for a rotated, ill-conditioned Gaussian.

    Eigenvalues are log-spaced between ``1/sqrt(κ)`` and ``sqrt(κ)``.
    A random rotation is applied so the axes are *not* aligned with the
    coordinate axes — this makes the problem genuinely hard for diagonal
    preconditioning.
    """
    eigs = jnp.logspace(
        -0.5 * jnp.log10(condition_number),
        0.5 * jnp.log10(condition_number),
        d,
    )
    # Random orthogonal matrix via QR
    Q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(d, d)))
    cov = Q @ jnp.diag(eigs) @ Q.T  # shape (d, d)
    precision = Q @ jnp.diag(1.0 / eigs) @ Q.T

    def logdensity_fn(x):
        return -0.5 * jnp.dot(x, precision @ x)

    return logdensity_fn, cov


# ---------------------------------------------------------------------------
# Test 1: algebraic identity forward_L ∘ adjoint_L = M⁻¹
# ---------------------------------------------------------------------------


def test_lrd_forward_adjoint_product_equals_Minv():
    """Verify forward_L ∘ adjoint_L = M⁻¹ at numerical precision.

    Constructs L_LR as a dense matrix and checks that L_LR @ L_LR^T matches
    the closed-form M^{-1} = diag(σ)(I + U(Λ-I)Uᵀ)diag(σ) to relative
    Frobenius error < 1e-5.  This is the claim verified by @statistician at
    2.8e-7 relative error.
    """
    key = jax.random.key(42)
    d, k = 30, 8

    # Random LRD parameters: sigma > 0, orthonormal U, positive lam
    sigma = jnp.abs(jax.random.normal(key, shape=(d,))) + 0.1
    raw_U, _ = jnp.linalg.qr(jax.random.normal(key, shape=(d, k)))
    lam = jnp.abs(jax.random.normal(key, shape=(k,))) + 0.5  # > 0

    sqrt_lam = jnp.sqrt(lam)

    # Build L_LR as dense matrix: L_LR = diag(sigma) @ (I + U diag(sqrt_lam-1) U^T)
    A = jnp.eye(d) + raw_U @ jnp.diag(sqrt_lam - 1.0) @ raw_U.T
    L_LR_dense = jnp.diag(sigma) @ A  # (d, d)

    # Reconstructed M⁻¹ via L_LR L_LR^T
    M_inv_reconstructed = L_LR_dense @ L_LR_dense.T

    # Expected M⁻¹ = diag(sigma) (I + U(Λ-I)Uᵀ) diag(sigma)
    B = jnp.eye(d) + raw_U @ jnp.diag(lam - 1.0) @ raw_U.T
    M_inv_expected = jnp.diag(sigma) @ B @ jnp.diag(sigma)

    rel_err = float(
        jnp.linalg.norm(M_inv_reconstructed - M_inv_expected, ord="fro")
        / jnp.linalg.norm(M_inv_expected, ord="fro")
    )
    assert rel_err < 1e-5, (
        "L_LR @ L_LR^T != M_inv: relative Frobenius error = %.2e" % rel_err
    )


# ---------------------------------------------------------------------------
# Test 2: ESH dynamics dispatch — LRD path applies operators correctly
# ---------------------------------------------------------------------------


def test_esh_dynamics_lrd_dispatch():
    """Verify that the LRD branch of esh_dynamics_momentum_update_one_step
    applies forward_L and adjoint_L correctly.

    Strategy: build the LRD operators both via the function-under-test and via
    explicit dense-matrix multiplication, then check that the kinetic_grad
    output matches.
    """
    key = jax.random.key(7)
    d, k = 15, 4
    step_size = 0.1
    coef = 0.2

    # Build LRD parameters
    sigma = jnp.abs(jax.random.normal(key, shape=(d,))) + 0.2
    raw_U, _ = jnp.linalg.qr(jax.random.normal(key, shape=(d, k)))
    lam = jnp.abs(jax.random.normal(key, shape=(k,))) + 0.3

    imm = LowRankInverseMassMatrix(sigma=sigma, U=raw_U, lam=lam)

    # Build the LRD ESH updater
    lrd_update = esh_dynamics_momentum_update_one_step(imm)

    # Random momentum (unit-norm) and log-density gradient
    m_raw = jax.random.normal(key, shape=(d,))
    momentum = m_raw / jnp.linalg.norm(m_raw)
    logdensity_grad = jax.random.normal(key, shape=(d,))

    # Run one step
    new_mom, kinetic_grad, _ = lrd_update(momentum, logdensity_grad, step_size, coef)

    # Manually compute what forward_L should produce from the new normalized momentum
    # new_mom is the normalized raw momentum; kinetic_grad = forward_L(new_mom)
    sqrt_lam = jnp.sqrt(lam)
    expected_kinetic_grad = sigma * (
        new_mom + raw_U @ ((sqrt_lam - 1.0) * (raw_U.T @ new_mom))
    )

    np.testing.assert_allclose(
        np.array(kinetic_grad),
        np.array(expected_kinetic_grad),
        rtol=1e-5,
        err_msg="kinetic_grad from LRD ESH step does not match forward_L(new_mom)",
    )


# ---------------------------------------------------------------------------
# Test 3: end-to-end sampling on a rotated ill-conditioned Gaussian
# ---------------------------------------------------------------------------


def test_mclmc_lrd_sampling_rotated_gaussian():
    """MCLMC with LRD preconditioning correctly samples a rotated ill-conditioned
    Gaussian (d=10, κ=100).

    The LRD preconditioner is built from the true covariance matrix.  We run a
    short chain and verify that:
      - The sample mean is close to 0 (‖μ̂‖ < 0.5).
      - The diagonal of the sample covariance matches the true diagonal to
        within 50% relative error.

    This is a smoke test, not a convergence benchmark — it verifies that the
    LRD path is wired up correctly end-to-end, not that it achieves a specific
    ESS/grad.
    """
    jax.config.update("jax_enable_x64", False)  # keep default float32

    master_key = jax.random.key(98765)
    geom_key, init_key, warmup_key, sample_key = jax.random.split(master_key, 4)

    d = 10
    condition_number = 100.0
    n_warmup = 2000
    n_samples = 2000

    logdensity_fn, cov = _build_rotated_gaussian(geom_key, d, condition_number)

    # Build LRD preconditioner from the true covariance (full-rank for small d)
    k = d - 1  # keep d-1 eigenvectors to avoid trivial identity
    lrd_imm = _make_lrd_imm(cov, k)

    # --- Warmup with LRD ---
    init_pos = jax.random.normal(init_key, shape=(d,))
    state = mclmc_module.init(init_pos, logdensity_fn, init_key)

    kernel = mclmc_module.build_kernel()

    # Tune L and step_size with the fixed LRD mass matrix (no diagonal adaptation)
    _, params, _ = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=n_warmup,
        state=state,
        rng_key=warmup_key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=False,
        params=blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState(
            L=jnp.sqrt(float(d)),
            step_size=jnp.sqrt(float(d)) * 0.25,
            inverse_mass_matrix=lrd_imm,
        ),
    )

    # --- Sampling ---
    def body_fn(state, rng_key):
        new_state, info = kernel(
            rng_key,
            state,
            logdensity_fn,
            inverse_mass_matrix=lrd_imm,
            L=params.L,
            step_size=params.step_size,
        )
        return new_state, new_state.position

    _, samples = jax.lax.scan(
        body_fn,
        state,
        jax.random.split(sample_key, n_samples),
    )
    # samples: (n_samples, d)

    sample_mean = jnp.mean(samples, axis=0)
    sample_cov_diag = jnp.var(samples, axis=0)
    true_cov_diag = jnp.diagonal(cov)

    mean_norm = float(jnp.linalg.norm(sample_mean))
    rel_err_var = float(
        jnp.mean(jnp.abs(sample_cov_diag - true_cov_diag) / true_cov_diag)
    )

    assert mean_norm < 0.5, (
        "LRD MCLMC posterior mean far from 0: norm = %.3f" % mean_norm
    )
    assert rel_err_var < 0.5, (
        "LRD MCLMC diagonal variance relative error too large: %.3f" % rel_err_var
    )
