import jax
import numpy as np
import pytest

from blackjax.adaptation.mass_matrix import mass_matrix_adaptation


@pytest.mark.parametrize("n_dim", [1, 3])
@pytest.mark.parametrize("is_mass_matrix_diagonal", [True, False])
def test_welford_adaptation(n_dim, is_mass_matrix_diagonal):
    num_samples = 3_000

    # Generate samples from a multivariate normal distribution whose
    # covariance matrix we are going to try to recover.
    np.random.seed(0)
    mu = np.random.randn(n_dim)
    a = np.random.randn(n_dim, n_dim)
    cov = np.matmul(a.T, a)
    samples = np.random.multivariate_normal(mu, cov, num_samples)

    init, update, final = mass_matrix_adaptation(is_mass_matrix_diagonal)
    update_step = lambda state, sample: (update(state, sample), None)

    mm_state = init(n_dim)
    mm_state, _ = jax.lax.scan(update_step, mm_state, samples)
    mm_state = final(mm_state)

    estimated_cov = mm_state.inverse_mass_matrix

    if is_mass_matrix_diagonal:
        if n_dim == 1:
            np.testing.assert_allclose(estimated_cov, cov.squeeze(), rtol=1e-1)
        else:
            np.testing.assert_allclose(estimated_cov, np.diagonal(cov), rtol=1e-1)
    else:
        np.testing.assert_allclose(estimated_cov, cov, rtol=1e-1)
