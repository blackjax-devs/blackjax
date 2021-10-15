"""Test the welford adaptation algorithm."""
import itertools

import chex
import jax
import numpy as np
from absl.testing import absltest, parameterized

from blackjax.adaptation.mass_matrix import mass_matrix_adaptation


class MassMatrixAdaptationTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product([1, 3], [True, False]))
    def test_welford_adaptation(self, n_dim, is_mass_matrix_diagonal):
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

        @self.variant
        def run_adaptation(init_state, samples):
            mm_state, _ = jax.lax.scan(update_step, init_state, samples)
            mm_state = final(mm_state)
            return mm_state.inverse_mass_matrix

        mm_state = init(n_dim)
        estimated_cov = run_adaptation(mm_state, samples)

        if is_mass_matrix_diagonal:
            if n_dim == 1:
                np.testing.assert_allclose(estimated_cov, cov.squeeze(), rtol=1e-1)
            else:
                np.testing.assert_allclose(estimated_cov, np.diagonal(cov), rtol=1e-1)
        else:
            np.testing.assert_allclose(estimated_cov, cov, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
