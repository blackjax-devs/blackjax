import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.mcmc.barker import _barker_pdf, _barker_sample_nd


class BarkerSamplingTest(chex.TestCase):
    @parameterized.parameters([1234, 5678])
    def test_nd(self, seed):
        n_samples = 100_000

        key = jax.random.key(seed)
        m, a, scale = (
            jnp.array([1.0, 0.5, 0.0, 0.0]),
            jnp.array([1.0, -2.0, 10.0, 0.0]),
            0.5,
        )

        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: _barker_sample_nd(k, m, a, scale))(keys)
        # Check that the emprical mean and the mean computed as sum(x * p(x) dx) are close
        _test_samples_vs_pdf(samples, lambda x: _barker_pdf(x, m, a, scale))


def _test_samples_vs_pdf(samples, pdf):
    samples_mean = jnp.mean(samples, 0)
    samples_squrared_mean = jnp.mean(samples**2, 0)
    linspace = jnp.linspace(-10, 10, 50_000)

    diff = jnp.diff(linspace, axis=0)

    # trapezoidal rule
    pdf_mean = 0.5 * jnp.sum(
        linspace[1:, None] * pdf(linspace[1:, None]) * diff[:, None], 0
    )
    pdf_mean += 0.5 * jnp.sum(
        linspace[:-1, None] * pdf(linspace[:-1, None]) * diff[:, None], 0
    )
    pdf_squared_mean = 0.5 * jnp.sum(
        linspace[1:, None] ** 2 * pdf(linspace[1:, None]) * diff[:, None], 0
    )
    pdf_squared_mean += 0.5 * jnp.sum(
        linspace[:-1, None] ** 2 * pdf(linspace[:-1, None]) * diff[:, None], 0
    )

    chex.assert_trees_all_close(samples_mean, pdf_mean, atol=1e-2, rtol=1e-2)
    chex.assert_trees_all_close(
        samples_squrared_mean, pdf_squared_mean, atol=1e-2, rtol=1e-2
    )


if __name__ == "__main__":
    absltest.main()
