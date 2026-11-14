# Copyright 2020- The Blackjax Authors.
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
"""Shared test utilities for BlackJAX tests."""

import datetime

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.stats import norm


class BlackJAXTest(chex.TestCase):
    """Base test case with date-based PRNG key management.

    Provides a fresh ``self.key`` for each test method, seeded from today's
    date so results are deterministic within a day and rotate automatically.

    Use ``self.next_key()`` to obtain a fresh subkey for each independent
    random operation — this advances ``self.key`` so no two calls share the
    same underlying key.

    Example
    -------

    .. code::

        class MyTest(BlackJAXTest):
            def test_something(self):
                state = init(self.next_key())
                new_state, info = kernel(self.next_key(), state, ...)
    """

    def setUp(self):
        super().setUp()
        seed = int(datetime.date.today().strftime("%Y%m%d"))
        self.key = jax.random.key(seed)

    def next_key(self):
        """Advance the key state and return a fresh subkey.

        Each call produces an independent subkey and mutates ``self.key`` so
        that successive calls yield different keys.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey


def std_normal_logdensity(x, scale=1.0):
    """Log density (unnormalised) of a zero-mean Gaussian.

    Default ``scale=1.0`` gives the standard isotropic normal — preserves the
    original signature for callers passing no ``scale``.  Supply a scalar to
    scale uniformly across all dimensions, or an array of shape matching
    ``ravel_pytree(x)[0].shape`` for an anisotropic per-element scale (i.e.
    the per-dim standard deviation of the target).

    The normalisation constants (``-d/2 * log(2*pi)`` and ``-sum log scale``)
    are dropped since they don't affect HMC sampling.

    Works with both array and PyTree positions.
    """
    if jnp.ndim(scale) == 0:
        leaves = jax.tree.leaves(x)
        return -0.5 / (scale**2) * sum(jnp.sum(leaf**2) for leaf in leaves)
    flat_x, _ = ravel_pytree(x)
    return -0.5 * jnp.sum((flat_x / scale) ** 2)


def neal_funnel_logdensity(x):
    """Neal's funnel (unnormalised): the canonical stress test for
    per-location step-size adaptation.

    ``x[0]`` is the "neck" variable ``y ~ N(0, 3**2)``; ``x[1:]`` are
    ``N(0, exp(y))`` i.i.d. given ``y``. A single global step size cannot
    work well here: the conditional scale of ``x[1:]`` ranges over
    ``exp(y / 2)`` as ``y`` sweeps its prior, from tiny near the neck
    (``y`` very negative) to huge in the mouth (``y`` very positive).

    Works with any array position of length >= 2 (one neck coordinate plus
    at least one funnel coordinate).
    """
    y = x[0]
    v = x[1:]
    logdensity_y = -0.5 * (y / 3.0) ** 2
    logdensity_v = -0.5 * jnp.exp(-y) * jnp.sum(v**2) - 0.5 * v.shape[0] * y
    return logdensity_y + logdensity_v


def smooth_skewed_logdensity(x):
    """A smooth, gradient-friendly asymmetric target: ``y = log(u)`` for
    ``u ~ Exponential(1)``, i.e. ``-y ~ Gumbel(0, 1)``.

    Unnormalised log density ``y - exp(y)``, defined and differentiable on
    all of ``R`` (gradient ``1 - exp(y)``, never zero/degenerate). A
    symmetric Gaussian target cannot detect a reversed-direction/sign bug in
    a leapfrog+flip or a forward-vs-reverse rollout, but an asymmetric one
    can (mirrors ``test_slice.py::test_multivariate_skewed_exponential``'s
    rationale) -- **without** the raw ``jnp.where(x > 0, ..., -inf)``
    representation of ``Exponential`` directly, whose *zero* gradient
    outside the support lets a leapfrog trajectory that crosses the
    boundary drift forever under a purely linear potential (no restoring
    force once ``logdensity_grad`` clamps to 0) -- pathologically
    degenerate for a gradient-based (HMC-family) sampler, even though it is
    perfectly fine for a density-evaluation-only sampler like slice
    sampling.

    Closed-form moments (per-coordinate, i.i.d.): mean ``-gamma`` (negative
    Euler-Mascheroni, approx -0.5772), variance ``pi**2 / 6`` (approx
    1.6449), skewness approx -1.1395 (left-skewed).
    """
    return jnp.sum(x - jnp.exp(x))


def assert_grand_mean_within_robust_tolerance(
    samples: np.ndarray,
    expected_mean: float = 0.0,
    atol_floor: float = 1.0,
    k_sigma: float = 5.0,
):
    """Multi-chain robust grand-mean assertion for difficult hierarchical targets.

    Replaces the single-chain ESS-gated assertion (3rd MC-tolerance escape, issue #970;
    lineage #957 → #959 → #971 skip): single-chain AR-ESS both over-estimates ESS on
    poorly-mixing targets (~7× on the funnel) and the ``ess_min`` floor was a
    per-realization lottery — 6–17% of legitimate chains failed the gate not because the
    sampler was biased but because their individual ESS draw fell below the floor.

    Design rationale
    ----------------
    - **LOCATION = plain grand mean** over all K chains: trapped chains' rare deep
      excursions are physically real target visits that balance the finite-N bias, so
      median/robust location is deliberately NOT used — it centers on the biased bulk
      (+0.2 measured skew-bias on the funnel).

    - **SCALE = MAD of K chain-means** (breakdown point ~50%): a single stuck chain
      cannot inflate the threshold and hide a real bias.  A std/MCSE-based scale loses
      up to half its detection power to exactly that trap-inflation mechanism.
      The Gaussian consistency constant 1.4826 converts MAD to an asymptotically
      unbiased standard-deviation estimate.

    - **FLOOR = ``atol_floor``**: same-start chains share a common finite-N bias that
      any cross-chain dispersion is structurally blind to, and a collapsible robust scale
      can fall below it (measured false-fail without the floor). The asserted claim is
      therefore "no GROSS systematic bias" (≥ ``atol_floor``), not exact unbiased
      recovery.

    - **K ≈ 24 recommended**: large K shrinks one trapped chain's leverage on the grand
      mean below the floor (power) and concentrates the null grand mean well inside it
      (null-escape rate ≲ 1e-3 by bootstrap).

    Parameters
    ----------
    samples : np.ndarray
        Array of shape ``(K, T)`` — K independent post-burn chains' samples.
    expected_mean : float
        Target mean. Default 0.0.
    atol_floor : float
        Minimum threshold (absolute tolerance floor). Guards against a collapsing robust
        scale when all K chains happen to agree closely. Default 1.0.
    k_sigma : float
        Multiplier on the robust SE for the upper threshold. Default 5.0.

    Raises
    ------
    AssertionError
        If ``|grand_mean - expected_mean| >= max(atol_floor, k_sigma * robust_se)``.
    """
    chain_means = np.asarray(samples).mean(axis=1)
    n_chains = chain_means.shape[0]
    grand_mean = float(chain_means.mean())
    # 1.4826 = Gaussian consistency constant: 1.4826 * MAD estimates a std deviation.
    mad = float(np.median(np.abs(chain_means - np.median(chain_means))))
    robust_se = 1.4826 * mad / np.sqrt(n_chains)
    threshold = max(atol_floor, k_sigma * robust_se)
    assert abs(grand_mean - expected_mean) < threshold, (
        f"Grand mean {grand_mean:.6f} is inconsistent with expected {expected_mean:.6f} "
        f"(difference {abs(grand_mean - expected_mean):.6f} >= threshold {threshold:.6f}). "
        f"atol_floor={atol_floor:.3f}, robust_se={robust_se:.6f}, n_chains={n_chains:d}."
    )


def correlated_gaussian(dim, seed=0):
    """A rotated Gaussian with scales spread over four decades.

    Returns the mean, the Cholesky factor of the covariance, the log-density
    and the whitening map ``x -> L^{-1}(x - mean)``.
    """
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    cholesky = rotation * np.logspace(-2, 2, dim)
    mean = np.arange(dim, dtype=float)
    inverse_cholesky = np.linalg.inv(cholesky)

    def logdensity_fn(x):
        whitened = inverse_cholesky.astype(x.dtype) @ (x - mean.astype(x.dtype))
        return -0.5 * jnp.sum(whitened**2)

    def whiten(x):
        return (np.asarray(x, dtype=float) - mean) @ inverse_cholesky.T

    return mean, cholesky, logdensity_fn, whiten


def whitened_moments(whitened):
    """First moments and the upper triangle of the second-moment matrix, with
    their values under a standard normal."""
    dim = whitened.shape[-1]
    rows, columns = np.triu_indices(dim)
    second = whitened[..., rows] * whitened[..., columns]
    return whitened, np.zeros(dim), second, np.eye(dim)[rows, columns]


def rosenbrock_logdensity(x):
    """``x[0] ~ N(1, 10)`` and ``x[1] | x[0] ~ N(x[0]**2, 0.1)``."""
    return -(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2) / 20.0


def sample_rosenbrock(rng_key, shape):
    key_first, key_second = jax.random.split(rng_key)
    first = 1.0 + jnp.sqrt(10.0) * jax.random.normal(key_first, shape)
    second = first**2 + jnp.sqrt(0.1) * jax.random.normal(key_second, shape)
    return jnp.stack([first, second], axis=-1)


MIXTURE_WEIGHT = 0.3
MIXTURE_MEANS = np.array([[-1.5, 0.0], [1.5, 0.0]])


def mixture_logdensity(x):
    means = MIXTURE_MEANS.astype(x.dtype)
    return jnp.logaddexp(
        jnp.log(MIXTURE_WEIGHT) - 0.5 * jnp.sum((x - means[0]) ** 2),
        jnp.log(1.0 - MIXTURE_WEIGHT) - 0.5 * jnp.sum((x - means[1]) ** 2),
    )


def sample_mixture(rng_key, shape):
    key_component, key_noise = jax.random.split(rng_key)
    component = jax.random.bernoulli(key_component, 1.0 - MIXTURE_WEIGHT, shape)
    noise = jax.random.normal(key_noise, shape + (2,))
    return jnp.asarray(MIXTURE_MEANS, noise.dtype)[component.astype(int)] + noise


def mixture_moments():
    """Mean, per-coordinate variance and ``P(x[0] < 0)`` of the mixture."""
    weights = np.array([MIXTURE_WEIGHT, 1.0 - MIXTURE_WEIGHT])
    mean = weights @ MIXTURE_MEANS
    variance = 1.0 + weights @ (MIXTURE_MEANS - mean) ** 2
    probability_negative = weights @ norm.cdf(-MIXTURE_MEANS[:, 0])
    return mean, variance, probability_negative


def assert_iid_mean(values, expected, num_sigma=5.0):
    """Check the mean of independent replicates, along the leading axis."""
    values = np.asarray(values, dtype=float)
    estimate = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    np.testing.assert_array_less(
        np.abs(estimate - expected), num_sigma * standard_error
    )


def assert_chain_mean(
    values, expected, max_standard_error, num_batches=25, num_sigma=5.0
):
    """Batch means of the walker-averaged series of ``values``, which has shape
    ``(num_steps, num_walkers, ...)``."""
    series = np.asarray(values, dtype=float).mean(axis=1)
    batch_length = series.shape[0] // num_batches
    batch_means = (
        series[: num_batches * batch_length]
        .reshape((num_batches, batch_length) + series.shape[1:])
        .mean(axis=1)
    )
    estimate = batch_means.mean(axis=0)
    standard_error = batch_means.std(axis=0, ddof=1) / np.sqrt(num_batches)
    np.testing.assert_array_less(standard_error, max_standard_error)
    np.testing.assert_array_less(
        np.abs(estimate - expected), num_sigma * standard_error
    )
