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
"""MCMC diagnostics."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.fftpack import next_fast_len  # type: ignore

from blackjax.types import Array, ArrayLike

__all__ = [
    "potential_scale_reduction",
    "effective_sample_size",
    "psis_weights",
]


def potential_scale_reduction(
    input_array: ArrayLike, chain_axis: int = 0, sample_axis: int = 1
) -> Array:
    """Gelman and Rubin (1992)'s potential scale reduction for computing multiple MCMC chain convergence.

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contains a chain dimension and a sample dimension.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    NDArray of the resulting statistics (r-hat), with the chain and sample dimensions squeezed.

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \\hat{R} = \\frac{\\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\\hat{V}` is the posterior variance
    estimate for the pooled traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged :cite:p:`stan_rhat,gelman1992inference`.

    """
    assert (
        input_array.shape[chain_axis] > 1
    ), "potential_scale_reduction as implemented only works for two or more chains."

    num_samples = input_array.shape[sample_axis]
    # Compute stats for each chain
    per_chain_mean = input_array.mean(axis=sample_axis, keepdims=True)
    per_chain_var = input_array.var(axis=sample_axis, ddof=1, keepdims=True)
    # Compute between-chain stats
    between_chain_variance = num_samples * per_chain_mean.var(
        axis=chain_axis, ddof=1, keepdims=True
    )
    # Compute within-chain stats
    within_chain_variance = per_chain_var.mean(axis=chain_axis, keepdims=True)
    # Estimate of marginal posterior variance
    rhat_value = jnp.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1)
        / (num_samples)
    )
    return rhat_value.squeeze()


def effective_sample_size(
    input_array: ArrayLike, chain_axis: int = 0, sample_axis: int = 1
) -> Array:
    """Compute estimate of the effective sample size (ess).

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contains a chain dimension and a sample dimension.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    NDArray of the resulting statistics (ess), with the chain and sample dimensions squeezed.

    Notes
    -----
    The basic ess (:math:`N_{\\mathit{eff}}`) diagnostic is computed by:

    .. math:: \\hat{N}_{\\mathit{eff}} = \\frac{MN}{\\hat{\\tau}}

    .. math:: \\hat{\\tau} = -1 + 2 \\sum_{t'=0}^K \\hat{P}_{t'}

    where :math:`M` is the number of chains, :math:`N` the number of draws,
    :math:`\\hat{\\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
    :math:`K` is the last integer for which :math:`\\hat{P}_{K} = \\hat{\\rho}_{2K} +
    \\hat{\\rho}_{2K+1}` is still positive :cite:p:`stan_ess,gelman1995bayesian`.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion :cite:p:`geyer1992practical,geyer2011introduction`.

    """
    input_shape = input_array.shape
    sample_axis = sample_axis if sample_axis >= 0 else len(input_shape) + sample_axis
    num_chains = input_shape[chain_axis]
    num_samples = input_shape[sample_axis]
    assert (
        num_samples > 1
    ), f"The input array must have at least 2 samples, got only {num_samples}."

    mean_across_chain = input_array.mean(axis=sample_axis, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=sample_axis)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=sample_axis)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=sample_axis) / num_samples
    )
    mean_autocov_var = autocov_value.mean(chain_axis, keepdims=True)
    mean_var0 = (
        jnp.take(mean_autocov_var, jnp.array([0]), axis=sample_axis)
        * num_samples
        / (num_samples - 1.0)
    )
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda mean_across_chain: weighted_var
        + mean_across_chain.var(axis=chain_axis, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=mean_across_chain,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(
        mean_autocov_var, jnp.arange(1, num_samples_even), axis=sample_axis
    )
    rho_hat = jnp.concatenate(
        [
            jnp.ones_like(mean_var0),
            1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,
        ],
        axis=sample_axis,
    )

    rho_hat = jnp.moveaxis(rho_hat, sample_axis, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat

    return ess.squeeze()


def splitR(position, num_chains, superchain_size, func_for_splitR=jnp.square):
    # combine the chains in super-chains to compute expectation values
    func_mk = jax.vmap(func_for_splitR)(position)  # shape = (# chains, # func)
    func_mk = func_mk.reshape(
        num_chains // superchain_size, superchain_size, func_mk.shape[-1]
    )  # shape = (# superchains, # chains in superchain, # func)
    func_k = jnp.average(func_mk, axis=1)  # shape = (# superchains, # func)
    func_sq_k = jnp.average(
        jnp.square(func_mk), axis=1
    )  # shape = (# superchains, # func)
    W_k = (
        (func_sq_k - jnp.square(func_k)) * superchain_size / (superchain_size - 1)
    )  # variance withing k-th superchain
    W = jnp.average(W_k, axis=0)  # average within superchain variance
    B = jnp.var(func_k, axis=0, ddof=1)  # between superchain variance

    R = jnp.sqrt(1.0 + (B / W))  # splitR, shape = (# func)

    return R


def _gpdfit(exceedances: Array) -> tuple[Array, Array]:
    """Empirical Bayes GPD fit via Zhang & Stephens (2009).

    Fits a Generalised Pareto Distribution to ``exceedances`` (a 1-D array of
    non-negative values in *ascending* order) using the Bayesian model-averaging
    estimator.  All operations on fixed-shape arrays so the function is
    JIT-compatible (``exceedances.shape[0]`` must be static).
    """
    n = exceedances.shape[0]  # static at trace time
    prior_bs, prior_k = 3, 10
    m_est = 30 + int(n**0.5)  # static

    # Guard against the degenerate case where all exceedances are zero
    # (e.g. all importance weights equal).  In that case k=0, sigma=0
    # and the caller will leave the tail unchanged.
    tiny = jnp.finfo(exceedances.dtype).tiny
    is_degenerate = exceedances[-1] < tiny
    safe_exc = jnp.where(is_degenerate, jnp.ones_like(exceedances), exceedances)

    # Grid of candidate rate parameters b.
    b = 1.0 - jnp.sqrt(m_est / (jnp.arange(1, m_est + 1) - 0.5))
    b = b / (prior_bs * safe_exc[int(n / 4 + 0.5) - 1])
    b = b + 1.0 / safe_exc[-1]

    # k estimate for each b (mean log1p).
    k_ary = jnp.mean(jnp.log1p(-b[:, None] * safe_exc[None, :]), axis=1)

    # Profile log-likelihood weights.
    len_scale = n * (jnp.log(-b / k_ary) - k_ary - 1.0)
    w = jnp.exp(len_scale - jax.nn.logsumexp(len_scale))

    # Posterior mean of b, then derive k and sigma.
    b_post = jnp.sum(b * w)
    k = jnp.mean(jnp.log1p(-b_post * safe_exc))
    sigma = -k / b_post
    # Weakly informative prior shrinks k toward 0.5.
    k = (n * k + prior_k * 0.5) / (n + prior_k)

    k = jnp.where(is_degenerate, jnp.zeros(()), k)
    sigma = jnp.where(is_degenerate, jnp.zeros(()), sigma)
    return k, sigma


def _gpinv(p: Array, k: Array, sigma: Array) -> Array:
    """GPD quantile function (inverse CDF)."""
    return jnp.where(
        jnp.abs(k) < 1e-10,
        -sigma * jnp.log1p(-p),
        sigma * jnp.expm1(-k * jnp.log1p(-p)) / k,
    )


def psis_weights(log_ratios: Array, r_eff: float = 1.0) -> tuple[Array, Array]:
    """Pareto Smoothed Importance Sampling (PSIS) log weights.

    Implements the PSIS smoothing step from :cite:p:`vehtari2017practical`:
    the ``M`` largest importance ratios (in ratio space) are replaced by sorted
    Generalised Pareto quantiles fitted by the empirical Bayes estimator of
    Zhang & Stephens (2009), then all weights are normalised.

    This is a pure-JAX, JIT-compatible implementation faithful to Algorithm 1
    of Vehtari, Gelman & Gabry (2017).

    Parameters
    ----------
    log_ratios
        Log importance ratios ``log p(θ) − log q(θ)``, shape ``(n,)``.
        Need not be normalised.
    r_eff
        Relative effective sample size of the proposal, ``S_eff / n``.
        Use the default of ``1.0`` for i.i.d. draws (e.g. Pathfinder);
        set to the actual ESS ratio for correlated MCMC chains.  Values
        below 1 increase the tail size ``M`` to compensate for correlation.

    Returns
    -------
    log_weights
        Normalised log importance weights, shape ``(n,)``.
        ``jnp.exp(log_weights).sum() == 1`` up to floating-point precision.
    pareto_k
        Pareto shape parameter estimate (scalar ``Array``).  Values below 0.5
        indicate reliable estimates; 0.5–0.7 are moderate; above 0.7 may give
        unreliable estimates.  ``jnp.inf`` means the tail was too small to fit
        (fewer than 5 samples).

    Notes
    -----
    Tail size: ``M = min(floor(3*sqrt(n/r_eff)), n//5)``, matching the paper.
    The GPD is only applied when ``k >= 1/3``; lighter tails are left
    unsmoothed (only normalised).  Fitting uses empirical Bayes in
    importance-ratio space, the same approach as ArviZ.
    """
    n = log_ratios.shape[0]
    M = min(max(int(3.0 * (n / r_eff) ** 0.5), 5), n // 5)

    if M < 5:
        # Too few tail samples for a reliable GPD fit; return normalised
        # weights and k=inf to signal this to the caller.
        log_w = log_ratios - jax.nn.logsumexp(log_ratios)
        return log_w, jnp.asarray(jnp.inf)

    # Stabilise numerically.
    lw = log_ratios - log_ratios.max()

    # Sort ascending so that the M largest are at positions [n-M:].
    sorted_idx = jnp.argsort(lw)
    lw_sorted = lw[sorted_idx]

    # Threshold: largest value below the tail (in log and ratio space).
    threshold_log = lw_sorted[n - M - 1]
    threshold_ratio = jnp.exp(threshold_log)

    # Work in ratio (non-log) space for GPD fitting, as in the original paper.
    tail_ratio = jnp.exp(lw_sorted[n - M :])  # ascending, shape (M,)
    exceedances = tail_ratio - threshold_ratio  # >= 0, ascending

    k, sigma = _gpdfit(exceedances)

    # Uniform quantile positions within the tail: (0.5/M, 1.5/M, ..., (M-0.5)/M).
    p = (jnp.arange(M) + 0.5) / M  # ascending
    smoothed = threshold_ratio + _gpinv(p, k, sigma)
    # Cap smoothed values at the observed tail maximum.
    smoothed = jnp.minimum(smoothed, tail_ratio[-1])

    # Only replace the tail when k >= 1/3 (paper Algorithm 1, step 7).
    # For lighter tails the raw order statistics are already reliable.
    lw_smooth = jnp.where(
        k >= 1.0 / 3.0,
        lw_sorted.at[n - M :].set(jnp.log(smoothed)),
        lw_sorted,
    )

    # Restore original ordering and normalise.
    lw_orig = jnp.zeros_like(lw_smooth).at[sorted_idx].set(lw_smooth)
    log_w = lw_orig - jax.nn.logsumexp(lw_orig)
    return log_w, k
