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

"""
Partially adapted from:

Title: NumPyro diagnostics
Type: Source code
Availability: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/diagnostics.py 
"""

"""MCMC diagnostics."""
from collections import OrderedDict
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax import device_get
from jax.tree_util import tree_flatten
from scipy.fftpack import next_fast_len  # type: ignore

from blackjax.types import Array

__all__ = ["potential_scale_reduction", "effective_sample_size"]


def potential_scale_reduction(
    input_array: Array, chain_axis: int = 0, sample_axis: int = 1
):
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
    input_array: Array, chain_axis: int = 0, sample_axis: int = 1
):
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
        num_chains > 1
    ), "effective_sample_size as implemented only works for two or more chains."

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
        lambda _: weighted_var
        + mean_across_chain.var(axis=chain_axis, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
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


def split_rhat(input_array: Array, chain_axis: int = 0, sample_axis: int = 1):
    """
    Computes split R-hat over chains of samples ``input_array``, where the first dimension
    of ``input_array`` is chain dimension and the second dimension of ``input_array`` is draw dimension.
    It is required that ``input_array.shape[1] >= 4``.

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contain a chain dimension and a sample dimension.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    split R-hat of ``input_array``

    """
    assert input_array.ndim >= 2
    assert input_array.shape[1] >= 4

    N_half = input_array.shape[1] // 2
    new_input = jnp.concatenate(
        [input_array[:, :N_half], input_array[:, -N_half:]], axis=0
    )
    split_rhat = potential_scale_reduction(
        new_input, chain_axis=chain_axis, sample_axis=sample_axis
    )
    return split_rhat


def hpdi(input_array: Array, prob: float = 0.90, axis: int = 0):
    """
    Computes "highest posterior density interval" (HPDI) which is the narrowest
    interval with probability mass ``prob``.

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contain a chain dimension and a sample dimension.
    prob:
        The probability mass of samples within the interval. Defaults to 0.90.
    axis:
        The dimension to calculate hpdi

    Returns
    -------
    quantiles of ``x`` at ``(1 - prob) / 2`` and ``(1 + prob) / 2``.
    """
    input_array = jnp.swapaxes(input_array, axis, 0)
    sorted_x = jnp.sort(input_array, axis=0)
    mass = input_array.shape[0]
    index_length = int(prob * mass)
    intervals_left = sorted_x[: (mass - index_length)]
    intervals_right = sorted_x[index_length:]
    intervals_length = intervals_right - intervals_left
    index_start = intervals_length.argmin(axis=0)
    index_end = index_start + index_length
    hpd_left = jnp.take_along_axis(sorted_x, index_start[None, ...], axis=0)
    hpd_left = jnp.swapaxes(hpd_left, axis, 0)
    hpd_right = jnp.take_along_axis(sorted_x, index_end[None, ...], axis=0)
    hpd_right = jnp.swapaxes(hpd_right, axis, 0)
    return jnp.concatenate([hpd_left, hpd_right], axis=axis)


def summary(
    samples: Array, prob: float = 0.90, chain_axis: int = 0, sample_axis: int = 1
):
    """
    Returns a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval :func:`~blackjax.diagnostics.hpdi`,
    :func:`~blackjax.diagnostics.effective_sample_size`, and
    :func:`~blackjax.diagnostics.split_rhat`.

    Parameters
    ----------
    samples:
        An array representing multiple chains of MCMC samples. The array must
        contain a chain dimension and a sample dimension.
    prob:
        The probability mass of samples within the interval for calculating hpdi.
        Defaults to 0.90.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    An OrderedDict summary table containing diagnostics of ``samples`` from the posterior.

    """

    if not isinstance(samples, dict):
        samples = {f"Param:{i}": v for i, v in enumerate(tree_flatten(samples)[0])}

    summary_dict = {}
    for name, value in samples.items():
        value = device_get(value)
        value_flat = jnp.reshape(value, (-1,) + value.shape[2:])
        mean = value_flat.mean(axis=0)
        std = value_flat.std(axis=0, ddof=1)
        median = jnp.median(value_flat, axis=0)
        hpd = hpdi(value_flat, prob=prob)
        n_eff = effective_sample_size(
            value, chain_axis=chain_axis, sample_axis=sample_axis
        )
        r_hat = split_rhat(value, chain_axis=chain_axis, sample_axis=sample_axis)
        hpd_lower = f"{50 * (1 - prob):.1f}%"
        hpd_upper = f"{50 * (1 + prob):.1f}%"
        summary_dict[name] = OrderedDict(
            [
                ("mean", mean),
                ("std", std),
                ("median", median),
                (hpd_lower, hpd[0]),
                (hpd_upper, hpd[1]),
                ("n_eff", n_eff),
                ("r_hat", r_hat),
            ]
        )
    return summary_dict


def print_summary(
    samples: Array, prob: float = 0.90, chain_axis: int = 0, sample_axis: int = 10
):
    """
    Prints a summary of diagnostic statistics for the input samples.

    Parameters
    ----------
    samples: a collection of input samples with left most dimension is chain
        dimension and second to left most dimension is draw dimension.
    prob:
        The probability mass of samples within the interval for calculating hpdi.
        Defaults to 0.90.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
        None

    Notes
    -----
        Prints the mean, standard deviation, median, the prob Credibility Interval,
        potential_scale_reduction, and effective_sample_size for each parameter.
        If group_by_chain is True, prints the statistics for each chain separately.
    """
    if not isinstance(samples, dict):
        samples = {f"Param:{i}": v for i, v in enumerate(tree_flatten(samples)[0])}
    summary_dict = summary(samples, prob, chain_axis, sample_axis)
    num_columns = len(
        summary_dict[list(summary_dict.keys())[0]]
    )  # number of columns for the first param

    row_names = {
        k: k + "[" + ",".join(map(lambda x: str(x - 1), v.shape[2:])) + "]"
        for k, v in samples.items()
    }
    max_len = max(max(map(lambda x: len(x), row_names.values())), 10)
    name_format = "{:>" + str(max_len) + "}"
    header_format = name_format + " {:>9}" * num_columns
    columns = [""] + list(list(summary_dict.values())[0].keys())

    print()
    print(header_format.format(*columns))

    row_format = name_format + " {:>9.2f}" * num_columns
    for name, stats_dict in summary_dict.items():
        shape = stats_dict["mean"].shape
        if len(shape) == 0:
            print(row_format.format(name, *stats_dict.values()))
        else:
            for idx in product(*map(range, shape)):
                idx_str = "[{}]".format(",".join(map(str, idx)))
                print(
                    row_format.format(
                        name + idx_str, *[v[idx] for v in stats_dict.values()]
                    )
                )
    print()
