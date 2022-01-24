"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, Tuple

import jax

import blackjax.inference.rmh.base as base
import blackjax.inference.rmh.proposals as proposals
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["rmh"]


def rmh(logprob_fn: Callable, sigma: Array) -> SamplingAlgorithm:
    """Random Walk Rosenbluth-Metropolis-Hastings algorithm with normal proposals.

    We currently only support a Gaussian proposal but the algorithm could easily
    be extended to include other proposals.

    Parameters
    ----------
    logprob_fn
        Log probability function we wish to sample from
    sigma
        Covariance matrix for the gaussian proposal distribution.

    Returns
    -------
    A `SamplingAlgorithm` with a state initialization and a step function if the
    value of `sigma` is specified.

    """
    kernel = rmh_kernel()

    def init_fn(position: PyTree):
        return jax.jit(rmh_init, static_argnums=(1,))(position, logprob_fn)

    def step_fn(
        rng_key: PRNGKey, state: base.RMHState
    ) -> Tuple[base.RMHState, base.RMHInfo]:
        # `np.ndarray` and `DeviceArray`s are not hashable and thus cannot be used as static arguments.`
        # Workaround: https://github.com/google/jax/issues/4572#issuecomment-709809897
        kernel_fn = jax.jit(kernel, static_argnums=(2))
        return kernel_fn(rng_key, state, logprob_fn, sigma)

    return SamplingAlgorithm(init_fn, step_fn)


def rmh_init(position: PyTree, logprob_fn: Callable):
    return base.new_rmh_state(position, logprob_fn)


def rmh_kernel():
    def kernel(
        rng_key: PRNGKey, state: base.RMHState, logprob_fn: Callable, sigma: Array
    ) -> Tuple[base.RMHState, base.RMHInfo]:
        proposal_generator = proposals.normal(sigma)
        kernel = base.rmh(logprob_fn, proposal_generator)
        return kernel(rng_key, state)

    return kernel
