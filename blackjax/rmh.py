"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, Optional, Union

import blackjax.inference.rmh.base as base
import blackjax.inference.rmh.proposals as proposals
from blackjax.base import SamplingAlgorithm, SamplingAlgorithmGenerator
from blackjax.types import Array, PyTree

__all__ = ["rmh"]


def rmh(
    logprob_fn: Callable, sigma: Optional[Array] = None
) -> Union[SamplingAlgorithm, SamplingAlgorithmGenerator]:
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
    value of `sigma` is specified. Otherwise a state initialization function and
    a kernel factory.

    """

    def init_fn(position: PyTree):
        return base.new_rmh_state(position, logprob_fn)

    def kernel_fn(sigma):
        proposal_generator = proposals.normal(sigma)
        kernel = base.rmh(logprob_fn, proposal_generator)
        return kernel

    if sigma is not None:
        kernel = kernel_fn(sigma)
        return SamplingAlgorithm(init_fn, kernel)
    else:
        return SamplingAlgorithmGenerator(init_fn, kernel_fn)
