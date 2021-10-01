"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable

import blackjax.inference.rmh.base as base
import blackjax.inference.rmh.proposals as proposals
from blackjax.types import Array

__all__ = ["new_state", "kernel"]


RWMHState = base.RMHState
RWMHInfo = base.RMHInfo
new_state = base.new_rmh_state


def kernel(logprob_fn: Callable, sigma: Array):
    """Random Walk Rosenbluth-Metropolis-Hastings algorithm with normal proposals.

    We currently only support a Gaussian proposal but the algorithm could easily
    be extended to include other proposals.

    Parameters
    ----------
    logprob_fn
        Log probability function we wish to sample from
    sigma
        Covariance matrix for the gaussian proposal distribution.

    """
    proposal_generator = proposals.normal(sigma)
    kernel = base.rmh(logprob_fn, proposal_generator)
    return kernel
