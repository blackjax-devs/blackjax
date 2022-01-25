"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, Tuple

import blackjax.inference.rmh.proposals as proposals
from blackjax.inference.rmh.base import RMHInfo, RMHState, new_rmh_state, rmh
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["rmh_init", "rmh_kernel"]


def rmh_init(position: PyTree, logprob_fn: Callable):
    return new_rmh_state(position, logprob_fn)


def rmh_kernel():
    def kernel(
        rng_key: PRNGKey, state: RMHState, logprob_fn: Callable, sigma: Array
    ) -> Tuple[RMHState, RMHInfo]:
        proposal_generator = proposals.normal(sigma)
        kernel = rmh(logprob_fn, proposal_generator)
        return kernel(rng_key, state)

    return kernel
