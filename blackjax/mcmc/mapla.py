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
"""Public API for Metropolis-adjusted Preconditioned Langevin Algorithm (MAPLA)."""
from typing import Callable

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.constraints import barrier_hessian
from blackjax.mcmc.diffusions import DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance
from blackjax.mcmc.smmala import build_kernel, init
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = ["as_top_level_api"]


def as_top_level_api(
    logdensity_fn: Callable,
    constraint,
    step_size: float,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the MAPLA kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    constraint
        A ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
        or ``ComposedConstraint`` from ``blackjax.mcmc.constraints`` describing
        the feasible region.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    References
    ----------
    .. [1] "High-accuracy sampling from constrained spaces with the Metropolis-adjusted Preconditioned Langevin Algorithm"
        (https://proceedings.mlr.press/v272/srinivasan25a.html)

    """
    mass_matrix_fn = lambda position: DiffusionMetric(
        *_format_covariance(barrier_hessian(constraint, position), is_inv=False)[:2]
    )
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, mass_matrix_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            mass_matrix_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)
