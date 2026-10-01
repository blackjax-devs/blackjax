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
"""Public API for the Dikin walk algorithm."""
from typing import Callable

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.constraints import barrier_hessian
from blackjax.mcmc.diffusions import DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance
from blackjax.mcmc.posdep_rwmh import build_kernel, init
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = [
    "init",
    "build_kernel",
    "as_top_level_api",
]


def dikin_metric(constraint):
    """Builds the Dikin metric callable for the given constraint.

    The Dikin metric is the Hessian of the log-barrier
    ``−Σ_i log(−c_i(x))``, which equals ``A^T diag(s)^{-2} A`` for
    linear constraints ``Ax ≤ b`` with slacks ``s = b − Ax``.
    """
    return lambda x: barrier_hessian(constraint, x)


def as_top_level_api(logdensity_fn: Callable, constraint, step_size) -> SamplingAlgorithm:
    """Implements the user interface for the Dikin walk.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
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
    .. [1] "Random walks on polytopes and an affine interior point method for linear programming"
        (https://dl.acm.org/doi/10.1145/1536414.1536491)

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
