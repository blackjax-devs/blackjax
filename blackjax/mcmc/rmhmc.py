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

from typing import Callable

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc import hmc

__all__ = ["init", "build_kernel", "as_top_level_api"]


init = hmc.init
build_kernel = hmc.build_kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    mass_matrix: metrics.Metric | Callable,
    num_integration_steps: int,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.implicit_midpoint,
) -> SamplingAlgorithm:
    """A Riemannian Manifold Hamiltonian Monte Carlo kernel

    Of note, this kernel is simply an alias of the ``hmc`` kernel with a
    different choice of default integrator (``implicit_midpoint`` instead of
    ``velocity_verlet``) since RMHMC is typically used for Hamiltonian systems
    that are not separable.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    mass_matrix
        A function which computes the mass matrix (not inverse) at a given
        position when drawing a value for the momentum and computing the kinetic
        energy. In practice, this argument will be passed to the
        ``metrics.default_metric`` function so it supports all the options
        discussed there.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the
        trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(integrator, divergence_threshold)
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(step_size, mass_matrix, num_integration_steps),
    )
