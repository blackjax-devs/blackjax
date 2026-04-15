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
"""Dynamic HMC on the Laplace-approximated marginal log-density.

Combines the warm-started Laplace marginalisation of
:mod:`~blackjax.mcmc.laplace_hmc` with the quasi-random integration-step
schedule of :mod:`~blackjax.mcmc.dynamic_hmc`.

The state carries both extra fields:

- ``theta_star``: MAP of latent variables at the current ``phi``, used to
  warm-start L-BFGS at every leapfrog step.
- ``random_generator_arg``: Halton index (or PRNG key) used by
  ``integration_steps_fn`` to draw the number of leapfrog steps each
  transition.

Two variants are available at the top level:

+---------------------------+------------------+------------------------------+
| Alias                     | Proposal         | Notes                        |
+===========================+==================+==============================+
| ``blackjax.laplace_dhmc`` | endpoint + M-H   | default                      |
| ``blackjax.laplace_dmhmc``| full trajectory  | multinomial, no rejection    |
+---------------------------+------------------+------------------------------+

Typical usage::

    sampler = blackjax.laplace_dhmc(
        log_joint, theta_init=jnp.zeros(n),
        step_size=0.1, inverse_mass_matrix=jnp.ones(d),
    )
    state = sampler.init(phi_init, rng_key)
    new_state, info = jax.jit(sampler.step)(rng_key, state)
    # new_state.theta_star  — MAP latent at accepted phi
    # new_state.random_generator_arg  — advanced Halton index
"""
from typing import Callable, NamedTuple

import jax

import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.laplace_marginal import LaplaceMarginal, laplace_marginal_factory
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "LaplaceDynamicHMCState",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class LaplaceDynamicHMCState(NamedTuple):
    """State of the Laplace dynamic HMC sampler.

    position
        Current hyperparameter position ``phi``.
    logdensity
        Current value of the Laplace log-marginal ``log p̂(phi | y)``.
    logdensity_grad
        Gradient of ``log p̂(phi | y)`` w.r.t. ``phi``.
    theta_star
        MAP of the latent variables at the current ``phi``.  Warm-starts
        L-BFGS at every leapfrog step.
    random_generator_arg
        Halton index or PRNG key consumed by ``integration_steps_fn`` to
        draw the number of leapfrog steps for the next transition.
    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    theta_star: ArrayTree
    random_generator_arg: Array


def init(
    position: ArrayLikeTree,
    laplace: LaplaceMarginal,
    random_generator_arg: Array,
) -> LaplaceDynamicHMCState:
    """Create an initial :class:`LaplaceDynamicHMCState`.

    Parameters
    ----------
    position
        Initial hyperparameter value ``phi``.
    laplace
        A :class:`~blackjax.mcmc.laplace_marginal.LaplaceMarginal` instance.
    random_generator_arg
        Initial value for the quasi-random step-count generator (e.g. a
        PRNG key or Halton index).  When called via the top-level API this
        is seeded automatically from the ``rng_key`` passed to ``.init``.
    """
    (logdensity, theta_star), logdensity_grad = jax.value_and_grad(
        laplace, has_aux=True
    )(position)
    return LaplaceDynamicHMCState(
        position, logdensity, logdensity_grad, theta_star, random_generator_arg
    )


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    build_proposal: Callable = hmc.hmc_proposal,
) -> Callable:
    """Build the Laplace dynamic HMC kernel.

    Parameters
    ----------
    integrator
        Symplectic integrator for the leapfrog trajectory.
    divergence_threshold
        Energy difference above which a transition is declared divergent.
    next_random_arg_fn
        Advances ``random_generator_arg`` each step.
    integration_steps_fn
        Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
        that draws the number of leapfrog steps for a single transition.
        Extra positional arguments are supplied at call time via
        ``integration_steps_params`` on the inner kernel.
    build_proposal
        Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
        (endpoint + M-H).  Pass :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal`
        for multinomial trajectory sampling (``blackjax.laplace_dmhmc``).

    Returns
    -------
    A kernel
    ``(rng_key, state, laplace, step_size, inverse_mass_matrix) -> (LaplaceDynamicHMCState, HMCInfo)``.
    """
    dynamic_kernel = dynamic_hmc.build_kernel(
        integrator,
        divergence_threshold,
        next_random_arg_fn,
        integration_steps_fn,
        build_proposal,
    )

    def kernel(
        rng_key: PRNGKey,
        state: LaplaceDynamicHMCState,
        laplace: LaplaceMarginal,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        integration_steps_params: tuple = (),
    ) -> tuple[LaplaceDynamicHMCState, hmc.HMCInfo]:
        """One Laplace dynamic-HMC transition."""
        theta_prev = state.theta_star

        def logdensity_fn(phi):
            lp, _ = laplace(phi, theta_prev)
            return lp

        dynamic_state = DynamicHMCState(
            state.position,
            state.logdensity,
            state.logdensity_grad,
            state.random_generator_arg,
        )
        new_dynamic_state, info = dynamic_kernel(
            rng_key,
            dynamic_state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            integration_steps_params,
        )

        new_theta_star = laplace.solve_theta(new_dynamic_state.position, theta_prev)

        new_state = LaplaceDynamicHMCState(
            new_dynamic_state.position,
            new_dynamic_state.logdensity,
            new_dynamic_state.logdensity_grad,
            new_theta_star,
            new_dynamic_state.random_generator_arg,
        )
        return new_state, info

    return kernel


def as_top_level_api(
    log_joint_fn: Callable,
    theta_init: ArrayLikeTree,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    integration_steps_params: tuple = (),
    build_proposal: Callable = hmc.hmc_proposal,
    **optimizer_kwargs,
) -> SamplingAlgorithm:
    """Dynamic HMC on the Laplace-approximated marginal log-density.

    Combines Laplace marginalisation over latent variables with a
    quasi-random number of leapfrog steps per transition, reducing
    periodic-orbit sensitivity while retaining the computational benefits
    of operating on the low-dimensional hyperparameter marginal.

    Parameters
    ----------
    log_joint_fn
        ``(theta, phi) -> float``.  Full log joint ``log p(theta, phi, y)``.
    theta_init
        Initial guess for theta; fixes the latent PyTree structure.
    step_size
        Leapfrog step size.
    inverse_mass_matrix
        Inverse mass matrix (1-D array for diagonal, scalar for isotropic).
    divergence_threshold
        Absolute energy difference above which a transition is divergent.
    integrator
        Symplectic integrator.  Default: velocity Verlet.
    next_random_arg_fn
        Advances ``random_generator_arg`` each step.
    integration_steps_fn
        Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
        that draws the number of leapfrog steps for a single transition.
    integration_steps_params
        Extra positional arguments unpacked into ``integration_steps_fn`` after
        ``random_generator_arg`` on every step.  Defaults to ``()`` so that a
        plain 1-arg ``integration_steps_fn`` works unchanged.
    build_proposal
        Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
        (``blackjax.laplace_dhmc``).  Pass
        :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal` for
        ``blackjax.laplace_dmhmc``.
    **optimizer_kwargs
        Forwarded to :func:`~blackjax.optimizers.lbfgs.minimize_lbfgs`.
        Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``.

    Returns
    -------
    A :class:`~blackjax.base.SamplingAlgorithm` whose ``step`` returns a
    :class:`LaplaceDynamicHMCState` and :class:`~blackjax.mcmc.hmc.HMCInfo`.

    Examples
    --------
    .. code::

        sampler = blackjax.laplace_dhmc(
            log_joint, theta_init=jnp.zeros(n_latent),
            step_size=0.1, inverse_mass_matrix=jnp.ones(d_phi),
            maxiter=100,
        )
        state = sampler.init(phi_init, rng_key)
        new_state, info = jax.jit(sampler.step)(rng_key, state)
        print(new_state.theta_star)          # MAP latent at new phi
        print(new_state.random_generator_arg)  # advanced Halton index
    """
    laplace = laplace_marginal_factory(log_joint_fn, theta_init, **optimizer_kwargs)
    kernel = build_kernel(
        integrator,
        divergence_threshold,
        next_random_arg_fn,
        integration_steps_fn,
        build_proposal,
    )
    return build_sampling_algorithm(
        kernel,
        init,
        laplace,
        kernel_args=(step_size, inverse_mass_matrix, integration_steps_params),
        pass_rng_key_to_init=True,
    )
