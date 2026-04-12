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
"""HMC on the Laplace-approximated marginal log-density with warm-starting.

Wraps the composable :func:`~blackjax.mcmc.laplace_marginal.laplace_marginal_factory`
in a standard BlackJAX three-layer sampler that carries the MAP latent variables
``theta_star`` through the chain.  At each step ``theta_star`` is used as the
warm-start hint for the L-BFGS solver at every leapfrog evaluation, so the
optimizer needs only a handful of iterations when ``phi`` moves by a small amount.

The proposal strategy is swappable via ``build_proposal``, giving two usable variants:

+---------------------------+------------------+------------------------------+
| Alias                     | Proposal         | Notes                        |
+===========================+==================+==============================+
| ``blackjax.laplace_hmc``  | endpoint + M-H   | default, standard HMC        |
| ``blackjax.laplace_mhmc`` | full trajectory | better ESS per gradient |
+---------------------------+------------------+------------------------------+

Typical usage::

    sampler = blackjax.laplace_hmc(
        log_joint, theta_init=jnp.zeros(n),
        step_size=0.1, inverse_mass_matrix=jnp.ones(d),
        num_integration_steps=10,
    )
    state = sampler.init(phi_init)
    new_state, info = jax.jit(sampler.step)(rng_key, state)
    # new_state.theta_star: MAP of theta at the accepted phi

    # Multinomial variant (no rejection step, samples from full trajectory):
    sampler = blackjax.laplace_mhmc(
        log_joint, theta_init=jnp.zeros(n),
        step_size=0.1, inverse_mass_matrix=jnp.ones(d),
        num_integration_steps=10,
    )
"""
from typing import Callable, NamedTuple

import jax

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.laplace_marginal import LaplaceMarginal, laplace_marginal_factory
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "LaplaceHMCState",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class LaplaceHMCState(NamedTuple):
    """State of the Laplace-HMC sampler.

    position
        Current hyperparameter position ``phi``.  Can be any PyTree.
    logdensity
        Current value of the Laplace log-marginal ``log p̂(phi | y)``.
    logdensity_grad
        Gradient of ``log p̂(phi | y)`` w.r.t. ``phi``.  Same PyTree structure
        as ``position``.
    theta_star
        MAP of the latent variables at the current ``phi``, i.e.
        ``theta*(phi) = argmax_theta log_joint(theta, phi)``.  Carried through
        the chain so the next L-BFGS solve can warm-start from here.
    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    theta_star: ArrayTree


def init(
    position: ArrayLikeTree,
    laplace: LaplaceMarginal,
) -> LaplaceHMCState:
    """Create an initial :class:`LaplaceHMCState`.

    Runs L-BFGS from cold start to find ``theta*(position)``, then evaluates
    the Laplace log-marginal and its gradient.

    Parameters
    ----------
    position
        Initial hyperparameter value ``phi``.
    laplace
        A :class:`~blackjax.mcmc.laplace_marginal.LaplaceMarginal` instance
        returned by :func:`~blackjax.mcmc.laplace_marginal.laplace_marginal_factory`.
    """
    (logdensity, theta_star), logdensity_grad = jax.value_and_grad(
        laplace, has_aux=True
    )(position)
    return LaplaceHMCState(position, logdensity, logdensity_grad, theta_star)


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    build_proposal: Callable = hmc.hmc_proposal,
) -> Callable:
    """Build the Laplace-HMC kernel.

    Parameters
    ----------
    integrator
        Symplectic integrator used for the HMC trajectory.
    divergence_threshold
        Energy difference above which a transition is declared divergent.
    build_proposal
        Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
        (endpoint + M-H).  Pass :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal`
        for multinomial trajectory sampling (``blackjax.laplace_mhmc``).

    Returns
    -------
    A kernel ``(rng_key, state, laplace, step_size, inverse_mass_matrix, num_integration_steps) -> (LaplaceHMCState, HMCInfo)``.
    """
    hmc_kernel = hmc.build_kernel(integrator, divergence_threshold, build_proposal)

    def kernel(
        rng_key: PRNGKey,
        state: LaplaceHMCState,
        laplace: LaplaceMarginal,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        num_integration_steps: int,
    ) -> tuple[LaplaceHMCState, hmc.HMCInfo]:
        """One Laplace-HMC transition.

        All log-density evaluations during the HMC trajectory warm-start
        L-BFGS from ``state.theta_star``.  After accept/reject, L-BFGS is
        called once more at the accepted position to refresh ``theta_star``
        for the next step.
        """
        theta_prev = state.theta_star

        # Close over theta_prev: every leapfrog evaluation warm-starts from here.
        def logdensity_fn(phi):
            lp, _ = laplace(phi, theta_prev)
            return lp

        hmc_state = hmc.HMCState(
            state.position, state.logdensity, state.logdensity_grad
        )
        new_hmc_state, info = hmc_kernel(
            rng_key,
            hmc_state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )

        # Refresh theta_star at the accepted position.  Use theta_prev as warm
        # start — cheap when phi moved only slightly (typical HMC behaviour).
        new_theta_star = laplace.solve_theta(new_hmc_state.position, theta_prev)

        new_state = LaplaceHMCState(
            new_hmc_state.position,
            new_hmc_state.logdensity,
            new_hmc_state.logdensity_grad,
            new_theta_star,
        )
        return new_state, info

    return kernel


def as_top_level_api(
    log_joint_fn: Callable,
    theta_init: ArrayLikeTree,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    num_integration_steps: int,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
    build_proposal: Callable = hmc.hmc_proposal,
    **optimizer_kwargs,
) -> SamplingAlgorithm:
    """HMC on the Laplace-approximated marginal log-density.

    For a hierarchical model ``log p(theta, phi, y)``, integrates out the
    latent variables ``theta`` via the Laplace approximation and runs HMC on
    the resulting marginal over the hyperparameters ``phi``.

    Gradients w.r.t. ``phi`` are computed via the implicit function theorem
    (:func:`jax.lax.custom_root`) — the L-BFGS iterations are *not* unrolled.
    ``theta*(phi)`` is warm-started from the previous MCMC state, reducing the
    number of L-BFGS iterations needed at each leapfrog step.

    Parameters
    ----------
    log_joint_fn
        ``(theta, phi) -> float``.  The full log joint ``log p(theta, phi, y)``.
        Both arguments may be arbitrary PyTrees.  Must be at least C³ in theta.
    theta_init
        Initial guess for theta.  Fixes the PyTree structure for all calls.
    step_size
        HMC leapfrog step size.
    inverse_mass_matrix
        Inverse mass matrix for HMC (1-D array for diagonal, scalar for isotropic).
    num_integration_steps
        Number of leapfrog steps per HMC transition.
    divergence_threshold
        Absolute energy difference above which a transition is declared divergent.
        Default 1000.
    integrator
        Symplectic integrator.  Default: velocity Verlet.
    build_proposal
        Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
        (endpoint + M-H).  Pass :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal`
        for multinomial trajectory sampling; this is what
        ``blackjax.laplace_mhmc`` uses.
    **optimizer_kwargs
        Forwarded to :func:`~blackjax.optimizers.lbfgs.minimize_lbfgs`.
        Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``.

    Returns
    -------
    A :class:`~blackjax.base.SamplingAlgorithm` whose ``step`` returns a
    :class:`LaplaceHMCState` (with ``theta_star`` field) and
    :class:`~blackjax.mcmc.hmc.HMCInfo`.

    Examples
    --------
    .. code::

        sampler = blackjax.laplace_hmc(
            log_joint, theta_init=jnp.zeros(n_latent),
            step_size=0.1, inverse_mass_matrix=jnp.ones(d_phi),
            num_integration_steps=10, maxiter=100,
        )
        state = sampler.init(phi_init)
        new_state, info = jax.jit(sampler.step)(rng_key, state)
        print(new_state.theta_star)   # MAP latent at the new phi
    """
    laplace = laplace_marginal_factory(log_joint_fn, theta_init, **optimizer_kwargs)
    kernel = build_kernel(integrator, divergence_threshold, build_proposal)
    return build_sampling_algorithm(
        kernel,
        init,
        laplace,
        kernel_args=(step_size, inverse_mass_matrix, num_integration_steps),
    )
