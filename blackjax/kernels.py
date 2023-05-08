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
"""Blackjax high-level interface with sampling algorithms."""
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from optax import GradientTransformation

import blackjax.adaptation as adaptation
import blackjax.mcmc as mcmc
import blackjax.mcmc.random_walk
import blackjax.sgmcmc as sgmcmc
import blackjax.smc as smc
import blackjax.vi as vi
from blackjax.base import AdaptationAlgorithm, MCMCSamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, PRNGKey, PyTree

__all__ = [
    "adaptive_tempered_smc",
    "hmc",
    "mala",
    "nuts",
    "ghmc",
    "orbital_hmc",
    "additive_step_random_walk",
    "rmh",
    "sgld",
    "sghmc",
    "meanfield_vi",
    "csgld",
    "elliptical_slice",
    "meads_adaptation",
    "tempered_smc",
    "window_adaptation",
    "irmh",
    "pathfinder",
    "pathfinder_adaptation",
    "mgrad_gaussian",
]


# -----------------------------------------------------------------------------
#                           SEQUENTIAL MONTE CARLO
# -----------------------------------------------------------------------------


class adaptive_tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.
    resampling_fn
        The function used to resample the particles.
    target_ess
        The number of effective sample size to aim for at each step.
    root_solver
        The solver used to adaptively compute the temperature given a target number
        of effective samples.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    init = staticmethod(smc.tempered.init)
    build_kernel = staticmethod(smc.adaptive_tempered.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        target_ess: float,
        root_solver: Callable = smc.solver.dichotomy,
        num_mcmc_steps: int = 10,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resampling_fn,
            target_ess,
            root_solver,
        )

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                num_mcmc_steps,
                mcmc_parameters,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.
    resampling_fn
        The function used to resample the particles.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    init = staticmethod(smc.tempered.init)
    build_kernel = staticmethod(smc.tempered.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        num_mcmc_steps: int = 10,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resampling_fn,
        )

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state, lmbda):
            return kernel(
                rng_key,
                state,
                num_mcmc_steps,
                lmbda,
                mcmc_parameters,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#                         MARKOV CHAIN MONTE CARLO
# -----------------------------------------------------------------------------


class hmc:
    """Implements the (basic) user interface for the HMC kernel.

    The general hmc kernel builder (:meth:`blackjax.mcmc.hmc.build_kernel`, alias `blackjax.hmc.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new HMC kernel can be initialized and used with the following code:

    .. code::

        hmc = blackjax.hmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
        state = hmc.init(position)
        new_state, info = hmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(hmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.hmc.build_kernel(integrators.mclachlan)
       state = blackjax.hmc.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.\

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(mcmc.hmc.init)
    build_kernel = staticmethod(mcmc.hmc.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = mcmc.integrators.velocity_verlet,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold)

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class mala:
    """Implements the (basic) user interface for the MALA kernel.

    The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new MALA kernel can be initialized and used with the following code:

    .. code::

        mala = blackjax.mala(logdensity_fn, step_size)
        state = mala.init(position)
        new_state, info = mala.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(mala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.mala.build_kernel(logdensity_fn)
       state = blackjax.mala.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    init = staticmethod(mcmc.mala.init)
    build_kernel = staticmethod(mcmc.mala.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, step_size)

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class nuts:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code::

        nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code::

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    You can always use the base kernel should you need to:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.nuts.build_kernel(integrators.yoshida)
       state = blackjax.nuts.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    max_num_doublings
        The maximum number of times we double the length of the trajectory before
        returning if no U-turn has been obserbed or no divergence has occured.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    init = staticmethod(mcmc.hmc.init)
    build_kernel = staticmethod(mcmc.nuts.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        *,
        max_num_doublings: int = 10,
        divergence_threshold: int = 1000,
        integrator: Callable = mcmc.integrators.velocity_verlet,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold, max_num_doublings)

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class mgrad_gaussian:
    """Implements the marginal sampler for latent Gaussian model of :cite:p:`titsias2018auxiliary`.

    It uses a first order approximation to the log_likelihood of a model with Gaussian prior.
    Interestingly, the only parameter that needs calibrating is the "step size" delta, which can be done very efficiently.
    Calibrating it to have an acceptance rate of roughly 50% is a good starting point.

    Examples
    --------
    A new marginal latent Gaussian MCMC kernel for a model q(x) ∝ exp(f(x)) N(x; m, C) can be initialized and
    used for a given "step size" delta with the following code:

    .. code::

        mgrad_gaussian = blackjax.mgrad_gaussian(f, C, use_inverse=False, mean=m)
        state = mgrad_gaussian.init(zeros)  # Starting at the mean of the prior
        new_state, info = mgrad_gaussian.step(rng_key, state, delta)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(mgrad_gaussian.step)
        new_state, info = step(rng_key, state, delta)

    Parameters
    ----------
    logdensity_fn
        The logarithm of the likelihood function for the latent Gaussian model.
    covariance
        The covariance of the prior Gaussian density.
    mean: optional
        Mean of the prior Gaussian density. Default is zero.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        covariance: Array,
        mean: Optional[Array] = None,
    ) -> MCMCSamplingAlgorithm:
        init, kernel = mcmc.marginal_latent_gaussian.init_and_kernel(
            logdensity_fn, covariance, mean
        )

        def init_fn(position: Array):
            return init(position)

        def step_fn(rng_key: PRNGKey, state, delta: float):
            return kernel(
                rng_key,
                state,
                delta,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#                        STOCHASTIC GRADIENT MCMC
# -----------------------------------------------------------------------------


class sgld:
    """Implements the (basic) user interface for the SGLD kernel.

    The general sgld kernel builder (:meth:`blackjax.mcmc.sgld.build_kernel`, alias
    `blackjax.sgld.build_kernel`) can be cumbersome to manipulate. Since most users
    only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    Example
    -------

    To initialize a SGLD kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    We can now initialize the sgld kernel and the state:

    .. code::

        sgld = blackjax.sgld(grad_fn)

    Assuming we have an iterator `batches` that yields batches of data we can
    perform one step:

    .. code::

        step_size = 1e-3
        minibatch = next(batches)
        new_position = sgld.step(rng_key, position, minibatch, step_size)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(sgld.step)
       new_position, info = step(rng_key, position, minibatch, step_size)

    Parameters
    ----------
    grad_estimator
       A function that takes a position, a batch of data and returns an estimation
       of the gradient of the log-density at this position.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    build_kernel = staticmethod(sgmcmc.sgld.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        grad_estimator: Callable,
    ) -> Callable:
        kernel = cls.build_kernel()

        def step_fn(
            rng_key: PRNGKey,
            state,
            minibatch: PyTree,
            step_size: float,
            temperature: float = 1,
        ):
            return kernel(
                rng_key, state, grad_estimator, minibatch, step_size, temperature
            )

        return step_fn


class sghmc:
    """Implements the (basic) user interface for the SGHMC kernel.

    The general sghmc kernel builder (:meth:`blackjax.mcmc.sghmc.build_kernel`, alias
    `blackjax.sghmc.build_kernel`) can be cumbersome to manipulate. Since most users
    only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    Example
    -------

    To initialize a SGHMC kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    We can now initialize the sghmc kernel and the state. Like HMC, SGHMC needs the user to specify a number of integration steps.

    .. code::

        sghmc = blackjax.sghmc(grad_estimator, num_integration_steps)

    Assuming we have an iterator `batches` that yields batches of data we can
    perform one step:

    .. code::

        step_size = 1e-3
        minibatch = next(batches)
        new_position = sghmc.step(rng_key, position, minibatch, step_size)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(sghmc.step)
       new_position, info = step(rng_key, position, minibatch, step_size)

    Parameters
    ----------
    grad_estimator
       A function that takes a position, a batch of data and returns an estimation
       of the gradient of the log-density at this position.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    build_kernel = staticmethod(sgmcmc.sghmc.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        grad_estimator: Callable,
        num_integration_steps: int = 10,
    ) -> Callable:
        kernel = cls.build_kernel()

        def step_fn(rng_key: PRNGKey, state, minibatch: PyTree, step_size: float):
            return kernel(
                rng_key,
                state,
                grad_estimator,
                minibatch,
                step_size,
                num_integration_steps,
            )

        return step_fn


class csgld:
    r"""Implements the (basic) user interface for the Contour SGLD kernel.

    Parameters
    ----------
    logdensity_estimator
        A function that returns an estimation of the model's logdensity given
        a position and a batch of data.
    gradient_estimator
        A function that takes a position, a batch of data and returns an estimation
        of the gradient of the log-density at this position.
    zeta
        Hyperparameter that controls the geometric property of the flattened
        density. If `zeta=0` the function reduces to the SGLD step function.
    temperature
        Temperature parameter.
    num_partitions
        The number of partitions we divide the energy landscape into.
    energy_gap
        The difference in energy :math:`\Delta u` between the successive
        partitions. Can be determined by running e.g. an optimizer to determine
        the range of energies. `num_partition` * `energy_gap` should match this
        range.
    min_energy
        A rough estimate of the minimum energy in a dataset, which should be
        strictly smaller than the exact minimum energy! e.g. if the minimum
        energy of a dataset is 3456, we can set min_energy to be any value
        smaller than 3456. Set it to 0 is acceptable, but not efficient enough.
        the closer the gap between min_energy and 3456 is, the better.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """
    init = staticmethod(sgmcmc.csgld.init)
    build_kernel = staticmethod(sgmcmc.csgld.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_estimator: Callable,
        gradient_estimator: Callable,
        zeta: float = 1,
        temperature: float = 0.01,
        num_partitions: int = 512,
        energy_gap: float = 100,
        min_energy: float = 0,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(num_partitions, energy_gap, min_energy)

        def init_fn(position: PyTree):
            return cls.init(position, num_partitions)

        def step_fn(
            rng_key: PRNGKey,
            state,
            minibatch: PyTree,
            step_size_diff: float,
            step_size_stoch: float,
        ):
            return kernel(
                rng_key,
                state,
                logdensity_estimator,
                gradient_estimator,
                minibatch,
                step_size_diff,
                step_size_stoch,
                zeta,
                temperature,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#                                 ADAPTATION
# -----------------------------------------------------------------------------


class AdaptationResults(NamedTuple):
    state: PyTree
    parameters: dict


class AdaptationInfo(NamedTuple):
    state: NamedTuple
    info: NamedTuple
    adaptation_state: NamedTuple


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logdensity_fn: Callable,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult` object.

    """

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_step, adapt_final = adaptation.window_adaptation.base(
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.acceptance_rate,
        )

        return (
            (new_state, new_adaptation_state),
            AdaptationInfo(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: PyTree, num_steps: int = 1000):
        init_state = algorithm.init(position, logdensity_fn)
        init_adaptation_state = adapt_init(position, initial_step_size)

        if progress_bar:
            print("Running window adaptation")
            one_step_ = jax.jit(progress_bar_scan(num_steps)(one_step))
        else:
            one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_steps)
        schedule = adaptation.window_adaptation.schedule(num_steps)
        last_state, info = jax.lax.scan(
            one_step_,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule),
        )
        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)


def meads_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
) -> AdaptationAlgorithm:
    """Adapt the parameters of the Generalized HMC algorithm.

    The Generalized HMC algorithm depends on three parameters, each controlling
    one element of its behaviour: step size controls the integrator's dynamics,
    alpha controls the persistency of the momentum variable, and delta controls
    the deterministic transformation of the slice variable used to perform the
    non-reversible Metropolis-Hastings accept/reject step.

    The step size parameter is chosen to ensure the stability of the velocity
    verlet integrator, the alpha parameter to make the influence of the current
    state on future states of the momentum variable to decay exponentially, and
    the delta parameter to maximize the acceptance of proposal but with good
    mixing properties for the slice variable. These characteristics are targeted
    by controlling heuristics based on the maximum eigenvalues of the correlation
    and gradient matrices of the cross-chain samples, under simpifyng assumptions.

    Good tuning is fundamental for the non-reversible Generalized HMC sampling
    algorithm to explore the target space efficienty and output uncorrelated, or
    as uncorrelated as possible, samples from the target space. Furthermore, the
    single integrator step of the algorithm lends itself for fast sampling
    on parallel computer architectures.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of chains used for cross-chain warm-up training.

    Returns
    -------
    A function that returns the last cross-chain state, a sampling kernel with the
    tuned parameter values, and all the warm-up states for diagnostics.

    """

    ghmc_kernel = ghmc.build_kernel()

    adapt_init, adapt_update = adaptation.meads_adaptation.base()

    batch_init = jax.vmap(lambda r, p: ghmc.init(r, p, logdensity_fn))

    def one_step(carry, rng_key):
        states, adaptation_state = carry

        keys = jax.random.split(rng_key, num_chains)
        new_states, info = jax.vmap(
            ghmc_kernel, in_axes=(0, 0, None, None, None, None, None)
        )(
            keys,
            states,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.position_sigma,
            adaptation_state.alpha,
            adaptation_state.delta,
        )
        new_adaptation_state = adapt_update(
            adaptation_state, new_states.position, new_states.logdensity_grad
        )

        return (new_states, new_adaptation_state), AdaptationInfo(
            new_states,
            info,
            new_adaptation_state,
        )

    def run(rng_key: PRNGKey, positions: PyTree, num_steps: int = 1000):
        key_init, key_adapt = jax.random.split(rng_key)

        rng_keys = jax.random.split(key_init, num_chains)
        init_states = batch_init(rng_keys, positions)
        init_adaptation_state = adapt_init(positions, init_states.logdensity_grad)

        keys = jax.random.split(key_adapt, num_steps)
        (last_states, last_adaptation_state), info = jax.lax.scan(
            one_step, (init_states, init_adaptation_state), keys
        )

        parameters = {
            "step_size": last_adaptation_state.step_size,
            "momentum_inverse_scale": last_adaptation_state.position_sigma,
            "alpha": last_adaptation_state.alpha,
            "delta": last_adaptation_state.delta,
        }

        return AdaptationResults(last_states, parameters), info

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]


class additive_step_random_walk:
    """Implements the user interface for the Additive Step RMH

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rw = blackjax.additive_step_random_walk(logdensity_fn, random_step)
        state = rw.init(position)
        new_state, info = rw.step(rng_key, state)

    The specific case of a Gaussian `random_step` is already implemented, either with independent components
    when `covariance_matrix` is a one dimensional array or with dependent components if a two dimensional array:

    .. code::

        rw_gaussian = blackjax.additive_step_random_walk.normal_random_walk(logdensity_fn, covariance_matrix)
        state = rw_gaussian.init(position)
        new_state, info = rw_gaussian.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    random_step
        A Callable that takes a random number generator and the current state and produces a step,
        which will be added to the current position to obtain a new position. Must be symmetric
        to maintain detailed balance. This means that P(step|position) = P(-step | position+step)

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(blackjax.mcmc.random_walk.init)
    build_kernel = staticmethod(blackjax.mcmc.random_walk.build_additive_step)

    @classmethod
    def normal_random_walk(cls, logdensity_fn: Callable, sigma):
        """
        Parameters
        ----------
        logdensity_fn
            The log density probability density function from which we wish to sample.
        sigma
            The value of the covariance matrix of the gaussian proposal distribution.
        Returns
        -------
             A ``MCMCSamplingAlgorithm``.
        """
        return cls(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))

    def __new__(  # type: ignore[misc]
        cls, logdensity_fn: Callable, random_step: Callable
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, random_step)

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class rmh:
    """Implements the user interface for the RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logdensity_fn, proposal_generator)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_generator
        A Callable that takes a random number generator and the current state and produces a new proposal.
    proposal_logdensity_fn
        The logdensity function associated to the proposal_generator. If the generator is non-symmetric,
         P(x_t|x_t-1) is not equal to P(x_t-1|x_t), then this parameter must be not None in order to apply
         the Metropolis-Hastings correction for detailed balance.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(blackjax.mcmc.random_walk.init)
    build_kernel = staticmethod(blackjax.mcmc.random_walk.build_rmh)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        proposal_generator: Callable[[PRNGKey, PyTree], PyTree],
        proposal_logdensity_fn: Optional[Callable[[PyTree], PyTree]] = None,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                proposal_generator,
                proposal_logdensity_fn,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class irmh:
    """Implements the (basic) user interface for the independent RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_distribution
        A Callable that takes a random number generator and produces a new proposal. The
        proposal is independent of the sampler's current state.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.

    """

    init = staticmethod(blackjax.mcmc.random_walk.init)
    build_kernel = staticmethod(blackjax.mcmc.random_walk.build_irmh)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        proposal_distribution: Callable,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, proposal_distribution)

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class orbital_hmc:
    """Implements the (basic) user interface for the Periodic orbital MCMC kernel.

    Each iteration of the periodic orbital MCMC outputs ``period`` weighted samples from
    a single Hamiltonian orbit connecting the previous sample and momentum (latent) variable
    with precision matrix ``inverse_mass_matrix``, evaluated using the ``bijection`` as an
    integrator with discretization parameter ``step_size``.

    Examples
    --------

    A new Periodic orbital MCMC kernel can be initialized and used with the following code:

    .. code::

        per_orbit = blackjax.orbital_hmc(logdensity_fn, step_size, inverse_mass_matrix, period)
        state = per_orbit.init(position)
        new_state, info = per_orbit.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(per_orbit.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The logarithm of the probability density function we wish to draw samples from.
    step_size
        The value to use for the step size in for the symplectic integrator to buid the orbit.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    period
        The number of steps used to build the orbit.
    bijection
        (algorithm parameter) The symplectic integrator to use to build the orbit.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(mcmc.periodic_orbital.init)
    build_kernel = staticmethod(mcmc.periodic_orbital.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,  # assume momentum is always Gaussian
        period: int,
        *,
        bijection: Callable = mcmc.integrators.velocity_verlet,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(bijection)

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn, period)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                period,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class elliptical_slice:
    """Implements the (basic) user interface for the Elliptical Slice sampling kernel.

    Examples
    --------

    A new Elliptical Slice sampling kernel can be initialized and used with the following code:

    .. code::

        ellip_slice = blackjax.elliptical_slice(loglikelihood_fn, cov_matrix)
        state = ellip_slice.init(position)
        new_state, info = ellip_slice.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(ellip_slice.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    loglikelihood_fn
        Only the log likelihood function from the posterior distributon we wish to sample.
    cov_matrix
        The value of the covariance matrix of the gaussian prior distribution from the posterior we wish to sample.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(mcmc.elliptical_slice.init)
    build_kernel = staticmethod(mcmc.elliptical_slice.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        loglikelihood_fn: Callable,
        *,
        mean: Array,
        cov: Array,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(cov, mean)

        def init_fn(position: PyTree):
            return cls.init(position, loglikelihood_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                loglikelihood_fn,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


class ghmc:
    """Implements the (basic) user interface for the Generalized HMC kernel.

    The Generalized HMC kernel performs a similar procedure to the standard HMC
    kernel with the difference of a persistent momentum variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step.

    This means that the sampling of the momentum variable depends on the previous
    momentum, the rate of persistence depends on the alpha parameter, and that the
    Metropolis-Hastings accept/reject step is done through slice sampling with a
    non-reversible slice variable also dependent on the previous slice, the determinisitc
    transformation is defined by the delta parameter.

    The Generalized HMC does not have a trajectory length parameter, it always performs
    one iteration of the velocity verlet integrator with a given step size, making
    the algorithm a good candiate for running many chains in parallel.

    Examples
    --------

    A new Generalized HMC kernel can be initialized and used with the following code:

    .. code::

        ghmc_kernel = blackjax.ghmc(logdensity_fn, step_size, alpha, delta)
        state = ghmc_kernel.init(rng_key, position)
        new_state, info = ghmc_kernel.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(ghmc_kernel.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        A PyTree of the same structure as the target PyTree (position) with the
        values used for as a step size for each dimension of the target space in
        the velocity verlet integrator.
    alpha
        The value defining the persistence of the momentum variable.
    delta
        The value defining the deterministic translation of the slice variable.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    noise_gn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(mcmc.ghmc.init)
    build_kernel = staticmethod(mcmc.ghmc.build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        momentum_inverse_scale: PyTree,
        alpha: float,
        delta: float,
        *,
        divergence_threshold: int = 1000,
        noise_gn: Callable = lambda _: 0.0,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(noise_gn, divergence_threshold)

        def init_fn(position: PyTree, rng_key: PRNGKey):
            return cls.init(rng_key, position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                momentum_inverse_scale,
                alpha,
                delta,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#                           VARIATIONAL INFERENCE
# -----------------------------------------------------------------------------


class PathFinderAlgorithm(NamedTuple):
    approximate: Callable
    sample: Callable


class pathfinder:
    """Implements the (basic) user interface for the pathfinder kernel.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.
    Pathfinder returns draws from the approximation with the lowest estimated
    Kullback-Leibler (KL) divergence to the true posterior.

    Note: all the heavy processing in performed in the init function, step
    function is just a drawing a sample from a normal distribution

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density of the model we want
        to sample from.

    Returns
    -------
    A ``VISamplingAlgorithm``.

    """

    approximate = staticmethod(vi.pathfinder.approximate)
    sample = staticmethod(vi.pathfinder.sample)

    def __new__(cls, logdensity_fn: Callable) -> PathFinderAlgorithm:  # type: ignore[misc]
        def approximate_fn(
            rng_key: PRNGKey,
            position: PyTree,
            num_samples: int = 200,
            **lbfgs_parameters,
        ):
            return cls.approximate(
                rng_key, logdensity_fn, position, num_samples, **lbfgs_parameters
            )

        def sample_fn(
            rng_key: PRNGKey, state: vi.pathfinder.PathfinderState, num_samples: int
        ):
            return cls.sample(rng_key, state, num_samples)

        return PathFinderAlgorithm(approximate_fn, sample_fn)


def pathfinder_adaptation(
    algorithm: Union[hmc, nuts],
    logdensity_fn: Callable,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to sample.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the
    tuned parameter values from an initial state.

    """

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_update, adapt_final = adaptation.pathfinder_adaptation.base(
        target_acceptance_rate,
    )

    def one_step(carry, rng_key):
        state, adaptation_state = carry
        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_update(
            adaptation_state, new_state.position, info.acceptance_rate
        )
        return (
            (new_state, new_adaptation_state),
            AdaptationInfo(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: PyTree, num_steps: int = 400):
        init_key, sample_key, rng_key = jax.random.split(rng_key, 3)

        pathfinder_state, _ = vi.pathfinder.approximate(
            init_key, logdensity_fn, position
        )
        init_warmup_state = adapt_init(
            pathfinder_state.alpha,
            pathfinder_state.beta,
            pathfinder_state.gamma,
            initial_step_size,
        )

        init_position, _ = vi.pathfinder.sample(sample_key, pathfinder_state)
        init_state = algorithm.init(init_position, logdensity_fn)

        keys = jax.random.split(rng_key, num_steps)
        last_state, info = jax.lax.scan(
            one_step,
            (init_state, init_warmup_state),
            keys,
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return AdaptationResults(last_chain_state, parameters), info

    return AdaptationAlgorithm(run)


class meanfield_vi:
    """High-level implementation of Mean-Field Variational Inference.

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density function associated with
        the distribution we want to sample from.
    optimizer
        Optax optimizer to use to optimize the ELBO.
    num_samples
        Number of samples to take at each step to optimize the ELBO.

    Returns
    -------
    A ``VIAlgorithm``.

    """

    init = staticmethod(vi.meanfield_vi.init)
    step = staticmethod(vi.meanfield_vi.step)
    sample = staticmethod(vi.meanfield_vi.sample)

    def __new__(
        cls,
        logdensity_fn: Callable,
        optimizer: GradientTransformation,
        num_samples: int = 100,
    ):  # type: ignore[misc]
        def init_fn(position: PyTree):
            return cls.init(position, optimizer)

        def step_fn(
            rng_key: PRNGKey, state: vi.meanfield_vi.MFVIState
        ) -> Tuple[vi.meanfield_vi.MFVIState, vi.meanfield_vi.MFVIInfo]:
            return cls.step(rng_key, state, logdensity_fn, optimizer, num_samples)

        def sample_fn(
            rng_key: PRNGKey, state: vi.meanfield_vi.MFVIState, num_samples: int
        ):
            return cls.sample(rng_key, state, num_samples)

        return VIAlgorithm(init_fn, step_fn, sample_fn)
