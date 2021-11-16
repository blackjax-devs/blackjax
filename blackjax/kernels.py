from typing import Callable, Dict, Union

import jax
import jax.numpy as jnp

import blackjax.adaptation as adaptation
import blackjax.mcmc as mcmc
import blackjax.smc as smc
from blackjax.base import AdaptationAlgorithm, SamplingAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, PRNGKey, PyTree

__all__ = [
    "adaptive_tempered_smc",
    "hmc",
    "mala",
    "nuts",
    "orbital_hmc",
    "rmh",
    "tempered_smc",
    "window_adaptation",
]


# -----------------------------------------------------------------------------
#                           SEQUENTIAL MONTE CARLO
# -----------------------------------------------------------------------------


class adaptive_tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(smc.tempered.init)
    kernel = staticmethod(smc.adaptive_tempered.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_algorithm: SamplingAlgorithm,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        target_ess: float,
        root_solver: Callable = smc.solver.dichotomy,
        use_log_ess: bool = True,
        mcmc_iter: int = 10,
    ) -> SamplingAlgorithm:
        def kernel_factory(logprob_fn):
            return mcmc_algorithm(logprob_fn, **mcmc_parameters).step

        step = cls.kernel(
            logprior_fn,
            loglikelihood_fn,
            kernel_factory,
            mcmc_algorithm.init,
            resampling_fn,
            target_ess,
            root_solver,
            use_log_ess,
            mcmc_iter,
        )

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.


    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(smc.tempered.init)
    kernel = staticmethod(smc.tempered.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_algorithm: SamplingAlgorithm,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        mcmc_iter: int = 10,
    ) -> SamplingAlgorithm:
        def kernel_factory(logprob_fn):
            return mcmc_algorithm(logprob_fn, **mcmc_parameters).step

        step = cls.kernel(
            logprior_fn,
            loglikelihood_fn,
            kernel_factory,
            mcmc_algorithm.init,
            resampling_fn,
            mcmc_iter,
        )

        def init_fn(position: PyTree):
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state, lmbda):
            return step(
                rng_key,
                state,
                lmbda,
            )

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#                         MARKOV CHAIN MONTE CARLO
# -----------------------------------------------------------------------------


class hmc:
    """Implements the (basic) user interface for the HMC kernel.

    The general hmc kernel (:meth:`blackjax.mcmc.hmc.kernel`, alias `blackjax.hmc.kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new HMC kernel can be initialized and used with the following code:

    .. code::

        hmc = blackjax.hmc(logprob_fn step_size, inverse_mass_matrix, num_integration_steps)
        state = hmc.init(position)
        new_state, info = hmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(hmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.hmc.kernel(integrators.mclachlan)
       state = blackjax.hmc.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size, inverse_mass_matrix, num_integration_steps)

    Parameters
    ----------
    logprob_fn
        The logprobability density function we wish to draw samples from. This
        is minus the potential function.
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
    A ``SamplingAlgorithm``.
    """

    init = staticmethod(mcmc.hmc.init)
    kernel = staticmethod(mcmc.hmc.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = mcmc.integrators.velocity_verlet,
    ) -> SamplingAlgorithm:

        step = cls.kernel(integrator, divergence_threshold)

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class mala:
    """Implements the (basic) user interface for the MALA kernel.

    The general mala kernel (:meth:`blackjax.mcmc.mala.kernel`, alias `blackjax.mala.kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new MALA kernel can be initialized and used with the following code:

    .. code::

        mala = blackjax.mala(logprob_fn, step_size)
        state = mala.init(position)
        new_state, info = mala.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(mala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.mala.kernel(logprob_fn)
       state = blackjax.mala.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size)

    Parameters
    ----------
    logprob_fn
        The logprobability density function we wish to draw samples from. This
        is minus the potential function.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(mcmc.mala.init)
    kernel = staticmethod(mcmc.mala.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
    ) -> SamplingAlgorithm:

        step = cls.kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(rng_key, state, logprob_fn, step_size)

        return SamplingAlgorithm(init_fn, step_fn)


class nuts:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code::

        nuts = blackjax.nuts(logprob_fn step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code::

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    You can always use the base kernel should you need to:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.nuts.kernel(integrators.yoshida)
       state = blackjax.nuts.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size, inverse_mass_matrix)

    Parameters
    ----------
    logprob_fn
        The logprobability density function we wish to draw samples from. This
        is minus the potential function.
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
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(mcmc.hmc.init)
    kernel = staticmethod(mcmc.nuts.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        *,
        max_num_doublings: int = 10,
        divergence_threshold: int = 1000,
        integrator: Callable = mcmc.integrators.velocity_verlet,
    ) -> SamplingAlgorithm:

        step = cls.kernel(integrator, divergence_threshold, max_num_doublings)

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
            )

        return SamplingAlgorithm(init_fn, step_fn)


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logprob_fn: Callable,
    num_steps: int = 1000,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
    progress_bar: bool = False,
    **parameters,
) -> AdaptationAlgorithm:
    """Adapt the parameters of algorithms in the HMC family.

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
    logprob_fn
        The log density probability density function from which we wish to sample.
    num_steps
        The number of adaptation steps.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    **parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the tuned parameter values from an initial state.

    """

    kernel = algorithm.kernel()

    def kernel_factory(step_size: float, inverse_mass_matrix: Array):
        def kernel_fn(rng_key, state):
            return kernel(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                **parameters,
            )

        return kernel_fn

    schedule = adaptation.window_adaptation.schedule(num_steps)
    init, update, final = adaptation.window_adaptation.base(
        kernel_factory,
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry
        state, adaptation_state, info = update(
            rng_key, state, adaptation_state, adaptation_stage
        )
        return ((state, adaptation_state), (state, info, adaptation_state))

    def run(rng_key: PRNGKey, position: PyTree):
        init_state = algorithm.init(position, logprob_fn)
        init_warmup_state = init(init_state, initial_step_size)

        if progress_bar:
            print("Running window adaptation")
            one_step_ = jax.jit(progress_bar_scan(num_steps)(one_step))
        else:
            one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_steps)
        last_state, warmup_chain = jax.lax.scan(
            one_step_,
            (init_state, init_warmup_state),
            (jnp.arange(num_steps), keys, schedule),
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = final(last_warmup_state)
        kernel = kernel_factory(step_size, inverse_mass_matrix)

        return last_chain_state, kernel, warmup_chain

    return AdaptationAlgorithm(run)


class rmh:
    """Implements the (basic) user interface for the gaussian random walk kernel

    Examples
    --------

    A new Gaussian Random Walk kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logprob_fn sigma)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logprob_fn
        The log density probability density function from which we wish to sample.
    sigma
        The value of the covariance matrix of the gaussian proposal distribution.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(mcmc.rmh.init)
    kernel = staticmethod(mcmc.rmh.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        sigma: Array,
    ) -> SamplingAlgorithm:

        step = cls.kernel()

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
                logprob_fn,
                sigma,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class orbital_hmc:
    """Implements the (basic) user interface for the Periodic orbital MCMC kernel

    Each iteration of the periodic orbital MCMC outputs ``period`` weighted samples from
    a single Hamiltonian orbit connecting the previous sample and momentum (latent) variable
    with precision matrix ``inverse_mass_matrix``, evaluated using the ``bijection`` as an
    integrator with discretization parameter ``step_size``.

    Examples
    --------

    A new Periodic orbital MCMC kernel can be initialized and used with the following code:

    .. code::

        per_orbit = blackjax.orbital_hmc(logprob_fn, step_size, inverse_mass_matrix, period)
        state = per_orbit.init(position)
        new_state, info = per_orbit.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(per_orbit.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logprob_fn
        The logarithm of the probability density function we wish to draw samples from. This
        is minus the potential energy function.
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
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(mcmc.periodic_orbital.init)
    kernel = staticmethod(mcmc.periodic_orbital.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,  # assume momentum is always Gaussian
        period: int,
        *,
        bijection: Callable = mcmc.integrators.velocity_verlet,
    ) -> SamplingAlgorithm:

        step = cls.kernel(bijection)

        def init_fn(position: PyTree):
            return cls.init(position, logprob_fn, period)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                period,
            )

        return SamplingAlgorithm(init_fn, step_fn)
