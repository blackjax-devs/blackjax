from typing import Callable, Union

import jax

from blackjax.base import SamplingAlgorithm, AdaptationAlgorithm
import blackjax.adaptation as adaptation
import blackjax.mcmc as mcmc
import blackjax.smc as smc
from blackjax.types import Array, PRNGKey, PyTree

__all__ = [
    "hmc",
    "nuts",
    "window_adaptation",
    "rmh",
    "adaptive_tempered_smc",
    "tempered_smc",
]


# -----------------------------------------------------------------------------
#                           SEQUENTIAL MONTE CARLO
# -----------------------------------------------------------------------------


class tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel."""

    init = staticmethod(smc.tempered.init)
    kernel = staticmethod(smc.tempered.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_kernel_factory: Callable,
        make_mcmc_state: Callable,
        resampling_fn: Callable,
        mcmc_iter: int = 10,
    ) -> SamplingAlgorithm:

        step = cls.kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_kernel_factory,
            make_mcmc_state,
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

        return SamplingAlgorithm(init_fn, step_fn)


class adaptive_tempered_smc:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel."""

    init = staticmethod(smc.tempered.init)
    kernel = staticmethod(smc.adaptive_tempered.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_kernel_factory: Callable,
        make_mcmc_state: Callable,
        resampling_fn: Callable,
        target_ess: float,
        root_solver: Callable = smc.solver.dichotomy,
        use_log_ess: bool = True,
        mcmc_iter: int = 10,
    ) -> SamplingAlgorithm:

        step = cls.kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_kernel_factory,
            make_mcmc_state,
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


# -----------------------------------------------------------------------------
#                         MARKOV CHAIN MONTE CARLO
# -----------------------------------------------------------------------------


class hmc:
    """Implements the (basic) user interface for the HMC kernel.

    The general hmc kernel (:meth:`blackjax.hmc_base.hmc_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    In addition, we add the general kernel as an attribute to this class so
    users only need to pass `blackjax.hmc` to the algorithm, and thus don't need
    to know about the existence of the base kernel.

    Examples
    --------

    A new HMC kernel can be initialized and used with the following code:

    .. code:

        hmc = blackjax.hmc(logprob_fn step_size, inverse_mass_matrix, num_integration_steps)
        state = hmc.init(position)
        new_state, info = hmc.step(rng_key, state)

    If we need to do something slightly fancier we can use the base kernel
    directly. Here if we want to use Yoshida's symplectic integrator instead of
    the usual velocity verlet:

    .. code:

       kernel = blackjax.hmc.new_kernel(integrators.yoshida)
       state = blackjax.hmc.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size, inverse_mass_matrix, num_integration_steps)

    """

    kernel = staticmethod(mcmc.hmc.kernel)
    init = staticmethod(mcmc.hmc.init)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        integrator: Callable = mcmc.integrators.velocity_verlet,
        divergence_threshold: int = 1000,
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


class nuts:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code:

        nuts = blackjax.nuts(logprob_fn step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code:

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    If we need to do something slightly fancier we can use the base kernel
    directly. Here if we want to use Yoshida's symplectic integrator instead of
    the usual velocity verlet:

    .. code:

       state = blackjax.nuts.init(position, logprob_fn)
       kernel = blackjax.nuts.new_kernel(integrators.yoshida)
       state, info = kernel(rng_key, state, logprob_fn, step_size, inverse_mass_matrix, num_integration_steps)

    """

    kernel = staticmethod(mcmc.nuts.kernel)
    init = staticmethod(mcmc.hmc.init)

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        *,
        integrator: Callable = mcmc.integrators.velocity_verlet,
        divergence_threshold: int = 1000,
        max_num_doublings: int = 10,
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
    **parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the
    tuned parameter values from an initial state.

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

    schedule_fn = adaptation.window_adaptation.schedule(num_steps)
    init, update, final = adaptation.window_adaptation.base(
        kernel_factory,
        schedule_fn,
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    @jax.jit
    def one_step(carry, rng_key):
        state, warmup_state = carry
        state, warmup_state, info = update(rng_key, state, warmup_state)
        return ((state, warmup_state), (state, warmup_state, info))

    def run(rng_key: PRNGKey, position: PyTree):
        init_state = algorithm.init(position, logprob_fn)
        init_warmup_state = init(init_state, initial_step_size)

        keys = jax.random.split(rng_key, num_steps)
        last_state, warmup_chain = jax.lax.scan(
            one_step,
            (init_state, init_warmup_state),
            keys,
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

    .. code:

        rmh = blackjax.rmh(logprob_fn sigma)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code:

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    """

    kernel = staticmethod(mcmc.rmh.kernel)
    init = staticmethod(mcmc.rmh.init)

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
