from typing import Callable, Tuple

import jax

from blackjax.base import SamplingAlgorithm
from blackjax.hmc_base import HMCInfo, HMCState, hmc_init, hmc_kernel
from blackjax.inference.hmc import integrators
from blackjax.nuts_base import NUTSInfo, nuts_kernel
from blackjax.rmh_base import RMHInfo, RMHState, rmh_init, rmh_kernel
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["hmc", "nuts", "rmh"]


class hmc:
    """Implements the (basic) user interface for the HMC kernel.

    The general hmc kernel (:meth:`blackjax.hmc_base.hmc_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel and jit-compiles it specifying the static
    arguments.

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

    new_kernel = hmc_kernel
    init = hmc_init

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        integrator: Callable = integrators.velocity_verlet,
        divergence_threshold: int = 1000,
    ) -> SamplingAlgorithm:

        kernel = cls.new_kernel(integrator, divergence_threshold)

        def init_fn(position: PyTree):
            return jax.jit(cls.init, static_argnums=(1,))(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state: HMCState) -> Tuple[HMCState, HMCInfo]:
            # `np.ndarray` and `DeviceArray`s are not hashable and thus cannot be used as static arguments.`
            # Workaround: https://github.com/google/jax/issues/4572#issuecomment-709809897
            kernel_fn = jax.jit(
                kernel,
                static_argnames=["logprob_fn", "num_integration_steps"],
            )
            return kernel_fn(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class nuts:
    """Implements the (basic) user interface for the nuts kernel"""

    new_kernel = nuts_kernel
    init = hmc_init

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        *,
        integrator: Callable = integrators.velocity_verlet,
        divergence_threshold: int = 1000,
        max_num_doublings: int = 10,
    ) -> SamplingAlgorithm:

        kernel = cls.new_kernel(integrator, divergence_threshold, max_num_doublings)

        def init_fn(position: PyTree):
            return jax.jit(cls.init, static_argnums=(1,))(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state: HMCState) -> Tuple[HMCState, NUTSInfo]:
            # `np.ndarray` and `DeviceArray`s are not hashable and thus cannot be used as static arguments.`
            # Workaround: https://github.com/google/jax/issues/4572#issuecomment-709809897
            kernel_fn = jax.jit(
                kernel,
                static_argnames=["logprob_fn"],
            )
            return kernel_fn(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
            )

        return SamplingAlgorithm(init_fn, step_fn)


class rmh:
    """Implements the (basic) user interface for the gaussian random walk kernel"""

    new_kernel = rmh_kernel
    init = rmh_init

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        sigma: Array,
    ) -> SamplingAlgorithm:

        kernel = cls.new_kernel()

        def init_fn(position: PyTree):
            return jax.jit(cls.init, static_argnums=(1,))(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state: RMHState) -> Tuple[RMHState, RMHInfo]:
            # `np.ndarray` and `DeviceArray`s are not hashable and thus cannot be used as static arguments.`
            # Workaround: https://github.com/google/jax/issues/4572#issuecomment-709809897
            kernel_fn = jax.jit(
                kernel,
                static_argnames=["logprob_fn"],
            )
            return kernel_fn(
                rng_key,
                state,
                logprob_fn,
                sigma,
            )

        return SamplingAlgorithm(init_fn, step_fn)
