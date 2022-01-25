from typing import Callable, Tuple

import jax

from blackjax.inference.hmc import integrators
from blackjax.hmc_base import hmc_kernel, hmc_init, HMCState, HMCInfo
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, PyTree, PRNGKey


class hmc(object):
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

       kernel = blackjax.hmc.kernel_gen(integrators.yoshida)
       state = blackjax.hmc.init(position, logprob_fn)
       state, info = kernel(rng_key, state, logprob_fn, step_size, inverse_mass_matrix, num_integration_steps)

    """

    kernel_gen = hmc_kernel
    init = hmc_init

    def __new__(
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        integrator: Callable = integrators.velocity_verlet,
        divergence_threshold: int = 1000,
    ) -> SamplingAlgorithm:

        kernel = cls.kernel_gen(integrator, divergence_threshold)

        def init_fn(position: PyTree):
            return jax.jit(cls.init, static_argnums=(1,))(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state: HMCState) -> Tuple[HMCState, HMCInfo]:
            # `np.ndarray` and `DeviceArray`s are not hashable and thus cannot be used as static arguments.`
            # Workaround: https://github.com/google/jax/issues/4572#issuecomment-709809897
            kernel_fn = jax.jit(
                kernel,
                static_argnames=["logprob_fn", "step_size", "num_integration_steps"],
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
