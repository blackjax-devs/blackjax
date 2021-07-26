from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.inference.hmc.base import PyTree


class SMCInfo(NamedTuple):
    """Additional information on the tempered SMC step.

    weights: jnp.ndarray
        The weights after the MCMC pass.
    """

    weights: jnp.ndarray


def smc(
    mcmc_kernel_factory: Callable,
    new_mcmc_state: Callable,
    resampling_method: Callable,
    n_iter: int,
):
    """Build a generic SMC step that takes a mcmc_kernel and a potential function, propagate through it,
    corrects using the weights function and resamples the end result.
    In Feynman-Kac equivalent terms, the algo goes roughly as follows:
    ```
        M_t = mcmc_kernel_factory
        for i in range(n_iter):
            x_t^i = M_t(..., x_t^i)
        G_t = log_weights_fn
        log_weights = G_t(x_t)
        idx = resample(log_weights)
        x_t = x_t[idx]
    ```


    Parameters
    ----------
    mcmc_kernel_factory: Callable
        A function of the Markov potential that returns a mcmc_kernel
    new_mcmc_state: Callable
        How to create a new MCMC state from the SMC particles
    resampling_method: Callable
        A random function that resamples generated particles based of weights
    n_iter: int
        Number of iterations of the MCMC kernel

    Returns
    -------
    A callable that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: jnp.ndarray,
        particles: PyTree,
        potential_fn: Callable,
        log_weight_fn: Callable,
    ) -> Tuple[PyTree, SMCInfo]:
        """

        Parameters
        ----------
        rng_key: DeviceArray[int],
            JAX PRNGKey for randomness
        state: SMCState
            Current state of the tempered SMC algorithm
        potential_fn: Callable
            A function takes represents the potential of the Markov kernel at time t.
        log_weight_fn: Callable
            A function that represents the Feynman-Kac log potential at time t.

        Returns
        -------
        state: SMCState,
            The new state of the tempered SMC algorithm
         info: SMCInfo,
            Additional information on the SMC step
        """
        n_particles = jax.tree_flatten(particles)[0][0].shape[0]

        step_mcmc_kernel = jax.vmap(mcmc_kernel_factory(potential_fn), in_axes=[0, 0])
        rng_key, resampling_key = jax.random.split(rng_key, 2)

        def mcmc_loop_body(mcmc_state, step_key):
            mcmc_keys = jax.random.split(step_key, n_particles)
            mcmc_state, _ = step_mcmc_kernel(mcmc_keys, mcmc_state)
            return mcmc_state, mcmc_state

        initial_mcmc_states = jax.vmap(new_mcmc_state, in_axes=[0, None])(
            particles, potential_fn
        )
        scan_keys = jax.random.split(rng_key, n_iter)
        last_mcmc_state, _ = jax.lax.scan(
            mcmc_loop_body, initial_mcmc_states, scan_keys, length=n_iter
        )

        particles = last_mcmc_state.position
        log_weights = jax.vmap(log_weight_fn, in_axes=[0])(particles)
        weights = _normalize(log_weights)
        resampling_index = resampling_method(weights, resampling_key)
        particles = jax.tree_map(
            lambda x: x[resampling_index], last_mcmc_state.position
        )
        info = SMCInfo(weights)
        return particles, info

    return kernel


def _normalize(log_weights):
    """Normalize log-weights into weights"""
    w = jnp.exp(log_weights - jnp.max(log_weights))
    return w / w.sum()
