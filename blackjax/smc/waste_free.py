import functools

import jax
import jax.lax
import jax.numpy as jnp


def update_waste_free(
    mcmc_init_fn,
    logposterior_fn,
    mcmc_step_fn,
    n_particles: int,
    p: int,
    num_resampled,
    num_mcmc_steps=None,
):
    """
    Given M particles, mutates them using p-1 steps. Returns M*P-1 particles,
    consistent of the initial plus all the intermediate steps, thus implementing a
    waste-free update function
    See Algorithm 2: https://arxiv.org/abs/2011.02328
    """
    if num_mcmc_steps is not None:
        raise ValueError(
            "Can't use waste free SMC with a num_mcmc_steps parameter, set num_mcmc_steps = None"
        )

    num_mcmc_steps = p - 1

    def mcmc_kernel(rng_key, position, step_parameters):
        state = mcmc_init_fn(position, logposterior_fn)

        def body_fn(state, rng_key):
            new_state, info = mcmc_step_fn(
                rng_key, state, logposterior_fn, **step_parameters
            )
            return new_state, (new_state, info)

        _, (states, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(rng_key, num_mcmc_steps)
        )
        return states, infos

    def update(rng_key, position, step_parameters):
        """
        Given the initial particles, runs a chain starting at each.
        The combines the initial particles with all the particles generated
        at each step of each chain.
        """
        states, infos = jax.vmap(mcmc_kernel)(rng_key, position, step_parameters)

        # step particles is num_resmapled, num_mcmc_steps, dimension_of_variable
        # want to transformed into num_resampled * num_mcmc_steps, dimension of variable
        def reshape_step_particles(x):
            _num_resampled, num_mcmc_steps, *dimension_of_variable = x.shape
            return x.reshape((_num_resampled * num_mcmc_steps, *dimension_of_variable))

        step_particles = jax.tree.map(reshape_step_particles, states.position)
        new_particles = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]), position, step_particles
        )
        return new_particles, infos

    return update, num_resampled


def waste_free_smc(n_particles, p):
    if not n_particles % p == 0:
        raise ValueError("p must be a divider of n_particles ")
    return functools.partial(update_waste_free, num_resampled=int(n_particles / p), p=p)
