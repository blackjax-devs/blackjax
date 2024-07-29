import jax.lax
import jax
import jax.numpy as jnp
import functools


def update_waste_free(mcmc_init_fn,
                      logposterior_fn,
                      mcmc_step_fn,
                      n_particles: int,
                      p: int,
                      num_resampled,
                      num_mcmc_steps):
    """
    Given M particles, mutates them using p-1 steps. Returns M*P-1 particles,
    consistent of the initial plus all the intermediate steps, thus implementing a
    waste-free update function
    See Algorithm 2: https://arxiv.org/abs/2011.02328
    """
    if num_mcmc_steps is not None:
        raise ValueError("Can't use waste free SMC with a num_mcmc_steps parameter")

    num_mcmc_steps = p-1

    def mcmc_kernel(rng_key, position, step_parameters):
        state = mcmc_init_fn(position, logposterior_fn)

        def body_fn(state, rng_key):
            new_state, info = mcmc_step_fn(
                rng_key, state, logposterior_fn, **step_parameters
            )
            return new_state, (new_state, info)

        _, (states, infos) = jax.lax.scan(body_fn, state, jax.random.split(rng_key, num_mcmc_steps))
        return states, infos

    def gather(rng_key, position, step_parameters):
        states, infos= jax.vmap(mcmc_kernel)(rng_key, position, step_parameters)
        step_particles = jax.tree.map(lambda x: x.reshape((num_resampled * num_mcmc_steps)), states.position)
        initial_particles = jax.tree.map(lambda x: x.reshape((num_resampled,)), position)
        new_particles = jax.tree.map(lambda x,y: jax.numpy.hstack([x,y]), initial_particles, step_particles)
        return new_particles, None

    return gather, num_resampled

def waste_free_smc(n_particles, p):
    return functools.partial(update_waste_free, num_resampled=int(n_particles / p), p=p)
