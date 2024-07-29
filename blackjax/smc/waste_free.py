def mutate_waste_free(  mcmc_init_fn,
                        tempered_logposterior_fn,
                         shared_mcmc_step_fn,
                         num_mcmc_steps):
    """
    Given N particles, runs num_mcmc_steps of a kernel starting at each particle, and
    returns the last values, waisting the previous num_mcmc_steps-1
    samples per chain.
    """
    def mcmc_kernel(rng_key, position, step_parameters):
        state = mcmc_init_fn(position, tempered_logposterior_fn)

        def body_fn(state, rng_key):
            new_state, info = shared_mcmc_step_fn(
                rng_key, state, tempered_logposterior_fn, **step_parameters
            )
            return new_state, info

        keys = jax.random.split(rng_key, num_mcmc_steps)
        last_state, info = jax.lax.scan(body_fn, state, keys)
        return last_state.position, info

    return  jax.vmap(mcmc_kernel)