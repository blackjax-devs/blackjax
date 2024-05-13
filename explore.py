import jax
import jax.numpy as jnp

import blackjax
from blackjax.util import run_inference_algorithm

init_key, tune_key, run_key = jax.random.split(jax.random.PRNGKey(0), 3)


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))


initial_position = jnp.ones(
    10,
)


def run_mclmc(logdensity_fn, key, num_steps, initial_position):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    alg = blackjax.mclmc(logdensity_fn=logdensity_fn, L=0.5, step_size=0.1, std_mat=1.)

    average, states = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=alg,
        num_steps=num_steps,
        progress_bar=True,
        transform=lambda x: x.position,
        streaming=True,
    )

    print(average)

    _, states, _ = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=alg,
        num_steps=num_steps,
        progress_bar=False,
        transform=lambda x: x.position,
        streaming=False,
    )

    print(states.mean(axis=0))

    return states


# out = run_hmc(initial_position)
out = run_mclmc(
    logdensity_fn=logdensity_fn,
    num_steps=5,
    initial_position=initial_position,
    key=jax.random.PRNGKey(0),
)
