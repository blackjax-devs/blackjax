import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import sys
import blackjax
from blackjax.base import SamplingAlgorithm
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import sys
import blackjax
from blackjax.mcmc.mclmc import Parameters

def logdensity_fn(x):
    return -0.5*jnp.sum(jnp.square(x-5))

# Build the kernel
inverse_mass_matrix = jnp.array([1.0, 1.0])

# Initialize the state
initial_position = jnp.array([1.0, 1.0])


mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    d=2,
    transform=lambda x: x,
    init_key=jax.random.PRNGKey(0),
    params=Parameters(0.56568545, 1.4142135, inverse_mass_matrix),
)

# ?
# tuning()

flip = lambda f: lambda s, k: f(k, s)

def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    state = sampling_algorithm.init(initial_val)
    keys = jax.random.split(rng_key, num_steps)
    _, info = jax.lax.scan(flip(sampling_algorithm.step), state, keys)
    return info

out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=10000,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=jax.random.PRNGKey(0),
)

print(jnp.mean(out.transformed_x, axis=0))

