import jax
import jax.numpy as jnp
import blackjax
from blackjax.base import SamplingAlgorithm
import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.mclmc import Parameters


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x - 5))


# Initialize the state
initial_position = jnp.array([1.0, 1.0])


mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn, d=2, transform=lambda x: x
)

params = Parameters(
    L=0.56568545, step_size=1.4142135, inverse_mass_matrix=jnp.array([1.0, 1.0])
)

# ?
# tuning()


def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    keys = jax.random.split(rng_key, num_steps + 1)
    state = sampling_algorithm.init(initial_val, keys[0])
    _, info = jax.lax.scan(
        lambda s, k: (sampling_algorithm.step(k, s, params=params)), state, keys[1:]
    )
    return info


out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=10000,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=jax.random.PRNGKey(0),
)

print(jnp.mean(out.transformed_x, axis=0))

assert jnp.array_equal(jnp.mean(out.transformed_x, axis=0), [5.0048037, 5.0181437])
