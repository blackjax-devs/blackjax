from typing import NamedTuple
from chex import Array
import jax
import jax.numpy as jnp
from scipy.fftpack import next_fast_len  # type: ignore

import blackjax
from blackjax.adaptation.step_size import tune
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import minimal_norm
from blackjax.mcmc.mclmc import MCLMCState, build_kernel
# from blackjax.diagnostics import effective_sample_size
from blackjax.types import PRNGKey



def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))


def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    # keys = jax.random.split(rng_key, num_steps)
    keys = jnp.array([jax.random.PRNGKey(0)]*num_steps)
    state = sampling_algorithm.init(initial_val)
    print("\n\n", state.position, "\n\n")
    print("\n\n", state.momentum, "\n\n")
    _, info = jax.lax.scan(
        lambda s, k: (sampling_algorithm.step(k, s)), state, keys
    )
    return info


key = jax.random.PRNGKey(0)
main_key, tune_key = jax.random.split(key)


gr = jax.value_and_grad(logdensity_fn)
kernel = build_kernel(grad_logp=gr, 
                                        integrator=minimal_norm, transform=lambda x:x)


L, eps, state = (tune(kernel, num_steps=100, rng_key=tune_key))
print("L, eps post tuning", L, eps)
raise Exception
mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    transform=lambda x: x,
    # L=0.56568545, step_size=1.4142135, inverse_mass_matrix=jnp.array([1.0, 1.0]
    step_size=0.56568545, L=1.4142135,
)


out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=100,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=main_key,
)

print(jnp.mean(out.transformed_x, axis=0))

# print(logdensity_fn(jnp.array([0.1, 0.1])))
# print(out)

assert jnp.array_equal(jnp.mean(out.transformed_x, axis=0), [-1.2130139,  1.5367734])


