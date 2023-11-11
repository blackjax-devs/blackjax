import jax
import jax.numpy as jnp

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.mclmc import Parameters


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x - 5))


# Initialize the state
initial_position = jnp.array([1.0, 1.0])


dim = 2 

mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    dim=dim,
    transform=lambda x: x,
    params=Parameters(
        L=0.56568545, step_size=1.4142135, inverse_mass_matrix=jnp.array([1.0, 1.0])
    ),
)





def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    keys = jax.random.split(rng_key, num_steps + 1)
    state = sampling_algorithm.init(initial_val, keys[0])
    _, info = jax.lax.scan(
        lambda s, k: (sampling_algorithm.step(k, s)), state, keys[1:]
    )
    return info


# ?
# tuning()
num_steps = 10000
initial_params = Parameters(L=jnp.sqrt(dim),step_size=0.4*jnp.sqrt(dim), inverse_mass_matrix=jnp.array([1.0, 1.0]))
mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    dim=dim,
    transform=lambda x: x,
    params=initial_params
)
out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps= int(num_steps * 0.1),
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=jax.random.PRNGKey(0),
)
Lfactor = 0.4
ESS = effective_sample_size(out.transformed_x)
Lnew = Lfactor * initial_params.step_size / ESS # = 0.4 * correlation length
print(Lnew)
raise Exception

# def tune3(self, x, u, l, g, random_key, L, eps, sigma, num_steps):
#     """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
#     X, xx, uu, ll, gg, key = self.sample_full(num_steps, x, u, l, g, random_key, L, eps, sigma)
#     ESS = ess_corr(X)
#     Lnew = self.Lfactor * eps / ESS # = 0.4 * correlation length

#     return Lnew, xx, uu, ll, gg, key


out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=10000,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=jax.random.PRNGKey(0),
)

print(jnp.mean(out.transformed_x, axis=0))

assert jnp.array_equal(jnp.mean(out.transformed_x, axis=0), [5.0048037, 5.0181437])



