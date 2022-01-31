import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import initialize_model
from numpyro.infer import MCMC, NUTS
import pandas as pd

import blackjax
import blackjax.diagnostics as diagnostics


GLOBAL = {"count": 0}
num_warmup_steps = 1_000_000
num_sampling_steps = 1_000_000


# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# Eight Schools example - Non-centered Reparametrization
def eight_schools_noncentered(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
                ),
            )
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


rng_key = jax.random.PRNGKey(0)

init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    eight_schools_noncentered,
    model_args=(J, sigma, y),
    dynamic_args=True,
)

initial_position = init_params.z


def logprob(position):
    GLOBAL["count"] += 1
    return -potential_fn_gen(J, sigma, y)(position)


warmup_key, inference_key = jax.random.split(rng_key, 2)

warmup = blackjax.window_adaptation(
    algorithm=blackjax.nuts,
    logprob_fn=logprob,
    num_steps=num_warmup_steps,
    target_acceptance_rate=0.8,
)

tic1 = pd.Timestamp.now()
state, kernel, _ = warmup.run(warmup_key, initial_position)


def inference_loop(kernel, num_samples, rng_key, initial_state):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


states = inference_loop(kernel, num_sampling_steps, inference_key, state)
tic2 = pd.Timestamp.now()
print("Runtime for Blackjax's sampling", tic2 - tic1)
print(f"Compiled the logprob {GLOBAL['count']} time")


#
# NUMPYRO
#

tic1 = pd.Timestamp.now()
nuts_kernel = NUTS(eight_schools_noncentered)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=num_warmup_steps,
    num_samples=num_sampling_steps,
    progress_bar=False,
)
mcmc.run(rng_key, J, sigma, y=y)
samples = mcmc.get_samples()
tic2 = pd.Timestamp.now()
print("Runtime for numpyro's NUTS warmup + sampling", tic2 - tic1)
