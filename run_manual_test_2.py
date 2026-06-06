import jax
import jax.numpy as jnp
from blackjax.adaptation.window_adaptation import window_adaptation
import blackjax.mcmc.slingshot as slingshot
import blackjax.mcmc.integrators as integrators

def logdensity_fn(x): return -0.5 * jnp.sum(x**2)
# simulate window_adaptation calling build_kernel
mcmc_kernel = slingshot.build_kernel(integrators.velocity_verlet)
print("SUCCESS")
