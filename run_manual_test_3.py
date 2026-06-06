import jax.numpy as jnp
import blackjax.mcmc.integrators as integrators

def build_kernel(step_size, inverse_mass_matrix=None, num_proposals=1000):
   if inverse_mass_matrix is None:
       local_cholesky = jnp.eye(dim)
   else:
       local_cholesky = jnp.linalg.cholesky(inverse_mass_matrix)
   return local_cholesky

build_kernel(integrators.velocity_verlet)
