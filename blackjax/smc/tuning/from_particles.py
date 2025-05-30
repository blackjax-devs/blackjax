"""
static (all particles get the same value) strategies to tune the parameters of mcmc kernels
used within SMC, based on particles.
"""
import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree

from blackjax.types import Array

__all__ = [
    "particles_means",
    "particles_stds",
    "particles_covariance_matrix",
    "inverse_mass_matrix_from_particles",
]


def particles_stds(particles):
    return jnp.std(particles_as_rows(particles), axis=0)


def particles_means(particles):
    return jnp.mean(particles_as_rows(particles), axis=0)


def particles_covariance_matrix(particles):
    return jnp.cov(particles_as_rows(particles), ddof=0, rowvar=False)


def inverse_mass_matrix_from_particles(particles) -> Array:
    """
    Implements tuning from section 3.1 from https://arxiv.org/pdf/1808.07730.pdf
    Computing an inverse mass matrix to be used in HMC from particles.

    Returns
    -------
    An inverse mass matrix
    """
    return jnp.diag(jnp.var(particles_as_rows(particles), axis=0))


def particles_as_rows(particles):
    """
    Adds end dimension for single-dimension variables, and then represents multivariables
    as a matrix where each column is a variable, each row a particle.
    """
    return jax.vmap(lambda x: ravel_pytree(x)[0])(particles)
