"""
strategies to tune the parameters of mcmc kernels
used within SMC, based on particles.
"""
import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = [
    "particles_means",
    "particles_stds",
    "particles_covariance_matrix",
    "mass_matrix_from_particles",
]


def particles_stds(particles):
    return jax.numpy.std(particles_as_rows(particles), axis=0)


def particles_means(particles):
    return jax.numpy.mean(particles_as_rows(particles), axis=0)


def particles_covariance_matrix(particles):
    return jax.numpy.cov(particles_as_rows(particles), ddof=0, rowvar=False)


def mass_matrix_from_particles(particles) -> Array:
    """
    Implements tuning from section 3.1 from https://arxiv.org/pdf/1808.07730.pdf
    Computing a mass matrix to be used in HMC from particles.
    Given the particles covariance matrix, set all non-diagonal elements as zero,
     take the inverse, and keep the diagonal.
    Returns
    -------
    A mass Matrix
    """
    stds = jax.numpy.std(particles_as_rows(particles), axis=0)
    return jnp.diag(jnp.atleast_1d(jnp.reciprocal(jnp.square(stds))))


def particles_as_rows(particles):
    """
    Adds end dimension for single-dimension variables, and then represents multivariables
    as a matrix where each column is a variable, each row a particle.
    """
    particles = jax.tree_util.tree_map(lambda x: jnp.atleast_2d(x.T).T, particles)
    return jnp.array(jax.numpy.hstack(jax.tree_util.tree_flatten(particles)[0]))
